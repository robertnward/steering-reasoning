import os
from dataclasses import asdict, dataclass
from typing import List

import pyrallis
import vllm
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import trange
from transformers import AutoTokenizer
from vllm import RequestOutput

from steering_reasoning.train.rl.policy_model import tokenize_example
from steering_reasoning.train.rl.trainer import is_answer_present
from steering_reasoning.utils.utils import set_logger


@dataclass
class Config:
    seed: int

    model_path: str
    dataset_name: str

    batch_size: int
    max_seq_length: int
    stop_token: str

    num_generations: int
    repetition_penalty: float
    top_p: float
    top_k: int
    generation_temperature: float

    template_type: str

    split_id: int
    total_splits: int

    savedir: str

    def __post_init__(self):
        assert self.split_id < self.total_splits, (
            f"split_id must be less than total_splits, instead {self.split_id} and {self.total_splits}"
        )
        self.savedir = os.path.join(
            self.savedir,
            os.path.basename(self.dataset_name),
            f"temp_{self.generation_temperature}_top_p_{self.top_p}",
            f"eval_seed-{self.seed}",
            f"split_{self.split_id}_{self.total_splits}",
        )
        os.makedirs(self.savedir, exist_ok=True)


@dataclass
class Entry:
    prompt_idx: int
    generation_idx: int
    prompt: str
    generation: str
    prompt_tokens: List[int]
    generation_tokens: List[int]
    reward: float


@pyrallis.wrap()
def run(config: Config):
    set_logger(verbose=False)
    set_seed(seed=config.seed, device_specific=False, deterministic=False)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    vllm_actor = vllm.LLM(
        model=config.model_path,
        trust_remote_code=True,
        seed=config.seed,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=config.max_seq_length,
        max_seq_len_to_capture=config.max_seq_length * 2,
        dtype="bfloat16",
        model_impl="vllm",
    )
    stop_token_id = tokenizer.encode(config.stop_token, add_special_tokens=False)
    assert len(stop_token_id) == 1, (
        f"Stop token must be a single token, instead {stop_token_id}"
    )
    stop_token_id = stop_token_id[0]

    sampling_params = vllm.SamplingParams(
        n=config.num_generations,
        repetition_penalty=config.repetition_penalty,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.generation_temperature,
        stop_token_ids=[stop_token_id],
        seed=config.seed,
        logprobs=0,
    )

    if config.dataset_name.endswith("parquet"):
        ds = load_dataset("parquet", data_files=config.dataset_name, split="train")
    else:
        ds = load_dataset(config.dataset_name)["train"]
    if "solution" not in ds:
        ds = ds.add_column("solution", ds["answer"])
    ds = ds.add_column("idx", list(range(len(ds))))
    ds = ds.filter(lambda x: len(x["answer"]) > 0)
    ds = ds.map(
        tokenize_example(
            tokenizer=tokenizer,
            stop_token=config.stop_token,
            max_seq_length=config.max_seq_length,
            template_type=config.template_type,
            append_to=False,
        ),
        batched=False,
        num_proc=None,
        remove_columns=["problem", "answer", "solution"],
    )
    ds = ds.filter(lambda x: len(x["prompt_attention_mask"]) <= config.max_seq_length)
    ds = ds.select(
        [x for x in range(len(ds)) if x % config.total_splits == config.split_id]
    )
    logger.info(
        f"Example from dataset_name: {[tokenizer.decode(x, skip_special_tokens=False) for x in ds[0]['prompt_input_ids']]}"
    )
    is_first_sample = True
    data = []
    for idx in trange(0, len(ds), config.batch_size, desc="Sample"):
        if (idx // config.batch_size) % 10 == 0:
            logger.info(f"Sample {idx}/{len(ds)}")

        ids = ds["idx"][idx : idx + config.batch_size]
        prompt_token_ids = ds["prompt_input_ids"][idx : idx + config.batch_size]
        prompt_attention_mask = ds["prompt_attention_mask"][
            idx : idx + config.batch_size
        ]
        answer_token_ids = ds["answer_input_ids"][idx : idx + config.batch_size]

        max_gen_tokens = [
            config.max_seq_length - len(prompt_attention_mask[i])
            for i in range(len(prompt_attention_mask))
        ]
        sampling_params.max_tokens = max(max_gen_tokens)

        request_outputs: List[RequestOutput] = vllm_actor.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for prompt_idx, request_output in enumerate(request_outputs):
            for generation_idx, generation in enumerate(request_output.outputs):
                generation_token_ids = generation.token_ids[
                    : max_gen_tokens[prompt_idx]
                ]
                if is_first_sample:
                    is_first_sample = False
                    logger.info(
                        f"Generation: {[tokenizer.decode(x, skip_special_tokens=False) for x in generation_token_ids]}"
                    )

                is_correct, _ = is_answer_present(
                    processing_class=tokenizer,
                    answer_token_ids=answer_token_ids[prompt_idx],
                    generation_token_ids=generation_token_ids,
                    template_type=config.template_type,
                )

                entry = Entry(
                    prompt_idx=ids[prompt_idx],
                    generation_idx=generation_idx,
                    prompt=tokenizer.decode(
                        prompt_token_ids[prompt_idx], skip_special_tokens=False
                    ),
                    generation=tokenizer.decode(
                        generation_token_ids, skip_special_tokens=False
                    ),
                    prompt_tokens=prompt_token_ids[prompt_idx],
                    generation_tokens=generation_token_ids,
                    reward=float(is_correct),
                )

                data.append(asdict(entry))

    data = Dataset.from_list(data)
    data.save_to_disk(config.savedir)


if __name__ == "__main__":
    run()
