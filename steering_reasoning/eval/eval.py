import os
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pyrallis
import torch
import vllm
from accelerate.utils import set_seed
from datasets import Value, concatenate_datasets, load_dataset
from loguru import logger
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from vllm import RequestOutput

from steering_reasoning.train.rl.policy_model import tokenize_example
from steering_reasoning.train.rl.trainer import is_answer_present
from steering_reasoning.utils.math import last_boxed_only_string, remove_boxed
from steering_reasoning.utils.utils import set_logger

device = "cuda" if torch.cuda.is_available else "cpu"


@dataclass
class Config:
    seed: int

    model_path: str
    adapter_path: Optional[str]
    datasets: List[str]
    small_datasets: List[str]

    model_backend: str

    batch_size: int
    gen_batch_size: int
    max_seq_length: int
    stop_token: str

    num_generations: int
    small_datasets_num_generations: int
    repetition_penalty: float
    top_p: float
    top_k: int
    generation_temperature: float

    template_type: str

    append_to: bool

    split_id: int
    total_splits: int

    savedir: str

    def __post_init__(self):
        if self.model_backend == "transformers":
            self.savedir = os.path.join(self.savedir, "transformers_backend")

        if self.append_to:
            self.savedir = os.path.join(
                self.savedir,
                f"append_to_{self.append_to}",
                f"temp_{self.generation_temperature}_top_p_{self.top_p}",
                f"eval_seed-{self.seed}",
            )
        else:
            self.savedir = os.path.join(
                self.savedir,
                f"temp_{self.generation_temperature}_top_p_{self.top_p}",
                f"eval_seed-{self.seed}",
            )

        if self.generation_temperature == 0.0:
            self.num_generations = 1
            self.small_datasets_num_generations = 1

        assert self.template_type in ["r1", "qwen_math"], self.template_type
        assert self.model_backend in ["vllm", "transformers"], self.model_backend

        os.makedirs(self.savedir, exist_ok=True)


@dataclass
class Solution:
    gen_length: int
    reward: float
    logprob: float
    finished_with_stop_token: bool
    generation_text: Optional[str] = None


@dataclass
class Problem:
    problem: str
    answer: str
    solutions: List[Solution]
    ds_idx: Optional[int] = None


def get_ds(dataset_name: str):
    #   - Maxwell-Jia/AIME_2024
    #   - opencompass/AIME2025
    #   - knoveleng/AMC-23
    #   - knoveleng/Minerva-Math
    #   - knoveleng/OlympiadBench
    #   - HuggingFaceH4/MATH-500
    if "AIME_2024" in dataset_name:
        ds = load_dataset(dataset_name)["train"]
        ds = (
            ds.rename_column("Problem", "problem")
            .rename_column("Answer", "answer")
            .rename_column("Solution", "solution")
        )
        ds = ds.remove_columns(["ID"])
    elif "AIME2025" in dataset_name:
        ds = concatenate_datasets(
            [
                load_dataset(dataset_name, "AIME2025-I")["test"],
                load_dataset(dataset_name, "AIME2025-II")["test"],
            ]
        )
        ds = ds.rename_column("question", "problem")
        ds = ds.map(lambda x: {"solution": ""})
    elif "AMC-23" in dataset_name:
        ds = load_dataset(dataset_name)["train"]
        ds = ds.remove_columns(["id", "url", "question"])
        ds = ds.map(lambda x: {"solution": ""})
    elif "Minerva-Math" in dataset_name:
        ds = load_dataset(dataset_name)["train"]
        ds = ds.map(
            lambda x: {"answer": remove_boxed(last_boxed_only_string(x["solution"]))}
        )
        ds = ds.remove_columns(["type", "idx"])
    elif "OlympiadBench" in dataset_name:
        ds = load_dataset(dataset_name)["train"]
        ds = ds.rename_column("question", "problem")
        ds = ds.map(lambda x: {"solution": x["solution"][0]})
        ds = ds.remove_columns(
            [
                "id",
                "subfield",
                "context",
                "final_answer",
                "unit",
                "answer_type",
                "error",
                "is_multiple_answer",
            ]
        )
    elif "MATH-500" in dataset_name:
        ds = load_dataset(dataset_name)["test"]
        ds = ds.map(lambda x: {"answer": x["answer"]})
        ds = ds.remove_columns(["subject", "level", "unique_id"])
    else:
        ds = load_dataset(dataset_name)

    def checker(example):
        if "boxed" in example["answer"]:
            example["answer"] = remove_boxed(last_boxed_only_string(example["answer"]))

        assert "boxed" not in example["answer"], dataset_name

        return example

    ds = ds.cast_column("answer", Value("string"))

    ds = ds.map(checker)

    return ds


def get_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    return tokenizer


class Model(ABC):
    def generate(
        self, prompt_token_ids: List[List[int]], num_generations: int, max_tokens: int
    ):
        raise NotImplementedError


class VllmModel(Model):
    def __init__(
        self,
        model_path: str,
        seed: int,
        max_seq_length: int,
        batch_size: int,
        stop_token_id: int,
        num_generations: int,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        generation_temperature: float,
        adapter_path: Optional[str],
    ):
        self.model = vllm.LLM(
            model=model_path,
            trust_remote_code=True,
            seed=seed,
            enable_prefix_caching=False,
            enforce_eager=False,
            max_model_len=max_seq_length,
            max_seq_len_to_capture=max_seq_length,
            dtype="bfloat16",
            model_impl="vllm",
            enable_lora=adapter_path is not None,
        )

        self.sampling_params = vllm.SamplingParams(
            n=num_generations,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            temperature=generation_temperature,
            stop_token_ids=[stop_token_id],
            seed=seed,
            logprobs=0,
        )
        self.adapter_path = adapter_path

    def generate(
        self, prompt_token_ids: List[List[int]], num_generations: int, max_tokens: int
    ):
        self.sampling_params.n = num_generations
        self.sampling_params.max_tokens = max_tokens

        if self.adapter_path is None:
            return self.model.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        else:
            return self.model.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=self.sampling_params,
                lora_request=vllm.lora.request.LoRARequest(
                    "lora", 1, self.adapter_path
                ),
                use_tqdm=False,
            )


@dataclass
class Generation:
    text: str
    token_ids: List[int]
    logprobs: List[float]  # kept for shape compatibility, left empty


@dataclass
class RequestOutputLite:
    outputs: List[Generation]


@torch.inference_mode()
def to_vllm_like_attr(
    sequences,  # torch.LongTensor or HF Generate*Output
    attention_mask: torch.Tensor,  # (B, T_in)
    tokenizer,
) -> List[RequestOutputLite]:
    # Move inputs we touch to CPU up front
    sequences = sequences.cpu()
    attn = attention_mask.cpu()

    BN, _ = sequences.shape
    B = attn.size(0)
    R = BN // B  # num_return_sequences

    pad_id = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    eos_set = set() if eos is None else ({eos} if isinstance(eos, int) else set(eos))

    results: List[RequestOutputLite] = []
    for i in range(B):
        gens: List[Generation] = []
        for r in range(R):
            k = i * R + r
            # Cut off the prompt using its true length
            toks = sequences[k, len(attn[i]) :].tolist()

            # Trim at first EOS or PAD (exclude EOS)
            trimmed = []
            for t in toks:
                trimmed.append(int(t))
                if (pad_id is not None and t == pad_id) or (t in eos_set):
                    break

            text = tokenizer.decode(trimmed, skip_special_tokens=True)
            gens.append(Generation(text=text, token_ids=trimmed, logprobs=[]))

        results.append(RequestOutputLite(outputs=gens))

    return results


class TransformersModel(Model):
    def __init__(
        self,
        model_path: str,
        generation_temperature: float,
        seed: int,
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        stop_token_id: int,
        num_generations: int,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        gen_batch_size: int,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch.set_float32_matmul_precision("high")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            use_cache=True,
        ).to(device)
        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune")

        self.do_sample = generation_temperature > 0.0

        self.gen_kwargs = dict(
            do_sample=self.do_sample,
            temperature=(max(generation_temperature, 1e-8) if self.do_sample else None),
            top_k=(top_k if self.do_sample and top_k > 0 else None),
            top_p=(top_p if self.do_sample and 0.0 < top_p <= 1.0 else None),
            repetition_penalty=(
                repetition_penalty if repetition_penalty != 1.0 else None
            ),
            num_return_sequences=num_generations,
            eos_token_id=stop_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=False,
            output_scores=False,  # just to be explicit
            output_hidden_states=False,
            output_attentions=False,
            cache_implementation="static",
            use_cache=True,
            max_length=max_seq_length,
        )
        self.gen_kwargs = {k: v for k, v in self.gen_kwargs.items() if v is not None}

        self.gen_batch_size = gen_batch_size
        self.max_seq_length = max_seq_length

    def generate(
        self, prompt_token_ids: List[List[int]], num_generations: int, max_tokens: int
    ):
        if self.gen_batch_size <= num_generations:
            assert num_generations % self.gen_batch_size == 0, (
                self.gen_batch_size,
                num_generations,
            )
        if (not self.do_sample) and num_generations > 1:
            print(
                "[note] temperature <= 0 → greedy decoding; multiple returns will likely be identical "
                "unless you implement beam search (not enabled here)."
            )

        self.gen_kwargs["num_return_sequences"] = min(
            self.gen_batch_size, num_generations
        )
        batch = [
            {"input_ids": x, "attention_mask": [1] * len(x)} for x in prompt_token_ids
        ]
        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        prompt_token_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if len(prompt_token_ids) == 1 and num_generations != 1:
                sequences = []

                max_length = -1
                for i in trange(
                    0, num_generations, self.gen_batch_size, desc="Generations"
                ):
                    single_sequences = self.model.generate(
                        input_ids=prompt_token_ids,
                        attention_mask=attention_mask,
                        **self.gen_kwargs,
                    )
                    sequences.append(single_sequences)

                    max_length = max(max_length, single_sequences.shape[1])

                for i in range(len(sequences)):
                    sequences[i] = torch.hstack(
                        [
                            sequences[i],
                            torch.full(
                                size=(
                                    sequences[i].shape[0],
                                    max_length - sequences[i].shape[1],
                                ),
                                fill_value=self.tokenizer.pad_token_id,
                                device=device,
                            ),
                        ]
                    )

                sequences = torch.vstack(sequences)
            else:
                self.gen_kwargs["num_return_sequences"] = num_generations
                sequences = self.model.generate(
                    input_ids=prompt_token_ids,
                    attention_mask=attention_mask,
                    **self.gen_kwargs,
                )

        assert len(sequences) == len(prompt_token_ids) * num_generations, (
            len(sequences),
            len(prompt_token_ids),
            num_generations,
        )

        vllm_out = to_vllm_like_attr(
            sequences=sequences, tokenizer=self.tokenizer, attention_mask=attention_mask
        )

        return vllm_out


@pyrallis.wrap()
def run(config: Config):
    set_logger(verbose=False)
    set_seed(seed=config.seed, device_specific=False, deterministic=False)

    tokenizer = get_tokenizer(config=config)

    stop_token_id = tokenizer.encode(config.stop_token, add_special_tokens=False)
    assert len(stop_token_id) == 1, (
        f"Stop token must be a single token, instead {stop_token_id}"
    )
    stop_token_id = stop_token_id[0]

    logger.info(f"Using {config.model_backend} as backend")
    if config.model_backend == "vllm":
        model = VllmModel(
            model_path=config.model_path,
            seed=config.seed,
            max_seq_length=config.max_seq_length,
            batch_size=config.batch_size,
            stop_token_id=stop_token_id,
            num_generations=config.num_generations,
            repetition_penalty=config.repetition_penalty,
            top_p=config.top_p,
            top_k=config.top_k,
            generation_temperature=config.generation_temperature,
            adapter_path=config.adapter_path,
        )
    elif config.model_backend == "transformers":
        model = TransformersModel(
            model_path=config.model_path,
            seed=config.seed,
            stop_token_id=stop_token_id,
            max_seq_length=config.max_seq_length,
            tokenizer=tokenizer,
            num_generations=config.num_generations,
            repetition_penalty=config.repetition_penalty,
            top_p=config.top_p,
            top_k=config.top_k,
            generation_temperature=config.generation_temperature,
            gen_batch_size=config.gen_batch_size,
        )
    else:
        raise ValueError(config.model_backend)

    for dataset in tqdm(config.datasets, desc="Dataset"):
        if os.path.exists(
            os.path.join(config.savedir, os.path.basename(dataset) + ".pkl")
        ):
            continue
        ds = get_ds(dataset)
        ds = ds.add_column("idx", list(range(len(ds))))
        ds = ds.select(list(range(len(ds)))[config.split_id :: config.total_splits])
        ds = ds.map(
            tokenize_example(
                tokenizer=tokenizer,
                stop_token=config.stop_token,
                max_seq_length=config.max_seq_length,
                template_type=config.template_type,
                append_to=config.append_to,
            ),
            batched=False,
            num_proc=None,
            remove_columns=["problem", "answer", "solution"],
        )
        logger.info(
            f"Example from {dataset}: {[tokenizer.decode(x, skip_special_tokens=False) for x in ds[0]['prompt_input_ids']]}"
        )
        if dataset in config.small_datasets:
            num_generations = config.small_datasets_num_generations
            batch_size = config.batch_size
        else:
            num_generations = config.num_generations
            if config.batch_size == 1 and config.num_generations == 1:
                batch_size = config.gen_batch_size
            else:
                batch_size = config.batch_size

        is_first_sample = True
        problems = []
        popa = []
        for idx in trange(0, len(ds), batch_size, desc="Sample"):
            ds_idx = ds["idx"][idx : idx + batch_size]
            prompt_token_ids = ds["prompt_input_ids"][idx : idx + batch_size]
            prompt_attention_mask = ds["prompt_attention_mask"][idx : idx + batch_size]
            answer_token_ids = ds["answer_input_ids"][idx : idx + batch_size]

            max_gen_tokens = [
                config.max_seq_length - len(prompt_attention_mask[i])
                for i in range(len(prompt_attention_mask))
            ]
            assert all(x > 0 for x in max_gen_tokens), [x for x in max_gen_tokens]
            max_tokens = max(max_gen_tokens)

            request_outputs: List[RequestOutput] = model.generate(
                prompt_token_ids=prompt_token_ids,
                num_generations=num_generations,
                max_tokens=max_tokens,
            )
            if idx + batch_size <= len(ds):
                assert len(request_outputs) == batch_size, (
                    len(request_outputs),
                    batch_size,
                )
            else:
                assert len(request_outputs) == len(ds) - idx, (
                    len(request_outputs),
                    len(ds),
                    idx,
                )

            for prompt_idx, request_output in enumerate(request_outputs):
                assert len(request_output.outputs) == num_generations, (
                    len(request_output.outputs),
                    num_generations,
                )

                solutions = []
                for generation in request_output.outputs:
                    popa.append(
                        (
                            generation.text,
                            last_boxed_only_string(generation.text),
                            tokenizer.decode(
                                answer_token_ids[prompt_idx], skip_special_tokens=False
                            ),
                        )
                    )
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

                    logprobs = []
                    for gen_tok_idx, logprob in enumerate(generation.logprobs):
                        assert generation_token_ids[gen_tok_idx] in logprob, (
                            generation_token_ids[gen_tok_idx] in logprob
                        )
                        logprob = logprob[generation_token_ids[gen_tok_idx]].logprob
                        logprobs.append(logprob)

                    solution = Solution(
                        gen_length=len(generation_token_ids),
                        reward=int(is_correct),
                        logprob=np.sum(logprobs),
                        finished_with_stop_token=generation_token_ids[-1]
                        == stop_token_id,
                        generation_text=tokenizer.decode(
                            generation_token_ids, skip_special_tokens=False
                        ),
                    )
                    solutions.append(solution)

                problem = Problem(
                    problem=tokenizer.decode(
                        prompt_token_ids[prompt_idx], skip_special_tokens=False
                    ),
                    answer=tokenizer.decode(
                        answer_token_ids[prompt_idx], skip_special_tokens=False
                    ),
                    solutions=solutions,
                    ds_idx=ds_idx[prompt_idx],
                )
                problems.append(problem)

        with open(
            os.path.join(
                config.savedir,
                os.path.basename(dataset)
                + f"_{config.split_id}_{config.total_splits}.pkl",
            ),
            "+wb",
        ) as f:
            pickle.dump(problems, f)

        # with open(
        #     os.path.join(config.savedir, os.path.basename(dataset) + "_logs.pkl"), "+wb"
        # ) as f:
        #     pickle.dump(popa, f)


if __name__ == "__main__":
    run()
