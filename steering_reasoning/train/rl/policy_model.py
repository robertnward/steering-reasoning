import math
import os
import sys
from typing import Any, Dict, List, Literal

import numpy as np
import ray
import torch
import torch.distributed
import torch.nn as nn
import transformers
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    LlamaForCausalLM,
    OlmoForCausalLM,
    PreTrainedTokenizer,
    Qwen2ForCausalLM,
    TrainingArguments,
)

from steering_reasoning.train.rl.config import Config
from steering_reasoning.train.rl.ray_workers.distributed_torch_ray_actor import (
    DistributedTorchRayActor,
)
from steering_reasoning.train.rl.ray_workers.reward_model import DummyRewardModel
from steering_reasoning.train.rl.trainer import RLTrainer
from steering_reasoning.utils.utils import Timeit, set_logger, wandb_init


class CustomCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_padded = super().__call__(
            features=[
                {
                    "input_ids": f["prompt_input_ids"],
                    "attention_mask": f["prompt_attention_mask"],
                }
                for f in features
            ]
        )
        answer_padded = super().__call__(
            features=[
                {
                    "input_ids": f["answer_input_ids"],
                    "attention_mask": f["answer_attention_mask"],
                }
                for f in features
            ]
        )
        gt_cot_padded = super().__call__(
            features=[
                {
                    "input_ids": f["gt_cot_input_ids"],
                    "attention_mask": f["gt_cot_attention_mask"],
                }
                for f in features
            ]
        )

        index = torch.tensor([f["index"] for f in features])

        return {
            "prompt_input_ids": prompt_padded["input_ids"],
            "prompt_attention_mask": prompt_padded["attention_mask"],
            "answer_input_ids": answer_padded["input_ids"],
            "answer_attention_mask": answer_padded["attention_mask"],
            "gt_cot_input_ids": gt_cot_padded["input_ids"],
            "gt_cot_attention_mask": gt_cot_padded["attention_mask"],
            "index": index,
        }


def define_metrics():
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("train_time/*", step_metric="train/global_step")
    wandb.define_metric("train_metrics/*", step_metric="train/global_step")

    wandb.define_metric("eval/global_step")
    wandb.define_metric("eval/*", step_metric="eval/global_step")
    wandb.define_metric("eval_time/*", step_metric="eval/global_step")
    wandb.define_metric("eval_metrics/*", step_metric="eval/global_step")

    wandb.define_metric("eval_gsm8k/global_step")
    wandb.define_metric("eval_gsm8k/*", step_metric="eval_gsm8k/global_step")
    wandb.define_metric("eval_gsm8k_time/*", step_metric="eval_gsm8k/global_step")
    wandb.define_metric("eval_gsm8k_metrics/*", step_metric="eval_gsm8k/global_step")

    wandb.define_metric("eval_math/global_step")
    wandb.define_metric("eval_math/*", step_metric="eval_math/global_step")
    wandb.define_metric("eval_math_time/*", step_metric="eval_math/global_step")
    wandb.define_metric("eval_math_metrics/*", step_metric="eval_math/global_step")

    wandb.define_metric("dataset/global_step")
    wandb.define_metric("dataset/*", step_metric="dataset/global_step")


# from https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/integrations/integration_utils.py#L616
def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    eval_gsm8k_prefix = "eval_gsm8k_"
    eval_gsm8k_prefix_len = len(eval_gsm8k_prefix)
    eval_math_prefix = "eval_math_"
    eval_math_prefix_len = len(eval_math_prefix)
    eval_aime2024_prefix = "eval_aime2024_"
    eval_aime2024_prefix_len = len(eval_aime2024_prefix)
    eval_eval_prefix = "eval_eval_"
    eval_eval_prefix_len = len(eval_eval_prefix)
    eval_train_prefix = "eval_train_"
    eval_train_prefix_len = len(eval_train_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    train_time_prefix = "time_"
    train_time_prefix_len = len(train_time_prefix)
    eval_time_prefix = "eval_time_"
    eval_time_prefix_len = len(eval_time_prefix)
    eval_gsm8k_time_prefix = "eval_gsm8k_time_"
    eval_gsm8k_time_prefix_len = len(eval_gsm8k_time_prefix)
    eval_math_time_prefix = "eval_math_time_"
    eval_math_time_prefix_len = len(eval_math_time_prefix)
    eval_aime2024_time_prefix = "eval_aime2024_time_"
    eval_aime2024_time_prefix_len = len(eval_aime2024_time_prefix)
    eval_eval_time_prefix = "eval_eval_time_"
    eval_eval_time_prefix_len = len(eval_eval_time_prefix)
    eval_train_time_prefix = "eval_train_time_"
    eval_train_time_prefix_len = len(eval_train_time_prefix)
    train_metrics_prefix = "metrics_"
    train_metrics_prefix_len = len(train_metrics_prefix)
    eval_metrics_prefix = "eval_metrics_"
    eval_metrics_prefix_len = len(eval_metrics_prefix)
    eval_gsm8k_metrics_prefix = "eval_gsm8k_metrics_"
    eval_gsm8k_metrics_prefix_len = len(eval_gsm8k_metrics_prefix)
    eval_math_metrics_prefix = "eval_math_metrics_"
    eval_math_metrics_prefix_len = len(eval_math_metrics_prefix)
    eval_aime2024_metrics_prefix = "eval_aime2024_metrics_"
    eval_aime2024_metrics_prefix_len = len(eval_aime2024_metrics_prefix)
    eval_eval_metrics_prefix = "eval_eval_metrics_"
    eval_eval_metrics_prefix_len = len(eval_eval_metrics_prefix)
    eval_train_metrics_prefix = "eval_train_metrics_"
    eval_train_metrics_prefix_len = len(eval_train_metrics_prefix)
    for k, v in d.items():
        if k.startswith(eval_gsm8k_time_prefix):
            new_d["eval_gsm8k_time/" + k[eval_gsm8k_time_prefix_len:]] = v
        elif k.startswith(eval_math_time_prefix):
            new_d["eval_math_time/" + k[eval_math_time_prefix_len:]] = v
        elif k.startswith(eval_aime2024_time_prefix):
            new_d["eval_aime2024_time/" + k[eval_aime2024_time_prefix_len:]] = v
        elif k.startswith(eval_eval_time_prefix):
            new_d["eval_eval_time/" + k[eval_eval_time_prefix_len:]] = v
        elif k.startswith(eval_train_time_prefix):
            new_d["eval_train_time/" + k[eval_train_time_prefix_len:]] = v
        elif k.startswith(eval_time_prefix):
            new_d["eval_time/" + k[eval_time_prefix_len:]] = v
        elif k.startswith(eval_gsm8k_metrics_prefix):
            new_d["eval_gsm8k_metrics/" + k[eval_gsm8k_metrics_prefix_len:]] = v
        elif k.startswith(eval_math_metrics_prefix):
            new_d["eval_math_metrics/" + k[eval_math_metrics_prefix_len:]] = v
        elif k.startswith(eval_aime2024_metrics_prefix):
            new_d["eval_aime2024_metrics/" + k[eval_aime2024_metrics_prefix_len:]] = v
        elif k.startswith(eval_eval_metrics_prefix):
            new_d["eval_eval_metrics/" + k[eval_eval_metrics_prefix_len:]] = v
        elif k.startswith(eval_train_metrics_prefix):
            new_d["eval_train_metrics/" + k[eval_train_metrics_prefix_len:]] = v
        elif k.startswith(eval_metrics_prefix):
            new_d["eval_metrics/" + k[eval_metrics_prefix_len:]] = v
        elif k.startswith(eval_gsm8k_prefix):
            new_d["eval_gsm8k/" + k[eval_gsm8k_prefix_len:]] = v
        elif k.startswith(eval_math_prefix):
            new_d["eval_math/" + k[eval_math_prefix_len:]] = v
        elif k.startswith(eval_aime2024_prefix):
            new_d["eval_aime2024/" + k[eval_aime2024_prefix_len:]] = v
        elif k.startswith(eval_eval_prefix):
            new_d["eval_eval/" + k[eval_eval_prefix_len:]] = v
        elif k.startswith(eval_train_prefix):
            new_d["eval_train/" + k[eval_train_prefix_len:]] = v
        elif k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith(train_time_prefix):
            new_d["train_time/" + k[train_time_prefix_len:]] = v
        elif k.startswith(train_metrics_prefix):
            new_d["train_metrics/" + k[train_metrics_prefix_len:]] = v
        else:
            new_d["train/" + k] = v

    return new_d


guidance_prompt = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def tokenize_example(
    tokenizer,
    stop_token: str,
    max_seq_length: int,
    template_type: Literal["r1", "qwen_math"],
    append_to: bool,
):
    if append_to:
        assert template_type == "qwen_math", (
            f"append_to is only supported for qwen_math, got {template_type}"
        )

    def tokenize_example_inner(example):
        if tokenizer.chat_template is not None:
            assert template_type == "qwen_math", template_type

            conversation = [
                {"role": "system", "content": guidance_prompt},
                {"role": "user", "content": example["problem"]},
            ]
            tokenized_prompt = tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=True,
                tokenize=True,
                padding=False,
                truncation=False,
                return_dict=True,
            )
            if append_to:
                to_token = tokenizer.encode("To", add_special_tokens=False)
                assert len(to_token) == 1, to_token
                to_token = to_token[0]
                tokenized_prompt["input_ids"] += [to_token]
                tokenized_prompt["attention_mask"] += [1]

            tokenized_answer = tokenizer(
                text="\\boxed{" + example["answer"] + "}",
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
                truncation=False,
            )
            tokenized_gt_cot = tokenizer(
                text=example["solution"] + stop_token,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
                truncation=False,
            )

            logger.debug(
                f"Tokenized prompt:\n{tokenized_prompt}\nTokenized answer:\n{tokenized_answer}"
            )
            return {
                "prompt_input_ids": tokenized_prompt["input_ids"],
                "prompt_attention_mask": tokenized_prompt["attention_mask"],
                "answer_input_ids": tokenized_answer["input_ids"][0],
                "answer_attention_mask": tokenized_answer["attention_mask"][0],
                "gt_cot_input_ids": tokenized_gt_cot["input_ids"][0],
                "gt_cot_attention_mask": tokenized_gt_cot["attention_mask"][0],
            }
        else:

            def get_templated(question: str):
                if template_type == "qwen_math":
                    # return (
                    #     f"User: {question}"
                    #     + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nAssistant:"
                    # )
                    return (
                        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please reason step by step, and put your final answer within \\boxed{}."
                        + f"\n\nUser: {question} Assistant:"
                    )
                elif template_type == "r1":
                    # return (
                    #     "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here -- a final expression without any other words </answer>."
                    #     + f"\nUser: {question}\nAssistant: <think>"
                    # )
                    # return (
                    #     "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The answer should contain only the final expression without any additional text."
                    #     + f"\nUser: {question}\nAssistant: <think>"
                    # )
                    return (
                        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
                        + f"\nUser: {question}\nAssistant: <think>"
                    )
                else:
                    raise NotImplementedError(f"Template type is: {template_type}")

            tokenized_prompt = tokenizer(
                tokenizer.bos_token + get_templated(example["problem"]),
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            tokenized_answer = tokenizer(
                "\\boxed{" + example["answer"] + "}",
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            tokenized_gt_cot = tokenizer(
                example["solution"] + tokenizer.eos_token,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )

            return {
                "prompt_input_ids": tokenized_prompt["input_ids"][0],
                "prompt_attention_mask": tokenized_prompt["attention_mask"][0],
                "answer_input_ids": tokenized_answer["input_ids"][0],
                "answer_attention_mask": tokenized_answer["attention_mask"][0],
                "gt_cot_input_ids": tokenized_gt_cot["input_ids"][0],
                "gt_cot_attention_mask": tokenized_gt_cot["attention_mask"][0],
            }

    return tokenize_example_inner


def subsample_ds(ds: Dataset, num_samples: int, np_rng):
    indices = np_rng.choice(
        range(len(ds)), size=min(num_samples, len(ds)), replace=False
    )
    ds = ds.select(indices.tolist())

    return ds


@ray.remote(num_gpus=1)
class PolicyModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()

    def init_model_from_pretrained(self, seed: int, timeout: int):
        return self._setup_distributed(seed=seed, timeout=timeout)

    def _dataset_and_collator_sanity_check(self, dataset, tokenizer, collator) -> None:
        logger.info(f"Train sample input_ids:\n{dataset[0]}")
        logger.info(
            f"Train sample prompt example:\n{tokenizer.decode(dataset[0]['prompt_input_ids'])}"
        )
        logger.info(
            f"Train sample answer example:\n{tokenizer.decode(dataset[0]['answer_input_ids'])}"
        )

    def get_model(self, config: Config, is_reference: bool) -> AutoModelForCausalLM:
        if "Qwen2" in config.model_path:
            model_cls = Qwen2ForCausalLM
        elif "llama3" in config.model_path:
            model_cls = LlamaForCausalLM
        elif "OLMo" in config.model_path:
            model_cls = OlmoForCausalLM
        else:
            raise NotImplementedError(config.model_path)

        if config.training_setup == "steering_rank":
            from steering_reasoning.models.model_with_steering_rank import get_wrapper

            model_cls = get_wrapper(model_cls, steering_rank=config.steering_rank)
        elif "ADD_PLACE" in os.environ:
            from steering_reasoning.models.steering_place.transformers import (
                get_hf_wrapper,
            )

            model_cls = get_hf_wrapper(cls=model_cls)

        model = model_cls.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )

        if not is_reference:
            # Train only steering vectors
            if config.training_setup == "steering":
                for name, param in model.named_parameters():
                    condition = "down_proj.bias" in name
                    if condition:
                        nn.init.zeros_(param)

                    if config.steering_at_layer is not None:
                        condition = (
                            condition and f"layers.{config.steering_at_layer}." in name
                        )

                    if condition:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            # Make lora
            elif config.training_setup == "lora":
                if (
                    "Qwen2" in config.model_path
                    or "llama3" in config.model_path
                    or "OLMo" in config.model_path
                ):
                    if config.steering_at_layer is not None:
                        target_modules = [
                            f"layers.{config.steering_at_layer}.mlp.down_proj"
                        ]
                    else:
                        target_modules = ["down_proj"]
                else:
                    raise NotImplementedError

                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=config.lora_rank,
                    lora_alpha=config.lora_rank,
                    lora_dropout=0.0,
                    target_modules=target_modules,
                    init_lora_weights=config.lora_init,
                )
                model = get_peft_model(model=model, peft_config=peft_config).to(
                    model.dtype
                )
                if torch.distributed.get_rank() == 0:
                    model.print_trainable_parameters()
            elif config.training_setup == "steering_rank":
                for name, param in model.named_parameters():
                    if "steering_matrix_A" in name:
                        # https://github.com/huggingface/peft/blob/b3130c9edb6071c6f90b42bede12c56b4ad59287/src/peft/tuners/lora/layer.py#L254
                        nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                        param.requires_grad = True
                    elif "steering_matrix_B" in name:
                        nn.init.zeros_(param)
                        param.requires_grad = True
                    elif "down_proj.bias" in name:
                        nn.init.zeros_(param)
                        param.requires_grad = False
                    else:
                        param.requires_grad = False

            elif config.training_setup in ["full_model", "steering"]:
                pass
            elif config.training_setup == "steering_at_place":
                for name, param in model.named_parameters():
                    if "steering_vector" == name:
                        nn.init.zeros_(param)
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                raise NotImplementedError(config.training_setup)

        # zero-initialize all extra weights
        if is_reference:
            if config.training_setup == "steering":
                for name, param in model.named_parameters():
                    if "down_proj.bias" in name:
                        nn.init.zeros_(param)
            elif config.training_setup == "steering_rank":
                for name, param in model.named_parameters():
                    if "steering_matrix_A" in name or "steering_matrix_B" in name:
                        nn.init.zeros_(param)
            elif config.training_setup in ["full_model", "lora"]:
                pass
            elif config.training_setup == "steering_at_place":
                for name, param in model.named_parameters():
                    if "steering_vector" == name:
                        nn.init.zeros_(param)
            else:
                raise NotImplementedError(config.training_setup)

            for name, param in model.named_parameters():
                param.requires_grad = False

        if model.config.max_position_embeddings < config.max_seq_length:
            logger.error(
                f"Model max_position_embeddings {model.config.max_position_embeddings} < config.max_seq_length {config.max_seq_length}"
            )
            sys.exit(1)

        return model

    def get_tokenizer(self, config: Config) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="right",
            truncation_side="right",
            model_max_length=config.max_seq_length,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # so there is no warning that lora tokenizer is not found
        if config.training_setup == "lora":
            tokenizer.save_pretrained(config.lora_savepath)

        return tokenizer

    def get_ds(self, config: Config, tokenizer: PreTrainedTokenizer):
        ds = load_dataset(config.dataset_path)
        ds = ds.filter(lambda x: len(x["answer"]) > 0)

        # ds["train"] = ds["train"].filter(lambda x: x["source"] in config.train_datasets)
        ds["train"] = ds["train"].add_column("index", list(range(len(ds["train"]))))
        if "test" in ds:
            # ds["test"] = ds["test"].filter(
            #     lambda x: x["source"] in config.test_datasets
            # )
            ds["test"] = ds["test"].add_column("index", list(range(len(ds["test"]))))

        np_rng = np.random.default_rng(config.subsample_ds_seed)
        if config.num_train_samples is not None:
            ds["train"] = subsample_ds(
                ds["train"], num_samples=config.num_train_samples, np_rng=np_rng
            )

        if "test" in ds and config.num_eval_samples is not None:
            ds["test"] = subsample_ds(
                ds["test"], num_samples=config.num_eval_samples, np_rng=np_rng
            )

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
        len_before_train, len_before_val = (
            len(ds["train"]),
            len(ds["test"]) if "test" in ds else None,
        )
        ds = ds.filter(
            lambda x: len(x["prompt_attention_mask"]) <= config.max_seq_length
        )
        ds = ds.shuffle(seed=config.seed)
        len_after_train, len_after_val = (
            len(ds["train"]),
            len(ds["test"]) if "test" in ds else None,
        )

        logger.info(
            f"After filtering.\nTrain: {len_after_train}/{len_before_train}.\nTest: {len_after_val}/{len_before_val}"
        )

        return ds

    def calc_eval_steps(self, config: Config, ds: DatasetDict):
        num_prompts_in_single_step = (
            config.num_prompts_per_step_per_device_train
            * config.policy_num_gpus_per_node
        )
        if config.max_steps != -1:
            num_steps = config.max_steps
        else:
            num_steps = config.num_train_epochs * (
                len(ds["train"]) // num_prompts_in_single_step
            )
        config.eval_steps = num_steps // config.num_evals
        config.save_steps = math.ceil(num_steps / config.num_evals)

        logger.info(
            f"Set eval steps to {config.eval_steps} because "
            + f"num_prompts_per_step_per_device_train: {config.num_prompts_per_step_per_device_train}, "
            + f"policy_num_gpus_per_node: {config.policy_num_gpus_per_node}, "
            + f"num_prompts_in_single_step: {num_prompts_in_single_step}, "
            + f"ds len: {len(ds['train'])} "
            + f"num_steps: {num_steps} "
            + f"and num_evals: {config.num_evals}"
        )

    def dataset_metrics(self, ds: DatasetDict):
        train_lens = ds["train"].map(
            lambda x: {"len": len(x["prompt_input_ids"])},
            remove_columns=ds["train"].column_names,
        )["len"]
        if "test" in ds:
            test_lens = ds["test"].map(
                lambda x: {"len": len(x["problem_input_ids"])},
                remove_columns=ds["test"].column_names,
            )["len"]
        else:
            test_lens = []

        if torch.distributed.get_rank() == 0:
            wandb.log(
                {
                    "dataset/train_lens": wandb.Histogram(train_lens),
                    "dataset/test_lens": wandb.Histogram(test_lens),
                    "dataset/train_size": len(ds["train"]),
                    "dataset/test_size": len(ds["test"]) if "test" in ds else 0,
                }
            )

    def run(
        self,
        config: Config,
        vllm_engines,
        wandb_run_id: str | None = None,
    ) -> None:
        set_logger(verbose=config.verbose)
        transformers.integrations.integration_utils.rewrite_logs = rewrite_logs
        if torch.distributed.get_rank() == 0:
            # don't create a new run, but log to the provided one
            # this is required for correct work of wandb sweeps
            wandb_init(config, run_id=wandb_run_id)
            define_metrics()

        with Timeit() as model_tok_time:
            model = self.get_model(config=config, is_reference=False)
            tokenizer = self.get_tokenizer(config=config)

        logger.info(f"Model setup time: {model_tok_time.elapsed_time_cpu}")

        ds = self.get_ds(config=config, tokenizer=tokenizer)
        self.dataset_metrics(ds=ds)
        self.calc_eval_steps(config=config, ds=ds)

        data_collator = CustomCollator(tokenizer=tokenizer)

        with Timeit() as ref_model_time:
            if config.kl_coef != 0:
                reference_model = self.get_model(config=config, is_reference=True)

                for _, param in reference_model.named_parameters():
                    param.requires_grad = False

                reference_model.eval()
            else:
                reference_model = None

        logger.info(f"Reference model time: {ref_model_time.elapsed_time_cpu}")

        training_args = TrainingArguments(
            seed=config.seed,
            output_dir=config.output_dir,
            run_name=config.name,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            eval_on_start=config.eval_on_start,
            per_device_train_batch_size=config.num_prompts_per_step_per_device_train,
            per_device_eval_batch_size=config.num_prompts_per_step_per_device_eval,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": config.use_reentrant}
            if config.gradient_checkpointing
            else None,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            logging_steps=config.logging_steps,
            fp16=config.fp16,
            bf16=config.bf16,
            optim=config.optim,
            save_total_limit=config.save_total_limit,
            dataloader_num_workers=config.dataloader_num_workers,
            max_grad_norm=config.max_grad_norm,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            save_only_model=config.save_only_model,
            report_to="wandb",
            deepspeed=config.deepspeed,
            remove_unused_columns=False,
            dataloader_drop_last=config.dataloader_drop_last,
            ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        )
        with Timeit() as get_trainer_time:
            trainer = RLTrainer(
                config=config,
                vllm_engines=vllm_engines,
                args=training_args,
                processing_class=tokenizer,
                policy=model,
                ref_model=reference_model,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds else None,
                data_collator=data_collator,
                reward_model=DummyRewardModel(vocab_size=tokenizer.vocab_size),
                callbacks=[],
                wandb_run_id=wandb_run_id,
            )

        logger.info(
            f"Elapsed get_trainer time: {get_trainer_time.elapsed_time_cpu} seconds"
        )

        if trainer.accelerator.is_main_process:
            self._dataset_and_collator_sanity_check(
                ds["train"], tokenizer, data_collator
            )

        os.makedirs(trainer.args.output_dir, exist_ok=True)

        trainer.train()

        output_dir = os.path.join(trainer.args.output_dir, "final")
        os.makedirs(output_dir, exist_ok=True)

        trainer.save_model(output_dir=output_dir)

        logger.info(f"Saved the model to: {output_dir}")

        wandb.finish(quiet=True)
