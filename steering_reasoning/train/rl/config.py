import math
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from loguru import logger

from steering_reasoning.train.rl.reward_processor import RewardProcessorType


@dataclass
class Config:
    seed: int
    subsample_ds_seed: int

    num_train_samples: Optional[int]
    num_eval_samples: Optional[int]

    num_evals: int

    # Trainer
    eval_steps: int
    eval_strategy: str
    eval_on_start: bool
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    use_reentrant: bool
    num_train_epochs: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    fp16: bool
    bf16: bool
    optim: str
    save_total_limit: int
    save_only_model: bool
    dataloader_num_workers: int
    max_grad_norm: float
    save_strategy: str
    save_steps: int
    report_to: str
    deepspeed: Optional[str]
    max_steps: int
    max_seq_length: int
    dataloader_drop_last: bool
    ddp_find_unused_parameters: bool

    # Output
    output_dir: str

    # Wandb
    project: str
    entity: str
    group: str
    job_type: str
    name: str

    # other
    num_proc: int
    dataset_path: str

    # policy model
    policy_num_nodes: int
    policy_num_gpus_per_node: int
    model_path: str

    # refernece model
    reference_model_replicas: int

    # reward model
    reward_model_replicas: int
    reward_model_pool_num_procs: int

    kl_coef: float
    kl_type: str
    mean_baseline_coef: float

    num_prompts_per_step_per_device_train: int
    num_prompts_per_step_per_device_eval: int

    gamma: float

    is_online_asft: bool

    neg_reward_is_zero: bool

    train_on_sft: bool

    # vllm server params
    vllm_num_engines: int
    vllm_tensor_parallel_size: int

    reward_processor_type: Union[str, RewardProcessorType]
    kl_is_summand_to: str
    num_forwards_per_generation_train: Optional[int]
    num_forwards_per_generation_eval: Optional[int]
    max_num_requests_for_gen: int

    # SAMPLING PARAMS
    repetition_penalty: float
    num_generations: int
    num_generations_in_eval: int
    top_p: float
    top_k: int
    generation_temperature: float
    stop_token: str

    verbose: bool
    deepspeed_timeout: int

    kl_reduce_strategy: str

    batch_equalizing_max_retries: int

    training_setup: str
    steering_at_layer: Optional[int]

    steering_rank: Optional[int]
    lora_rank: Optional[int]
    lora_init: str

    lora_savepath: Optional[str]

    is_dist: bool

    template_type: str
    append_to: bool

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

        model_name = os.path.basename(self.model_path)

        assert self.training_setup in [
            "full_model",
            "steering",
            "lora",
            "steering_rank",
            "steering_at_place",
        ], self.training_setup

        if self.training_setup == "lora":
            assert self.lora_rank is not None
        else:
            assert self.lora_rank is None

        if self.lora_init.lower() in ["true", "false"]:
            self.lora_init = self.lora_init.lower() == "true"

        if self.dataset_path == "agentica-org/deep_scale_r-preview-dataset":
            dataset = "deepscaler"
        else:
            dataset = self.dataset_path

        assert self.template_type in ["r1", "qwen_math"], self.template_type

        self.group = os.path.join(model_name, dataset)
        self.job_type = self.training_setup
        if self.training_setup == "lora":
            self.job_type += f"-{self.lora_rank}"
        if self.training_setup == "steering_rank":
            self.job_type += f"-{self.steering_rank}"
        if (
            self.training_setup in ["steering", "lora"]
        ) and self.steering_at_layer is not None:
            self.job_type += f"-layer-{self.steering_at_layer}"
        self.name = f"seed-{self.seed}_lr-{self.learning_rate}"

        if self.append_to:
            self.name = f"{self.name}_append_to_{self.append_to}"
        if "ADD_PLACE" in os.environ:
            add_place = "_".join(os.environ["ADD_PLACE"].split(".")[2:])
            logger.info(f"add_place: {add_place}")
            self.job_type = f"add_place_{add_place}"

        self.reward_processor_type = RewardProcessorType(self.reward_processor_type)

        effective_num_previous_samples = 1 / (1 - self.mean_baseline_coef)
        self.mean_baseline_coef = 1 - 1 / (
            effective_num_previous_samples * self.gradient_accumulation_steps
        )

        assert self.kl_type in [
            "vanilla",
            "shulman_sequence_level",
            "shulman_token_level",
        ], self.kl_type

        assert self.kl_is_summand_to in ["reward", "loss"], self.kl_is_summand_to

        self.output_dir = os.path.join(
            self.output_dir, self.group, self.job_type, self.name
        )
        if self.is_dist:
            self.policy_num_gpus_per_node = torch.cuda.device_count()
            self.vllm_num_engines = torch.cuda.device_count()
        else:
            self.policy_num_gpus_per_node = torch.cuda.device_count() // 2
            self.vllm_num_engines = torch.cuda.device_count() // 2

            self.num_prompts_per_step_per_device_train *= 2
            self.num_prompts_per_step_per_device_eval *= 2

        assert self.kl_reduce_strategy in ["mean", "sum"], self.kl_reduce_strategy

        if (
            self.num_prompts_per_step_per_device_train * self.num_generations
        ) % self.batch_size != 0 and (
            self.batch_size
            % self.num_prompts_per_step_per_device_train
            * self.num_generations
            == 0
        ):
            self.batch_size = (
                self.num_prompts_per_step_per_device_train * self.num_generations
            )
        if (
            self.num_prompts_per_step_per_device_eval * self.num_generations_in_eval
        ) % self.batch_size != 0 and (
            self.batch_size
            % self.num_prompts_per_step_per_device_eval
            * self.num_generations_in_eval
            == 0
        ):
            self.batch_size = (
                self.num_prompts_per_step_per_device_eval * self.num_generations_in_eval
            )

        assert (
            self.num_prompts_per_step_per_device_train * self.num_generations
        ) % self.batch_size == 0, (
            self.num_prompts_per_step_per_device_train,
            self.num_generations,
            self.batch_size,
        )
        assert (
            self.num_prompts_per_step_per_device_eval * self.num_generations_in_eval
        ) % self.batch_size == 0, (
            self.num_prompts_per_step_per_device_eval,
            self.num_generations_in_eval,
            self.batch_size,
        )

        self.num_forwards_per_generation_train = math.ceil(
            self.num_prompts_per_step_per_device_train
            * self.num_generations
            / self.batch_size
        )
        self.num_forwards_per_generation_eval = math.ceil(
            self.num_prompts_per_step_per_device_eval
            * self.num_generations_in_eval
            / self.batch_size
        )

        self.gradient_accumulation_steps *= self.num_forwards_per_generation_train

        assert self.dataloader_drop_last, self.dataloader_drop_last
