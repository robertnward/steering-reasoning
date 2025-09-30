import math
import multiprocessing as mp
import random
import re
import socket
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import deepspeed
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import wandb
from accelerate import Accelerator
from accelerate.utils import is_deepspeed_available, set_seed
from datasets import Dataset
from loguru import logger
from math_verify import parse, verify
from peft import PeftModelForCausalLM
from torch.utils.data.sampler import Sampler, SequentialSampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from vllm import RequestOutput, SamplingParams

from steering_reasoning.train.rl.config import Config
from steering_reasoning.train.rl.ray_workers.distributed import get_mean_std, get_sum
from steering_reasoning.train.rl.ray_workers.raygroup import RayGroup
from steering_reasoning.train.rl.ray_workers.reward_model import RewardModel
from steering_reasoning.train.rl.ray_workers.vllm_engine import LLMRayActor
from steering_reasoning.train.rl.ray_workers.vllm_worker_wrap import (
    stateless_init_process_group,
)
from steering_reasoning.train.rl.reward_processor import (
    GRPONoStdRewardProcessor,
    GRPORewardProcessor,
    IDRewardProcessor,
    MeanBaselineRewardProcessor,
    RewardProcessorType,
    RLOORewardProcessor,
)
from steering_reasoning.utils.math import last_boxed_only_string
from steering_reasoning.utils.utils import (
    Timeit,
    dir_to_zip_bytes,
    disable_dropout_in_model,
    pad_sequences,
    set_logger,
)


def input_ids_to_list(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> List[int]:
    list_input_ids = input_ids[attention_mask.to(dtype=torch.bool)].tolist()

    return list_input_ids


def get_last_one_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    last_one_indices = (
        attention_mask.size(1) - 1 - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
    )

    return last_one_indices


def get_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, generation_temperature: float
):
    logits = logits / generation_temperature

    all_logprob = F.log_softmax(logits, dim=-1)

    logprob = torch.gather(
        all_logprob[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    return logprob


def nanstd(values: torch.Tensor, mean: torch.Tensor, divisor: torch.Tensor):
    var = torch.nansum((values - mean) ** 2, dim=-1, keepdim=True) / (divisor - 1)
    std = var.sqrt()

    return std


def prepare_model_for_deepspeed(
    model: PreTrainedModel | torch.nn.Module, accelerator: Accelerator
) -> PreTrainedModel | torch.nn.Module:
    if not is_deepspeed_available():
        raise ValueError("Deepspeed is not installed")

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    if model is not None:
        if hasattr(model, "config"):
            hidden_size: int | None = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )

            if (
                hidden_size is not None
                and config_kwargs["zero_optimization"]["stage"] == 3
            ):
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size
                        * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10
                        * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9
                        * hidden_size
                        * hidden_size,
                    }
                )

    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    config_kwargs["optimizer"] = {"type": None}

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def prepare_model(
    model: PreTrainedModel | torch.nn.Module,
    accelerator: Accelerator,
    is_deepspeed_enabled: bool = False,
) -> PreTrainedModel | torch.nn.Module:
    if is_deepspeed_enabled:
        model = prepare_model_for_deepspeed(model, accelerator)
    else:
        model = accelerator.prepare_model(model, evaluation_mode=True)

    return model


def sum_all_parameter_values(model):
    if isinstance(model, deepspeed.DeepSpeedEngine):
        model = model.module
    total_sum = sum(p.sum().item() for p in model.parameters())
    return total_sum


def get_reward_processor(config: Config):
    if config.reward_processor_type == RewardProcessorType.REINFORCE:
        return IDRewardProcessor()
    elif config.reward_processor_type == RewardProcessorType.REINFORCE_WITH_BASELINE:
        return MeanBaselineRewardProcessor(mean_baseline_coef=config.mean_baseline_coef)
    elif config.reward_processor_type == RewardProcessorType.RLOO:
        return RLOORewardProcessor(num_generations=config.num_generations)
    elif config.reward_processor_type == RewardProcessorType.GRPO:
        return GRPORewardProcessor(num_generations=config.num_generations)
    elif config.reward_processor_type == RewardProcessorType.GRPO_NO_STD:
        return GRPONoStdRewardProcessor(num_generations=config.num_generations)
    else:
        raise NotImplementedError


def reduce_logprobs(
    logprobs: torch.Tensor,
    tokens_mask: torch.Tensor,
    reduce_strategy: Literal["mean", "sum"],
) -> torch.Tensor:
    assert logprobs.dim() == 2 and tokens_mask.dim() == 2, (
        logprobs.dim(),
        tokens_mask.dim(),
    )

    if reduce_strategy == "mean":
        reduced_logprobs = (logprobs * tokens_mask[:, 1:]).sum(-1) / tokens_mask[
            :, 1:
        ].sum(-1)
        reduced_logprobs[tokens_mask.sum(-1) == 0] = torch.nan

        return reduced_logprobs
    elif reduce_strategy == "sum":
        reduced_logprobs = (logprobs * tokens_mask[:, 1:]).sum(-1)
        reduced_logprobs[tokens_mask.sum(-1) == 0] = torch.nan

        return reduced_logprobs
    else:
        raise NotImplementedError


class RepeatedSampler(Sampler):
    """
    Gradient Accumulation Steps in my implementations are to make a single
    gradient update per generated sequences and all prompts in current batch.
    Therefore, I want to sample the same prompt at each step while gradient update wasn't performed.

    """

    def __init__(self, base_sampler, batch_size, num_repeats):
        """
        Args:
            base_sampler (Sampler): The base sampler (e.g. RandomSampler) yielding indices.
            batch_size (int): The size of each batch.
            num_repeats (int): How many times to repeat each batch.
        """
        self.base_sampler = base_sampler
        self.batch_size = batch_size
        self.num_repeats = num_repeats

    def __iter__(self):
        batch = []
        # Iterate over the base sampler indices
        for idx in self.base_sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # For each full batch, yield it num_repeats times
                for _ in range(self.num_repeats):
                    for i in batch:
                        yield i
                batch = []
        # If drop_last is used in DataLoader, the last incomplete batch will be dropped,
        # so we don't yield it here.

    def __len__(self):
        # Only count full batches multiplied by the repeats.
        # Note: len(self.base_sampler) might not be divisible by batch_size.
        full_batches = len(self.base_sampler) // self.batch_size
        return full_batches * self.batch_size * self.num_repeats


def get_index(param_name: str):
    match = re.search(r"layers\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))
    else:
        return


def broadcast_core(name: str, param: torch.Tensor, vllm_engines, model_update_group):
    # FIXME i dont know will it work with zero-3
    if torch.distributed.get_rank() == 0:
        refs = [
            engine.collective_rpc.remote(
                "update_weight", args=(name, param.dtype, param.shape)
            )
            for engine in vllm_engines
        ]
        model_update_group.broadcast(param, src=0, stream=torch.cuda.current_stream())
        ray.get(refs)


@torch.no_grad()
def broadcast_weights(
    model,
    model_update_group,
    vllm_engines,
    lora_savepath: Optional[str],
    is_pissa: bool,
    is_first: bool,
):
    if is_first and not isinstance(model, PeftModelForCausalLM):
        for name, param in model.named_parameters():
            broadcast_core(
                name=name,
                param=param,
                vllm_engines=vllm_engines,
                model_update_group=model_update_group,
            )

    # Broadcast lora-weights
    if isinstance(model, PeftModelForCausalLM):
        if torch.distributed.get_rank() == 0:
            if is_pissa:
                model.save_pretrained(
                    lora_savepath, path_initial_model_for_weight_conversion="pissa_init"
                )
            else:
                model.save_pretrained(lora_savepath)
            # Send lora to other machine
            # Is relevant for distributed training
            lora_bytes = dir_to_zip_bytes(lora_savepath)

            refs = [engine.accept_lora.remote(lora_bytes) for engine in vllm_engines]
            ray.get(refs)

    # Broadcast a steering-rank model
    elif hasattr(model, "steering_matrix_A") or hasattr(model, "cores"):
        steering_vectors = model.get_steering_vectors()
        for name, param in model.named_parameters():
            if (
                "steering_matrix_A" in name
                or "steering_matrix_B" in name
                or "cores" in name
            ):
                continue
            elif "down_proj.bias" in name:
                assert torch.all(param == 0.0), param
                layer_idx = get_index(param_name=name)
                if layer_idx is None:
                    raise Exception(f"layer_idx is None in {name}")
                assert param.shape == steering_vectors[layer_idx].shape, (
                    param.shape,
                    steering_vectors.shape,
                )
                assert param.dtype == steering_vectors[layer_idx].dtype, (
                    param.dtype,
                    steering_vectors.dtype,
                )

                broadcast_core(
                    name=name,
                    param=steering_vectors[layer_idx],
                    vllm_engines=vllm_engines,
                    model_update_group=model_update_group,
                )

            else:
                broadcast_core(
                    name=name,
                    param=param,
                    vllm_engines=vllm_engines,
                    model_update_group=model_update_group,
                )
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                broadcast_core(
                    name=name,
                    param=param,
                    vllm_engines=vllm_engines,
                    model_update_group=model_update_group,
                )


class BroadcastWeightsCallback(TrainerCallback):
    def __init__(
        self,
        vllm_engines,
        model_update_group,
        lora_savepath: Optional[str],
        is_pissa: bool,
    ):
        self.vllm_engines = vllm_engines
        self.model_update_group = model_update_group
        self.lora_savepath = lora_savepath
        self.is_pissa = is_pissa
        self.times = []

    def on_optimizer_step(self, args, state, control, optimizer, **kwargs):
        model = kwargs.get("model")
        if model is None:
            raise ValueError("Model not found in kwargs.")

        if torch.distributed.get_rank() == 0:
            log_memory(model=model, optimizer=optimizer)

        with Timeit() as broadcast_time:
            broadcast_weights(
                model=model,
                model_update_group=self.model_update_group,
                vllm_engines=self.vllm_engines,
                lora_savepath=self.lora_savepath,
                is_pissa=self.is_pissa,
                is_first=False,
            )

        if torch.distributed.get_rank() == 0:
            self.times.append(broadcast_time.elapsed_time_cpu)
            wandb.log(
                {
                    "broadcast_time": broadcast_time.elapsed_time_cpu,
                    "mean_broadcast_time": np.mean(self.times),
                }
            )

    def on_train_begin(self, args, state, control, optimizer, **kwargs):
        model = kwargs.get("model")
        if model is None:
            raise ValueError("Model not found in kwargs.")

        with Timeit() as broadcast_time:
            broadcast_weights(
                model=model,
                model_update_group=self.model_update_group,
                vllm_engines=self.vllm_engines,
                lora_savepath=self.lora_savepath,
                is_pissa=self.is_pissa,
                is_first=True,
            )

        if torch.distributed.get_rank() == 0:
            self.times.append(broadcast_time.elapsed_time_cpu)
            wandb.log(
                {
                    "broadcast_time": broadcast_time.elapsed_time_cpu,
                    "mean_broadcast_time": np.mean(self.times),
                }
            )


def is_answer_present(
    processing_class: PreTrainedTokenizerBase,
    answer_token_ids: torch.Tensor,
    generation_token_ids: torch.Tensor,
    template_type: Literal["r1", "qwen_math"],
):
    completion_decoded = processing_class.decode(
        generation_token_ids, skip_special_tokens=False
    )

    answer_decoded = processing_class.decode(answer_token_ids, skip_special_tokens=True)

    if template_type == "qwen_math":
        completion_cut = last_boxed_only_string(completion_decoded)
    elif template_type == "r1":
        pattern = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
        finds = pattern.findall(completion_decoded)
        if len(finds) == 0:
            contents = ""
        else:
            contents = finds[-1]
        completion_cut = "\\boxed{" + contents + "}"
    else:
        raise NotImplementedError(f"Template type is: {template_type}")

    if completion_cut is None:
        answer_is_present = False
        strict_answer_is_present = False
    else:
        try:
            completion_parsed = parse(completion_cut)
            was_error = False
        except:
            was_error = True

        if was_error:
            answer_is_present = False
            strict_answer_is_present = False
        else:
            assert answer_decoded is not None

            answer_parsed = parse(answer_decoded)
            i = 0
            while len(answer_parsed) == 0 and i < 5:
                i += 1
                answer_parsed = parse(answer_decoded)

            if len(answer_parsed) == 0:
                answer_is_present = False
            else:
                answer_is_present = verify(answer_parsed, completion_parsed)

            strict_answer_is_present = answer_decoded == completion_cut

    return answer_is_present, strict_answer_is_present


def log_num_trainable_parameters(model: PreTrainedModel):
    all_parameters = 0
    trainable_parameters = 0
    for _, param in model.named_parameters():
        all_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    wandb.log(
        {"all_parameters": all_parameters, "trainable_parameters": trainable_parameters}
    )


def bytes_used_by_named_tensors(tensors):
    """tensors: iterable of torch.Tensor or nn.Parameter"""
    return sum(t.numel() * t.element_size() for t in tensors)


def log_memory(model, optimizer):
    model_bytes = bytes_used_by_named_tensors(model.parameters())

    opt_state_tensors = []
    for s in optimizer.state.values():  # e.g. exp_avg, exp_avg_sq …
        opt_state_tensors += [v for v in s.values() if torch.is_tensor(v)]

    opt_bytes = bytes_used_by_named_tensors(opt_state_tensors)

    wandb.log(
        {
            "memory/model_bs": model_bytes,
            "memory/model_kbs": round(model_bytes / 2**10, 1),
            "memory/model_mbs": round(model_bytes / 2**20, 1),
            "memory/model_gbs": round(model_bytes / 2**30, 1),
            "memory/opt_bs": opt_bytes,
            "memory/opt_kbs": round(opt_bytes / 2**10, 1),
            "memory/opt_mbs": round(opt_bytes / 2**20, 1),
            "memory/opt_gbs": round(opt_bytes / 2**30, 1),
        }
    )


class RLTrainer(Trainer):
    def __init__(
        self,
        vllm_engines: list,
        config: Config,
        args: TrainingArguments,
        processing_class: PreTrainedTokenizerBase,
        policy: PreTrainedModel | torch.nn.Module,
        reward_model: Union[RewardModel],
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | torch.nn.Module | None = None,
        callbacks: list[TrainerCallback] | None = None,
        wandb_run_id: str | None = None,
        **kwargs,
    ) -> None:
        set_logger(config.verbose)
        torch.backends.cudnn.benchmark = False
        set_seed(seed=config.seed, device_specific=False, deterministic=False)

        super().__init__(
            model=policy,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )

        if torch.distributed.get_rank() == 0:
            log_num_trainable_parameters(model=self.model)

        self.config = config
        self.vllm_engines = vllm_engines
        self.ref_model = ref_model

        self.start_vllm_engines()
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            self.add_callback(
                BroadcastWeightsCallback(
                    vllm_engines=self.vllm_engines,
                    model_update_group=self.model_update_group,
                    lora_savepath=self.config.lora_savepath,
                    is_pissa=self.config.lora_init is not None
                    and isinstance(self.config.lora_init, str)
                    and self.config.lora_init == "pissa",
                )
            )

        self.start_ref_model()

        disable_dropout_in_model(self.model)

        self.reward_model = reward_model

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.kl_coef = config.kl_coef

        self.stop_token_id = self.processing_class.encode(
            self.config.stop_token, add_special_tokens=False
        )
        assert len(self.stop_token_id) == 1, (
            f"Stop token must be a single token, instead {self.stop_token_id}"
        )
        self.stop_token_id = self.stop_token_id[0]

        assert (
            self.processing_class.eos_token is not None
            and self.processing_class.eos_token_id is not None
        ), (self.processing_class.eos_token, self.processing_class.eos_token_id)

        self.sampling_params = SamplingParams(
            repetition_penalty=config.repetition_penalty,
            top_p=config.top_p,
            top_k=config.top_k,
            temperature=config.generation_temperature,
            stop_token_ids=[self.stop_token_id],
            seed=config.seed,
        )

        with Timeit() as reward_processor_time:
            self.reward_processor = get_reward_processor(config=config)
            logger.info(f"Reward Processor: {type(self.reward_processor)}")

        logger.info(f"Reward Processor Time: {reward_processor_time.elapsed_time_cpu}")

        self.rng = random.Random(self.config.seed)

        self.best_num_solved_strict = defaultdict(lambda: -1)

        self.generation_buffer = []

        self.num_gen_tokens_in_batch = None

        logger.info(f"IS DEEPSPEED ENABLED: {self.is_deepspeed_enabled}")

        self.prev_should_evaluate = None
        self.prev_global_step = None

        self.is_evaluating = None
        self.curent_metrics_prefix = ""

        # Record hiddens states at residual (after each layer)
        # self.hiddens_values: Dict[str, torch.Tensor] = {}
        # self.hiddens_values_captures_handles = []

        # def hiddens_values_capture(name):
        #     def hook(module, inputs, output):
        #         assert isinstance(output, tuple), type(output)
        #         self.hiddens_values[name] = output[0].detach()

        #     return hook

        # logger.info(f"Model: {self.get_inner_model().model}")

        # for layer_idx, layer in enumerate(self.get_inner_model().model.layers):
        #     handle = layer.register_forward_hook(
        #         hiddens_values_capture(f"layer_{layer_idx}_hiddens")
        #     )
        #     self.hiddens_values_captures_handles.append(handle)

        self.reward_model_pool = mp.Pool(
            processes=self.config.reward_model_pool_num_procs
        )

        if torch.distributed.get_rank() == 0:
            wandb.log({"memory": torch.cuda.max_memory_allocated("cuda:0")})

    def get_inner_model(self):
        if isinstance(self.model, PeftModelForCausalLM):
            return self.model.model
        else:
            return self.model

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None):
        """
        See `RepeatedRandomSampler` for details.
        """
        base_sampler = super()._get_train_sampler(train_dataset=train_dataset)

        return RepeatedSampler(
            base_sampler=base_sampler,
            batch_size=self.args.per_device_train_batch_size
            * torch.distributed.get_world_size(),
            num_repeats=self.args.gradient_accumulation_steps,
        )

    def _get_eval_sampler(self, eval_dataset):
        """
        See `RepeatedRandomSampler` for details.
        """
        base_sampler = super()._get_eval_sampler(eval_dataset=eval_dataset)
        if base_sampler is None:
            base_sampler = SequentialSampler(data_source=eval_dataset)

        return RepeatedSampler(
            base_sampler=base_sampler,
            batch_size=self.args.per_device_eval_batch_size
            * torch.distributed.get_world_size(),
            num_repeats=self.config.num_forwards_per_generation_eval,
        )

    def start_vllm_engines(self):
        with Timeit() as vllm_engines_time:
            if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
                with Timeit() as time1:
                    master_address = ray._private.services.get_node_ip_address()
                with Timeit() as time2:
                    with socket.socket() as sock:
                        sock.bind(("", 0))
                        master_port = sock.getsockname()[1]

                # TODO_RLOO assert tp_size same for all engines
                with Timeit() as time3:
                    world_size = (
                        self.config.vllm_num_engines
                        * self.config.vllm_tensor_parallel_size
                        + 1
                    )

                with Timeit() as time4:
                    refs = [
                        engine.init_weight_update_group.remote(
                            master_address,
                            master_port,
                            i * self.config.vllm_tensor_parallel_size + 1,
                            world_size,
                        )
                        for i, engine in enumerate(self.vllm_engines)
                    ]

                with Timeit() as time5:
                    # https://github.com/vllm-project/vllm/issues/11399
                    # https://github.com/vllm-project/vllm/pull/12084
                    # https://github.com/vllm-project/vllm/issues/5723
                    self.model_update_group = stateless_init_process_group(
                        master_address=master_address,
                        master_port=master_port,
                        world_size=world_size,
                        rank=0,
                        device=torch.device("cuda:0"),
                    )

                with Timeit() as time6:
                    ray.get(refs)

            torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            logger.info(
                f"VLLM Engines init time: {vllm_engines_time.elapsed_time_cpu}s"
            )
            logger.info(f"VLLM Engines Time 1: {time1.elapsed_time_cpu}s")
            logger.info(f"VLLM Engines Time 2: {time2.elapsed_time_cpu}s")
            logger.info(f"VLLM Engines Time 3: {time3.elapsed_time_cpu}s")
            logger.info(f"VLLM Engines Time 4: {time4.elapsed_time_cpu}s")
            logger.info(f"VLLM Engines Time 5: {time5.elapsed_time_cpu}s")
            logger.info(f"VLLM Engines Time 6: {time6.elapsed_time_cpu}s")

    def start_ref_model(self):
        with Timeit() as ref_model_time:
            if self.ref_model is not None:
                # TODO: TODO_RLOO watch later
                if self.ref_model is not None and not isinstance(
                    self.ref_model, RayGroup
                ):
                    self.ref_model = prepare_model(
                        self.ref_model, self.accelerator, self.is_deepspeed_enabled
                    )
                    disable_dropout_in_model(self.ref_model)

                elif isinstance(self.ref_model, RayGroup):
                    ray.get(
                        self.ref_model.prepare_reference_model(
                            self.accelerator, self.is_deepspeed_enabled
                        )
                    )

        logger.info(f"Reference Model Time: {ref_model_time.elapsed_time_cpu}")

    def get_valid_mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns a mask indicating whether `stop_token_id` is present
        in the token sequence on the last position.
        """

        last_one_indices = get_last_one_indices(attention_mask=attention_mask)

        assert len(last_one_indices) == attention_mask.size(0), (
            len(last_one_indices),
            attention_mask,
        )

        last_input_ids = input_ids[
            torch.arange(len(last_one_indices)).to(device=input_ids.device),
            last_one_indices,
        ]

        valid_mask = last_input_ids == self.stop_token_id

        return valid_mask

    @torch.no_grad()
    def get_rewards(
        self,
        input_ids: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
        answer_token_ids: torch.Tensor,
        answer_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        rewards = []
        for idx in range(len(input_ids)):
            answer_idx = idx // self.get_num_generations()
            answer_tokens = input_ids_to_list(
                answer_token_ids[answer_idx], answer_token_mask[answer_idx]
            )
            generation_tokens = input_ids_to_list(
                input_ids[idx], generation_tokens_mask[idx]
            )

            reward, strict_reward = self.is_answer_present(
                answer_token_ids=answer_tokens, generation_token_ids=generation_tokens
            )

            reward = 1 if reward else (0 if self.config.neg_reward_is_zero else -1)

            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)

        return rewards

    def get_max_generation_tokens(
        self, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns the maximum number of tokens to generate for each example.
        """
        return self.config.max_seq_length - inputs["prompt_attention_mask"].sum(-1)

    def choose_inference_server(self) -> LLMRayActor:
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        return llm

    def get_max_tokens(self, max_gen_tokens: torch.Tensor) -> int:
        return max_gen_tokens.max().item()

    def get_num_prompts_for_request(self):
        return math.ceil(
            self.config.max_num_requests_for_gen / self.get_num_generations()
        )

    def send_request_for_generations(
        self,
        inputs: Dict[str, torch.Tensor],
        sampling_params: None | SamplingParams = None,
    ) -> List[List[int]]:
        if sampling_params is None:
            sampling_params = self.sampling_params

        if not self.config.train_on_sft or self.is_evaluating:
            list_input_ids = []
            for idx in range(len(inputs["prompt_input_ids"])):
                non_pad_input_ids = input_ids_to_list(
                    input_ids=inputs["prompt_input_ids"][idx],
                    attention_mask=inputs["prompt_attention_mask"][idx],
                )
                list_input_ids.append(non_pad_input_ids)

            llm = self.choose_inference_server()

            max_gen_tokens = self.get_max_generation_tokens(inputs)
            assert torch.all(max_gen_tokens >= 0) and torch.all(
                max_gen_tokens < self.config.max_seq_length
            ), max_gen_tokens

            sampling_params.max_tokens = self.get_max_tokens(max_gen_tokens)

            generation_token_ids = []
            if not sampling_params.max_tokens == 0:
                for batch_idx in range(
                    0, len(list_input_ids), self.get_num_prompts_for_request()
                ):
                    request_outputs: List[RequestOutput] = ray.get(
                        llm.generate.remote(
                            prompt_token_ids=list_input_ids[
                                batch_idx : batch_idx
                                + self.get_num_prompts_for_request()
                            ],
                            sampling_params=sampling_params,
                        )
                    )

                    for generation, max_tokens in zip(
                        request_outputs,
                        max_gen_tokens[
                            batch_idx : batch_idx + self.get_num_prompts_for_request()
                        ],
                    ):
                        max_tokens = max_tokens.item()

                        for gen in generation.outputs:
                            sliced_generation_token_ids = list(
                                gen.token_ids[:max_tokens]
                            )
                            generation_token_ids.append(sliced_generation_token_ids)
            else:
                for _ in range(
                    len(inputs["prompt_input_ids"]) * self.get_num_generations()
                ):
                    generation_token_ids.append([])
        else:
            generation_token_ids = []
            for idx in range(len(inputs["gt_cot_input_ids"])):
                single_generation_token_ids = input_ids_to_list(
                    input_ids=inputs["gt_cot_input_ids"][idx],
                    attention_mask=inputs["gt_cot_attention_mask"][idx],
                )
                generation_token_ids.append(single_generation_token_ids)

        return generation_token_ids

    def get_num_generations(self):
        if self.is_evaluating:
            return self.config.num_generations_in_eval
        else:
            return self.config.num_generations

    def is_answer_present(
        self, answer_token_ids: torch.Tensor, generation_token_ids: torch.Tensor
    ):
        return is_answer_present(
            processing_class=self.processing_class,
            answer_token_ids=answer_token_ids,
            generation_token_ids=generation_token_ids,
            template_type=self.config.template_type,
        )

    def get_completions_metrics(
        self,
        inputs: Dict[str, torch.Tensor],
        generations: List[List[int]],
        device: torch.device,
    ) -> Dict[str, float]:
        no_end_token_is_present = []
        ends_with_im_end = []
        ends_with_eos = []
        answer_is_present = []
        strict_answer_is_present = []
        completion_lens = []
        no_end_completion_lens = []
        end_completion_lens = []

        for gen_idx, generation_token_ids in enumerate(generations):
            prompt_idx = gen_idx // self.get_num_generations()
            single_no_end_token_is_present = (
                self.stop_token_id not in generation_token_ids
                and self.processing_class.eos_token_id not in generation_token_ids
            )
            no_end_token_is_present.append(single_no_end_token_is_present)
            if single_no_end_token_is_present:
                no_end_completion_lens.append(len(generation_token_ids))
                end_completion_lens.append(torch.nan)
            else:
                no_end_completion_lens.append(torch.nan)
                end_completion_lens.append(len(generation_token_ids))

            ends_with_im_end.append(
                len(generation_token_ids) != 0
                and generation_token_ids[-1] == self.stop_token_id
            )
            ends_with_eos.append(
                len(generation_token_ids) != 0
                and generation_token_ids[-1] == self.processing_class.eos_token_id
            )

            # Check whether the answer is among the generated tokens
            answer_token_ids = input_ids_to_list(
                input_ids=inputs["answer_input_ids"][prompt_idx],
                attention_mask=inputs["answer_attention_mask"][prompt_idx],
            )

            single_answer_is_present, single_strict_answer_is_present = (
                is_answer_present(
                    processing_class=self.processing_class,
                    answer_token_ids=answer_token_ids,
                    generation_token_ids=generation_token_ids,
                    template_type=self.config.template_type,
                )
            )
            answer_is_present.append(single_answer_is_present)
            strict_answer_is_present.append(single_strict_answer_is_present)

            completion_lens.append(len(generation_token_ids))

        no_end_token_is_present = torch.tensor(
            no_end_token_is_present, dtype=torch.bool, device=device
        ).reshape(-1, self.get_num_generations())
        ends_with_im_end = torch.tensor(
            ends_with_im_end, dtype=torch.bool, device=device
        ).reshape(-1, self.get_num_generations())
        ends_with_eos = torch.tensor(
            ends_with_eos, dtype=torch.bool, device=device
        ).reshape(-1, self.get_num_generations())
        answer_is_present = torch.tensor(
            answer_is_present, dtype=torch.bool, device=device
        ).reshape(-1, self.get_num_generations())
        strict_answer_is_present = torch.tensor(
            strict_answer_is_present, dtype=torch.bool, device=device
        ).reshape(-1, self.get_num_generations())
        completion_lens = torch.tensor(
            completion_lens, dtype=torch.float, device=device
        ).reshape(-1, self.get_num_generations())
        no_end_completion_lens = torch.tensor(
            no_end_completion_lens, dtype=torch.float, device=device
        ).reshape(-1, self.get_num_generations())
        end_completion_lens = torch.tensor(
            end_completion_lens, dtype=torch.float, device=device
        ).reshape(-1, self.get_num_generations())

        metrics = {}
        at = 1
        while True:
            if at > no_end_token_is_present.shape[1]:
                break

            metrics.update(
                {
                    f"metrics_at_{at}_prompt_no_end_token_is_present": no_end_token_is_present[
                        :, :at
                    ]
                    .any(dim=1)
                    .to(torch.float32),
                    f"metrics_at_{at}_frac_for_prompt_no_end_token_is_present": no_end_token_is_present[
                        :, :at
                    ]
                    .to(torch.float32)
                    .mean(dim=1),
                    f"metrics_at_{at}_prompt_ends_with_im_end": ends_with_im_end[:, :at]
                    .any(dim=1)
                    .to(torch.float32),
                    f"metrics_at_{at}_frac_for_prompt_ends_with_im_end": ends_with_im_end[
                        :, :at
                    ]
                    .to(torch.float32)
                    .mean(dim=1),
                    f"metrics_at_{at}_prompt_ends_with_eos": ends_with_eos[:, :at]
                    .any(dim=1)
                    .to(torch.float32),
                    f"metrics_at_{at}_frac_for_prompt_ends_with_eos": ends_with_eos[
                        :, :at
                    ]
                    .to(torch.float32)
                    .mean(dim=1),
                    f"metrics_at_{at}_prompt_answer_is_present": answer_is_present[
                        :, :at
                    ]
                    .any(dim=1)
                    .to(torch.float32),
                    f"metrics_at_{at}_frac_for_prompt_answer_is_present": answer_is_present[
                        :, :at
                    ]
                    .to(torch.float32)
                    .mean(dim=1),
                    f"metrics_at_{at}_prompt_strict_answer_is_present": strict_answer_is_present[
                        :, :at
                    ]
                    .any(dim=1)
                    .to(torch.float32),
                    f"metrics_at_{at}_frac_for_prompt_strict_answer_is_present": strict_answer_is_present[
                        :, :at
                    ]
                    .to(torch.float32)
                    .mean(dim=1),
                    f"metrics_at_{at}_num_solved": answer_is_present[:, :at]
                    .any(dim=1)
                    .to(torch.float32)
                    .sum(),
                    f"metrics_at_{at}_num_solved_strict": strict_answer_is_present[
                        :, :at
                    ]
                    .any(dim=1)
                    .to(torch.float32)
                    .sum(),
                    f"metrics_at_{at}_frac_solved": answer_is_present[:, :at]
                    .any(dim=1)
                    .to(torch.float32)
                    .sum()
                    / (
                        self.accelerator.num_processes
                        * (
                            self.args.per_device_train_batch_size
                            if not self.is_evaluating
                            else self.args.per_device_eval_batch_size
                        )
                    ),
                    f"metrics_at_{at}_frac_solved_strict": strict_answer_is_present[
                        :, :at
                    ]
                    .any(dim=1)
                    .to(torch.float32)
                    .sum()
                    / (
                        self.accelerator.num_processes
                        * (
                            self.args.per_device_train_batch_size
                            if not self.is_evaluating
                            else self.args.per_device_eval_batch_size
                        )
                    ),
                }
            )

            at *= 2

        metrics.update(
            {
                "metrics_completion_lens": completion_lens.flatten(),
                "metrics_no_end_completion_lens": no_end_completion_lens.flatten(),
                "metrics_end_completion_lens": end_completion_lens.flatten(),
                "metrics_is_solved_strict": strict_answer_is_present.flatten().to(
                    torch.float32
                ),
                "metrics_num_prompts": torch.tensor(
                    [len(inputs["prompt_input_ids"])],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

        return metrics

    def single_merge_inputs_and_generations(
        self, inputs: Dict[str, torch.Tensor], generation_token_ids: List[int]
    ):
        prompt_token_ids = input_ids_to_list(
            input_ids=inputs["prompt_input_ids"],
            attention_mask=inputs["prompt_attention_mask"],
        )

        input_ids = torch.hstack(
            [torch.tensor(prompt_token_ids), torch.tensor(generation_token_ids)]
        )
        attention_mask = torch.ones_like(input_ids)
        prompt_tokens_mask = torch.hstack(
            [torch.ones(len(prompt_token_ids)), torch.zeros(len(generation_token_ids))]
        )
        generation_tokens_mask = torch.hstack(
            [torch.zeros(len(prompt_token_ids)), torch.ones(len(generation_token_ids))]
        )

        return (
            input_ids.to(torch.long),
            attention_mask.to(torch.long),
            prompt_tokens_mask.to(torch.long),
            generation_tokens_mask.to(torch.long),
        )

    def check_single_generations_health(
        self,
        single_input_ids: torch.Tensor,
        single_attention_mask: torch.Tensor,
        single_prompt_tokens_mask: torch.Tensor,
        single_generation_tokens_mask: torch.Tensor,
    ):
        assert (
            len(single_input_ids) <= self.config.max_seq_length
            and len(single_attention_mask) <= self.config.max_seq_length
            and len(single_prompt_tokens_mask) <= self.config.max_seq_length
            and len(single_generation_tokens_mask) <= self.config.max_seq_length
        ), (
            len(single_input_ids),
            len(single_attention_mask),
            len(single_prompt_tokens_mask),
            len(single_generation_tokens_mask),
            self.config.max_seq_length,
        )
        assert (
            len(single_input_ids) == len(single_attention_mask)
            and len(single_attention_mask) == len(single_prompt_tokens_mask)
            and len(single_prompt_tokens_mask) == len(single_generation_tokens_mask)
        ), (
            len(single_input_ids),
            len(single_attention_mask),
            len(single_prompt_tokens_mask),
            len(single_generation_tokens_mask),
        )

    def merge_inputs_and_generations(
        self,
        inputs: Dict[str, torch.Tensor],
        generations: List[RequestOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompt_tokens_mask = []
        generation_tokens_mask = []

        for gen_idx, generation_token_ids in enumerate(generations):
            prompt_idx = gen_idx // self.get_num_generations()
            inputs_at_idx = {k: v[prompt_idx] for k, v in inputs.items()}

            (
                single_input_ids,
                single_attention_mask,
                single_prompt_tokens_mask,
                single_generation_tokens_mask,
            ) = self.single_merge_inputs_and_generations(
                inputs=inputs_at_idx, generation_token_ids=generation_token_ids
            )

            self.check_single_generations_health(
                single_input_ids=single_input_ids,
                single_attention_mask=single_attention_mask,
                single_prompt_tokens_mask=single_prompt_tokens_mask,
                single_generation_tokens_mask=single_generation_tokens_mask,
            )

            input_ids.append(single_input_ids)
            attention_mask.append(single_attention_mask)
            prompt_tokens_mask.append(single_prompt_tokens_mask)
            generation_tokens_mask.append(single_generation_tokens_mask)

        max_length = max(input_id.size(0) for input_id in input_ids)
        assert max_length <= self.config.max_seq_length, (
            max_length,
            self.config.max_seq_length,
        )

        input_ids = pad_sequences(
            sequences=input_ids,
            pad_length=max_length,
            pad_value=self.processing_class.pad_token_id,
        )
        attention_mask = pad_sequences(
            sequences=attention_mask, pad_length=max_length, pad_value=0
        )
        prompt_tokens_mask = pad_sequences(
            sequences=prompt_tokens_mask, pad_length=max_length, pad_value=0
        )
        generation_tokens_mask = pad_sequences(
            sequences=generation_tokens_mask, pad_length=max_length, pad_value=0
        )

        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(attention_mask.to(torch.bool) == 0, 0)

        return (
            input_ids.to(device=device),
            attention_mask.to(device=device),
            prompt_tokens_mask.to(device=device),
            generation_tokens_mask.to(device=device),
            position_ids.to(device=device),
        )

    def check_generations_health(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens_mask: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        assert (
            input_ids.shape[1] <= self.config.max_seq_length
            and attention_mask.shape[1] <= self.config.max_seq_length
            and prompt_tokens_mask.shape[1] <= self.config.max_seq_length
            and generation_tokens_mask.shape[1] <= self.config.max_seq_length
            and position_ids.shape[1] <= self.config.max_seq_length
        ), (
            len(input_ids),
            len(attention_mask),
            len(prompt_tokens_mask),
            len(generation_tokens_mask),
            len(position_ids),
            self.config.max_seq_length,
        )

        assert (
            input_ids.shape == attention_mask.shape
            and attention_mask.shape == prompt_tokens_mask.shape
            and prompt_tokens_mask.shape == generation_tokens_mask.shape
            and generation_tokens_mask.shape == position_ids.shape
        ), (
            input_ids.shape,
            attention_mask.shape,
            prompt_tokens_mask.shape,
            position_ids.shape,
        )

        if self.is_evaluating:
            num_prompts = self.args.per_device_eval_batch_size
        else:
            num_prompts = self.args.per_device_train_batch_size
        assert len(input_ids) == num_prompts * self.get_num_generations(), (
            len(input_ids),
            self.is_evaluating,
            num_prompts,
            self.get_num_generations(),
        )

    def get_completions(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with Timeit() as completions_time:
            generations = self.send_request_for_generations(inputs=inputs)

        with Timeit() as completions_metrics_time:
            completions_metrics = self.get_completions_metrics(
                inputs=inputs,
                generations=generations,
                device=inputs["prompt_input_ids"].device,
            )

        with Timeit() as padding_time:
            (
                input_ids,
                attention_mask,
                prompt_tokens_mask,
                generation_tokens_mask,
                position_ids,
            ) = self.merge_inputs_and_generations(
                inputs=inputs,
                generations=generations,
                device=inputs["prompt_input_ids"].device,
            )

        with Timeit() as check_health_time:
            self.check_generations_health(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_tokens_mask=prompt_tokens_mask,
                generation_tokens_mask=generation_tokens_mask,
                position_ids=position_ids,
            )

        time_metrics = {
            "time_padding": padding_time.elapsed_time_cpu,
            "time_completions": completions_time.elapsed_time_cpu,
            "time_completions_metrics": completions_metrics_time.elapsed_time_cpu,
            "time_check_health": check_health_time.elapsed_time_cpu,
        }

        return (
            input_ids,
            attention_mask,
            prompt_tokens_mask,
            generation_tokens_mask,
            position_ids,
            completions_metrics,
            time_metrics,
        )

    def get_ref_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.ref_model, RayGroup):
            ref_logits = ray.get(
                self.ref_model.reference_forward(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        #'use_cache': False
                    },
                    index=torch.distributed.get_rank()
                    % self.config.reference_model_replicas,
                    # index=0,
                )
            )

        else:
            ref_logits = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits

        return ref_logits

    def get_kl_term(
        self,
        input_ids: torch.Tensor,
        policy_logprobs: torch.Tensor,
        reference_policy_logprobs: torch.Tensor,
        tokens_mask: torch.Tensor,
        reduce_strategy: Literal["sum", "mean"],
    ):
        if self.config.kl_type == "vanilla":
            return reduce_logprobs(
                logprobs=policy_logprobs - reference_policy_logprobs,
                tokens_mask=tokens_mask,
                reduce_strategy=reduce_strategy,
            )
        elif self.config.kl_type == "shulman_sequence_level":
            diff_reduced = reduce_logprobs(
                logprobs=reference_policy_logprobs - policy_logprobs,
                tokens_mask=tokens_mask,
                reduce_strategy=reduce_strategy,
            )
            return torch.exp(diff_reduced) - diff_reduced - 1
        elif self.config.kl_type == "shulman_token_level":
            return reduce_logprobs(
                logprobs=(
                    torch.exp(reference_policy_logprobs - policy_logprobs)
                    - (reference_policy_logprobs - policy_logprobs)
                    - 1
                ),
                tokens_mask=tokens_mask,
                reduce_strategy=reduce_strategy,
            )
        else:
            raise NotImplementedError

    def get_rl_loss(
        self,
        reward: torch.Tensor,
        logprobs: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        reduced_logprobs = reduce_logprobs(
            logprobs=logprobs,
            tokens_mask=generation_tokens_mask,
            reduce_strategy="sum",
        )
        if not self.config.train_on_sft:
            rl_loss = reward * reduced_logprobs
        else:
            rl_loss = reduced_logprobs

        return rl_loss

    def fill_buffer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens_mask: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
        position_ids: torch.Tensor,
        ref_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        baselined_rewards: torch.Tensor,
    ):
        assert (
            len(input_ids) == len(attention_mask)
            and len(input_ids) == len(prompt_tokens_mask)
            and len(input_ids) == len(generation_tokens_mask)
            and len(input_ids) == len(position_ids)
            and (ref_logprobs is None or len(input_ids) == len(ref_logprobs))
            and (old_logprobs is None or len(input_ids) == len(old_logprobs))
            and len(input_ids) == len(baselined_rewards)
        ), (
            len(input_ids),
            len(attention_mask),
            len(prompt_tokens_mask),
            len(generation_tokens_mask),
            len(position_ids),
            len(ref_logprobs),
            len(old_logprobs),
            len(baselined_rewards),
        )

        for start_idx in range(0, len(input_ids), self.config.batch_size):
            # Cut off the part with non-padding tokens
            # It should be a lot of that because padding is performed
            # on all the `num_prompts * num_generations` sequences.
            segment = attention_mask[start_idx : start_idx + self.config.batch_size]
            assert torch.all(segment[:, 0] == 1), segment[:, 0]
            segment_seq_len = segment.sum(-1).max()

            batch_data = {}
            batch_data["input_ids"] = input_ids[
                start_idx : start_idx + self.config.batch_size, :segment_seq_len
            ]
            batch_data["attention_mask"] = attention_mask[
                start_idx : start_idx + self.config.batch_size, :segment_seq_len
            ]
            batch_data["prompt_tokens_mask"] = prompt_tokens_mask[
                start_idx : start_idx + self.config.batch_size, :segment_seq_len
            ]
            batch_data["generation_tokens_mask"] = generation_tokens_mask[
                start_idx : start_idx + self.config.batch_size, :segment_seq_len
            ]
            batch_data["position_ids"] = position_ids[
                start_idx : start_idx + self.config.batch_size, :segment_seq_len
            ]
            if ref_logprobs is not None:
                batch_data["ref_logprobs"] = ref_logprobs[
                    start_idx : start_idx + self.config.batch_size,
                    : segment_seq_len - 1,
                ]
            if old_logprobs is not None:
                batch_data["old_logprobs"] = old_logprobs[
                    start_idx : start_idx + self.config.batch_size,
                    : segment_seq_len - 1,
                ]
            batch_data["baselined_rewards"] = baselined_rewards[
                start_idx : start_idx + self.config.batch_size
            ]

            self.generation_buffer.append(batch_data)

    def eval_func_batched(
        self, func, batch_inputs: Dict[str, Any], pass_inputs: Dict[str, Any]
    ) -> torch.Tensor:
        outs = []

        lenv = [len(v) for v in batch_inputs.values()]
        assert len(set(lenv)) == 1, lenv

        for start_idx in range(0, lenv[0], self.config.batch_size):
            slice = {
                k: v[start_idx : start_idx + self.config.batch_size]
                for k, v in batch_inputs.items()
            }
            out = func(**slice, **pass_inputs)
            outs.append(out)

        return torch.vstack(outs)

    def get_steering_vector(self, layer: nn.Module, layer_idx: int):
        if (
            self.config.steering_at_layer is not None
            and layer_idx != self.config.steering_at_layer
        ):
            return None

        if (
            isinstance(self.model, PeftModelForCausalLM)
            and layer.mlp.down_proj.lora_B.default.weight.data.shape[-1] == 1
        ):
            steering_vector = layer.mlp.down_proj.lora_B.default.weight.data.squeeze(-1)
        elif (
            (
                self.config.training_setup == "steering_rank"
                and hasattr(self.get_inner_model(), "steering_matrix_A")
            )
            or (
                self.config.training_setup == "tt_rank"
                and hasattr(self.get_inner_model(), "cores")
            )
            or (
                self.config.training_setup == "steering_at_place"
                and hasattr(self.get_inner_model(), "steering_matrix")
            )
        ):
            steering_vectors = self.get_inner_model().get_steering_vectors()
            steering_vector = steering_vectors[layer_idx]
        elif layer.mlp.down_proj.bias is not None:
            steering_vector = layer.mlp.down_proj.bias.data
        else:
            steering_vector = None

        return steering_vector

    def get_logprobs_from_inputs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        generation_temperature: float,
        record_hidden_states: bool,
    ):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logprobs = get_logprobs(
            logits=logits,
            input_ids=input_ids,
            generation_temperature=generation_temperature,
        )

        # if record_hidden_states:
        #     with torch.no_grad():
        #         for layer_idx, layer in enumerate(self.get_inner_model().model.layers):
        #             hidden_states = self.hiddens_values[f"layer_{layer_idx}_hiddens"]

        #             steering_vector = self.get_steering_vector(
        #                 layer=layer, layer_idx=layer_idx
        #             )
        #             if steering_vector is None:
        #                 continue

        #             post_steering_hiddens_norm = torch.norm(
        #                 hidden_states.detach(), dim=-1
        #             ).flatten()
        #             pre_steering_hiddens_norm = torch.norm(
        #                 hidden_states.detach()
        #                 - steering_vector.unsqueeze(0).unsqueeze(0),
        #                 dim=-1,
        #             ).flatten()
        #             steering_vector_norm = torch.norm(steering_vector)
        #             ratio = steering_vector_norm / pre_steering_hiddens_norm
        #             self.store_metrics(
        #                 metrics={
        #                     f"post_steering_hidden_norm_at_{layer_idx}": post_steering_hiddens_norm,
        #                     f"pre_steering_hidden_norm_at_{layer_idx}": pre_steering_hiddens_norm,
        #                     f"steering_vector_over_pre_hiddens_ratio_at_{layer_idx}": ratio,
        #                 }
        #             )

        return logprobs

    def get_invalid_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens_mask: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
    ):
        return torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)

    def calc_num_items_in_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens_mask: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
        baselined_rewards: torch.Tensor,
    ) -> None:
        invalid_mask = self.get_invalid_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_tokens_mask=prompt_tokens_mask,
            generation_tokens_mask=generation_tokens_mask,
        )

        self.num_gen_tokens_in_batch = generation_tokens_mask.sum(-1)[
            ~invalid_mask
        ].sum()
        self.num_gen_tokens_in_batch = (
            self.accelerator.gather(self.num_gen_tokens_in_batch).sum().item()
        )
        self.num_prompts_in_batch = torch.sum((~invalid_mask).to(torch.long))
        self.num_prompts_in_batch = (
            self.accelerator.gather(self.num_prompts_in_batch).sum().item()
        )

        # self.store_metrics(
        #     metrics={"num_gen_tokens_in_batch": float(self.num_gen_tokens_in_batch)}
        # )

    def check_start_step_health(self):
        if self.prev_should_evaluate is not None:
            if self.prev_should_evaluate != self.is_evaluating:
                assert len(self.generation_buffer) == 0, len(self.generation_buffer)

        self.prev_should_evaluate = self.is_evaluating

        if self.prev_global_step is not None:
            if self.prev_global_step != self.state.global_step:
                assert len(self.generation_buffer) == 0, len(self.generation_buffer)

        self.prev_global_step = self.state.global_step

        assert self.is_evaluating is not None

    @torch.no_grad()
    def calc_steering_vectors_norm(self):
        steering_vectors_norms = []
        for layer_idx, layer in enumerate(self.get_inner_model().model.layers):
            steering_vector = self.get_steering_vector(layer=layer, layer_idx=layer_idx)
            if steering_vector is None:
                continue

            steering_vector_norm = torch.norm(steering_vector, p=2).item()
            steering_vectors_norms.append(steering_vector_norm)

        self.store_metrics({"steering_vectors_norms": steering_vectors_norms})

    def get_num_same_rewards(self, rewards: torch.Tensor):
        rewards_ = rewards.reshape(-1, self.get_num_generations())

        return (rewards_.min(-1).values == rewards_.max(-1).values).sum().item()

    @torch.no_grad()
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        self.sampling_params.n = self.get_num_generations()
        if self.config.reward_processor_type in [
            RewardProcessorType.RLOO,
            RewardProcessorType.GRPO,
            RewardProcessorType.GRPO_NO_STD,
        ]:
            self.reward_processor.num_generations = self.get_num_generations()
        self.check_start_step_health()

        if len(self.generation_buffer) == 0:
            self.prompt_indices_at_this_step = set(
                self.accelerator.gather(inputs["index"]).tolist()
            )

            (
                input_ids,
                attention_mask,
                prompt_tokens_mask,
                generation_tokens_mask,
                position_ids,
                completions_metrics,
                completions_time_metrics,
            ) = self.get_completions(inputs=inputs)

            with Timeit() as reference_forward_time:
                if self.ref_model is not None:
                    ref_logprobs = self.eval_func_batched(
                        func=self.get_logprobs_from_inputs,
                        batch_inputs=dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        ),
                        pass_inputs=dict(
                            model=self.get_ref_logits,
                            generation_temperature=self.config.generation_temperature,
                            record_hidden_states=False,
                        ),
                    )
                else:
                    ref_logprobs = None

            with Timeit() as policy_forward_time:
                if False:
                    if not self.is_evaluating:
                        self.model.eval()

                    old_logprobs = self.eval_func_batched(
                        func=self.get_logprobs_from_inputs,
                        batch_inputs=dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        ),
                        pass_inputs=dict(
                            model=(
                                lambda input_ids,
                                attention_mask,
                                position_ids: self.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    use_cache=False,
                                ).logits
                            ),
                            generation_temperature=self.config.generation_temperature,
                            record_hidden_states=False,
                        ),
                    )
                    old_logprobs = old_logprobs.detach()

                    if not self.is_evaluating:
                        self.model.train()
                else:
                    old_logprobs = None

            with Timeit() as reward_time:
                raw_rewards = self.get_rewards(
                    input_ids=input_ids,
                    generation_tokens_mask=generation_tokens_mask,
                    answer_token_ids=inputs["answer_input_ids"],
                    answer_token_mask=inputs["answer_attention_mask"],
                )
                raw_rewards = raw_rewards.detach()  # just in case

                assert raw_rewards.size(0) == input_ids.size(0), (
                    raw_rewards.shape,
                    input_ids.shape,
                )
                assert raw_rewards.dim() == 1, raw_rewards.shape

            with Timeit() as reward_processing_time:
                valid_mask = self.get_valid_mask(
                    input_ids=input_ids,
                    attention_mask=generation_tokens_mask,
                )
                rewards = self.process_rewards(
                    rewards=raw_rewards,
                    valid_mask=valid_mask,
                    generation_tokens_mask=generation_tokens_mask,
                )

            if self.config.kl_is_summand_to == "reward" and self.config.kl_coef > 0.0:
                with Timeit() as kl_time:
                    kl_term = self.get_kl_term(
                        input_ids=input_ids,
                        policy_logprobs=old_logprobs,
                        reference_policy_logprobs=ref_logprobs,
                        tokens_mask=generation_tokens_mask,
                        reduce_strategy=self.config.kl_reduce_strategy,
                    )

                regularized_rewards = rewards - self.kl_coef * kl_term

            else:
                regularized_rewards = rewards

            with Timeit() as baseline_reward_time:
                invalid_mask = self.get_invalid_mask(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_tokens_mask=prompt_tokens_mask,
                    generation_tokens_mask=generation_tokens_mask,
                )
                baselined_rewards, baseline_metrics = (
                    self.reward_processor.baseline_rewards(
                        rewards=regularized_rewards, invalid_mask=invalid_mask
                    )
                )
                assert torch.all(
                    baselined_rewards[invalid_mask] == 0.0
                ) and not torch.any(torch.isnan(baselined_rewards)), [
                    (x.item(), y.item(), z.item(), w.item())
                    for x, y, z, w in zip(
                        invalid_mask,
                        baselined_rewards,
                        rewards,
                        raw_rewards,
                    )
                ]

            self.calc_num_items_in_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_tokens_mask=prompt_tokens_mask,
                generation_tokens_mask=generation_tokens_mask,
                baselined_rewards=baselined_rewards,
            )

            self.fill_buffer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_tokens_mask=prompt_tokens_mask,
                generation_tokens_mask=generation_tokens_mask,
                position_ids=position_ids,
                ref_logprobs=ref_logprobs,
                old_logprobs=old_logprobs,
                baselined_rewards=baselined_rewards,
            )

            self.store_metrics(metrics=completions_metrics)
            self.store_metrics(metrics=completions_time_metrics)
            self.store_metrics(metrics=baseline_metrics)
            self.store_metrics(
                metrics={
                    "raw_rewards": raw_rewards,
                    "rewards": rewards,
                    "regularized_rewards": regularized_rewards,
                    "baselined_reward": baselined_rewards,
                    "num_same_rewards": self.get_num_same_rewards(raw_rewards),
                    "frac_same_rewards": self.get_num_same_rewards(raw_rewards)
                    / (
                        self.accelerator.num_processes
                        * (
                            self.args.per_device_train_batch_size
                            if not self.is_evaluating
                            else self.args.per_device_eval_batch_size
                        )
                    ),
                    "valid_mask": valid_mask.float(),
                    "time_reference_forward": reference_forward_time.elapsed_time_cpu,
                    "time_reward_model": reward_time.elapsed_time_cpu,
                    "time_reward_processing": reward_processing_time.elapsed_time_cpu,
                    "time_baseline_reward": baseline_reward_time.elapsed_time_cpu,
                    "time_policy_forward": policy_forward_time.elapsed_time_cpu,
                },
            )
            if self.config.kl_is_summand_to == "reward":
                self.store_metrics(
                    metrics={
                        "time_kl": kl_time.elapsed_time_cpu,
                        "kl_term": kl_term,
                    }
                )

            self.calc_steering_vectors_norm()

        assert set(inputs["index"].tolist()) <= self.prompt_indices_at_this_step, (
            set(inputs["index"].tolist()),
            self.prompt_indices_at_this_step,
        )
        return self.generation_buffer.pop()

    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module | PreTrainedModel,
        inputs: dict[str, torch.Tensor],
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_tokens_mask = inputs["prompt_tokens_mask"]
        generation_tokens_mask = inputs["generation_tokens_mask"]
        position_ids = inputs["position_ids"]
        if "ref_logprobs" in inputs:
            ref_logprobs = inputs["ref_logprobs"]
        if "old_logprobs" in inputs:
            old_logprobs = inputs["old_logprobs"]
        baselined_rewards = inputs["baselined_rewards"]

        #####

        # answer_tokens_mask = torch.logical_xor(
        #     attention_mask,
        #     torch.logical_or(prompt_tokens_mask, generation_tokens_mask),
        # ).to(torch.long)
        # if torch.distributed.get_rank() == 0:
        #     for idx in range(len(input_ids)):
        #         loh = [
        #             (
        #                 self.processing_class.decode(iid, skip_special_tokens=False),
        #                 iid.item(),
        #                 am.item(),
        #                 ptm.item(),
        #                 gtm.item(),
        #                 atm.item(),
        #                 pi.item(),
        #             )
        #             for iid, am, ptm, gtm, atm, pi in zip(
        #                 input_ids[idx],
        #                 attention_mask[idx],
        #                 prompt_tokens_mask[idx],
        #                 generation_tokens_mask[idx],
        #                 answer_tokens_mask[idx],
        #                 position_ids[idx],
        #             )
        #         ]

        #         logger.info(
        #             f"Step {self.state.global_step}. Text at index {idx}: {self.processing_class.decode(input_ids[idx], skip_special_tokens=False)}"
        #         )
        #         for slice_idx in range(0, len(loh), 100):
        #             logger.info(
        #                 f"Alignment of tokens and masks at index {idx}; slice: {slice_idx}: {loh[slice_idx : slice_idx + 100]}"
        #             )

        ####

        with Timeit() as policy_forward_time:
            logprobs = self.get_logprobs_from_inputs(
                model=(
                    lambda input_ids, attention_mask, position_ids: self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    ).logits
                ),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                generation_temperature=self.config.generation_temperature,
                record_hidden_states=True,
            )

        with Timeit() as rl_loss_time:
            rl_loss = self.get_rl_loss(
                reward=baselined_rewards,
                logprobs=logprobs,
                generation_tokens_mask=generation_tokens_mask,
            )

        with Timeit() as loss_time:
            loss = -rl_loss

        kl_condition = (
            self.config.kl_is_summand_to == "loss" and self.config.kl_coef > 0.0
        )
        if kl_condition:
            with Timeit() as kl_time:
                kl_term = self.get_kl_term(
                    input_ids=input_ids,
                    policy_logprobs=logprobs,
                    reference_policy_logprobs=ref_logprobs,
                    tokens_mask=generation_tokens_mask,
                    reduce_strategy="sum",
                )

            loss = loss + self.kl_coef * kl_term

        loss = loss / self.num_gen_tokens_in_batch

        assert len(loss.shape) == 1, loss.shape

        with torch.no_grad():
            with Timeit() as metrics_time:
                metrics = dict(
                    loss=loss.detach(),
                    generation_logprobs=-reduce_logprobs(
                        logprobs=logprobs.detach(),
                        tokens_mask=generation_tokens_mask,
                        reduce_strategy="sum",
                    ),
                )
                if self.ref_model is not None:
                    metrics["ref_generation_logprob"] = -reduce_logprobs(
                        logprobs=ref_logprobs,
                        tokens_mask=generation_tokens_mask,
                        reduce_strategy="sum",
                    )
                if kl_condition:
                    metrics["kl_term"] = kl_term.detach()
                    metrics["time_kl"] = kl_time.elapsed_time_cpu

                metrics["time_policy_forward"] = policy_forward_time.elapsed_time_cpu
                metrics["time_rl_loss"] = rl_loss_time.elapsed_time_cpu
                metrics["time_loss"] = loss_time.elapsed_time_cpu

            metrics["time_metrics"] = metrics_time.elapsed_time_cpu

        return loss.sum(), metrics

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        with Timeit() as total_time:
            loss, metrics = self.get_batch_loss_metrics(model=model, inputs=inputs)

        metrics["time_total"] = total_time.elapsed_time_cpu

        self.store_metrics(metrics=metrics)

        loss = loss * self.args.gradient_accumulation_steps
        loss = loss * self.accelerator.num_processes

        self.prev_should_evaluate = self.is_evaluating
        self.prev_global_step = self.state.global_step

        return (loss, metrics) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        self.is_evaluating = False
        self.curent_metrics_prefix = ""
        return super().training_step(
            model=model, inputs=inputs, num_items_in_batch=num_items_in_batch
        )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self.is_evaluating = True
        self.curent_metrics_prefix = f"{metric_key_prefix}_"
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs=inputs)

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model=model, inputs=inputs)

        self.store_metrics(metrics=metrics)

        return loss.detach(), None, None

    def process_rewards(
        self,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor,
        generation_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Second, this token should be the <eot_id>.
        Otherwise, return a fixed reward of -1.
        """

        # discount the rewards
        generation_lengths = generation_tokens_mask.sum(dim=1)
        discount = self.config.gamma**generation_lengths
        rewards = rewards * discount

        return rewards

    def store_metrics(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self._stored_metrics[self.curent_metrics_prefix][key].append(value)

    @torch.no_grad()
    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        train_eval_keys = list(self._stored_metrics.keys())

        sum_keys = ["metrics_num_prompts", "num_same_rewards", "frac_same_rewards"]
        for at in [1, 2, 4, 8, 16, 32, 64, 128]:
            sum_keys.extend(
                [
                    f"metrics_at_{at}_num_solved",
                    f"metrics_at_{at}_num_solved_strict",
                    f"metrics_at_{at}_frac_solved",
                    f"metrics_at_{at}_frac_solved_strict",
                ]
            )

        for train_eval_key in train_eval_keys:
            metric_keys = self._stored_metrics[train_eval_key].keys()
            for metric_key in metric_keys:
                metrics = self._stored_metrics[train_eval_key][metric_key]
                if not isinstance(metrics, (list, tuple)):
                    metrics = [metrics]

                for metric_idx in range(len(metrics)):
                    assert isinstance(
                        metrics[metric_idx], (int, bool, float, torch.Tensor, list)
                    ), (
                        metric_key,
                        type(metrics[metric_idx]),
                        metrics[metric_idx],
                    )
                    if isinstance(metrics[metric_idx], torch.Tensor):
                        if metrics[metric_idx].dim() != 1:
                            metrics[metric_idx] = metrics[metric_idx].squeeze()
                        assert metrics[metric_idx].dim() in [0, 1], (
                            metric_key,
                            metrics[metric_idx].dim(),
                            metrics[metric_idx].shape,
                            metrics[metric_idx],
                        )

                if all(isinstance(metric, torch.Tensor) for metric in metrics):
                    metrics = torch.hstack(metrics).to(self.accelerator.device)
                elif all(isinstance(metric, (list, tuple)) for metric in metrics):
                    metrics = torch.from_numpy(np.hstack(metrics)).to(
                        self.accelerator.device
                    )
                else:
                    metrics = torch.tensor(metrics).to(self.accelerator.device)

                if metric_key in sum_keys:
                    logs[f"{train_eval_key}{metric_key}"] = get_sum(
                        tensor=metrics, use_global=True
                    ).item()
                else:
                    try:
                        mean_val, std_val = get_mean_std(
                            tensor=metrics.cuda(), use_global=True
                        )
                    except Exception as e:
                        logger.info(
                            f"Exception {str(e)} for key {metric_key} with value {metrics}"
                        )

                    logs[f"{train_eval_key}{metric_key}_mean"] = mean_val.item()
                    logs[f"{train_eval_key}{metric_key}_std"] = std_val.item()

                if metric_key == "metrics_num_solved_strict":
                    self.best_num_solved_strict[f"{train_eval_key}{metric_key}"] = max(
                        self.best_num_solved_strict[f"{train_eval_key}{metric_key}"],
                        logs[f"{train_eval_key}{metric_key}"],
                    )
                    logs[f"{train_eval_key}{metric_key}_best"] = (
                        self.best_num_solved_strict[f"{train_eval_key}{metric_key}"]
                    )

            if train_eval_key == "":
                logs[f"{train_eval_key}global_step"] = int(self.state.global_step)
            else:
                logs[f"{train_eval_key}global_step"] = int(
                    self.state.global_step // self.config.eval_steps
                )

            if (
                False
                and torch.distributed.get_rank() == 0
                and self.state.global_step % self.config.eval_steps == 0
            ):
                completion_lens = torch.hstack(
                    self._stored_metrics[train_eval_key]["metrics_completion_lens"]
                )
                rewards = torch.hstack(self._stored_metrics[train_eval_key]["rewards"])
                is_solved_strict = torch.hstack(
                    self._stored_metrics[train_eval_key]["metrics_is_solved_strict"]
                ).to(torch.bool)

                plt.scatter(
                    completion_lens[~is_solved_strict].cpu().numpy(),
                    rewards[~is_solved_strict].cpu().numpy(),
                    label="Not solved",
                )
                plt.scatter(
                    completion_lens[is_solved_strict].cpu().numpy(),
                    rewards[is_solved_strict].cpu().numpy(),
                    label="Solved",
                )
                train_or_eval = "train" if train_eval_key == "" else "eval"
                plt.title(f"Completion lens vs rewards. {train_or_eval}")
                plt.xlabel("Completion lens")
                plt.ylabel("Rewards")
                plt.legend(loc="best")
                log_name = f"completion_lens_vs_rewards_{train_or_eval}"
                wandb.log({log_name: wandb.Image(plt)})

                plt.close()

            del self._stored_metrics[train_eval_key]

        return super().log(logs)
