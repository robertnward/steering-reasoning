from typing import List, Optional

import ray
import torch
import vllm
from loguru import logger
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import RequestOutput

from steering_reasoning.utils.utils import zip_bytes_to_dir


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        kwargs["worker_extension_cls"] = (
            "steering_reasoning.train.rl.ray_workers.vllm_worker_wrap.WorkerWrap"
        )
        self.lora_savepath = kwargs.pop("lora_savepath", None)
        self.lora_counter = 1
        self.vllm_actor = vllm.LLM(*args, **kwargs)

    def generate(
        self, prompt_token_ids: torch.Tensor, sampling_params: vllm.SamplingParams
    ) -> List[RequestOutput]:
        logger.debug(f"PROMPT TOKEN IDS IN VLLM: {prompt_token_ids}")

        if self.lora_savepath is not None:
            lora_request = vllm.lora.request.LoRARequest(
                "lora", self.lora_counter, self.lora_savepath
            )
            self.lora_counter += 1
        else:
            lora_request = None

        return self.vllm_actor.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )

    def init_weight_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
    ):
        if self.use_gpu_executor:
            return self.vllm_actor.collective_rpc(
                "init_weight_update_group",
                args=(master_address, master_port, rank_offset, world_size),
            )
        else:
            return self.vllm_actor.llm_engine.model_executor._run_workers(
                "init_weight_update_group",
                master_address,
                master_port,
                rank_offset,
                world_size,
            )

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
    ):
        if self.use_gpu_executor:
            return self.vllm_actor.collective_rpc(method, timeout, args, kwargs)
        else:
            return self.vllm_actor.llm_engine.model_executor._run_workers(
                "collective_rpc",
                method,
                timeout,
                args,
                kwargs,
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.vllm_actor.collective_rpc(
                "update_weight", args=(name, dtype, shape, empty_cache)
            )
        else:
            return self.vllm_actor.llm_engine.model_executor._run_workers(
                "update_weight", name, dtype, shape, empty_cache
            )

    def accept_lora(self, lora_bytes):
        zip_bytes_to_dir(data=lora_bytes, dest=self.lora_savepath)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.vllm_actor.collective_rpc("stop_remote_worker_execution_loop", args=())


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain_model_path: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    enable_lora: bool,
    lora_savepath: Optional[str],
):
    vllm_engines = []
    for i in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain_model_path,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
                max_seq_len_to_capture=max_model_len * 2,
                enable_lora=enable_lora,
                lora_savepath=lora_savepath,
            )
        )

    return vllm_engines
