from datetime import timedelta

import pyrallis
import ray

from steering_reasoning.train.rl.config import Config
from steering_reasoning.train.rl.policy_model import PolicyModel
from steering_reasoning.train.rl.ray_workers.raygroup import RayGroup
from steering_reasoning.train.rl.ray_workers.vllm_engine import create_vllm_engines
from steering_reasoning.utils.utils import set_logger


@pyrallis.wrap()
def main(config: Config) -> None:
    set_logger(verbose=False)
    ray.init(address="auto")
    # Init wandb. It is here so that the run is created in the main process
    # run by wandb agent
    # wandb_run = wandb_init(config=config)

    policy_models = RayGroup(
        num_nodes=config.policy_num_nodes,
        num_gpus_per_node=config.policy_num_gpus_per_node,
        ray_actor_type=PolicyModel,
    )

    assert 1 <= config.reward_model_replicas <= 8, config.reward_model_replicas

    # TODO_RLOO if possible hide init inside RayGroup
    # TODO add settings fields to reward model
    ray.get(
        policy_models.async_init_model_from_pretrained(
            seed=config.seed, timeout=timedelta(seconds=config.deepspeed_timeout)
        )
    )

    """
    TODO_RLOO:
    2. PARAMS to REINFORCETrainExperimentSettings
    3. if possible hide creating of vllm engines inside trainer
    """

    vllm_engines = create_vllm_engines(
        num_engines=config.vllm_num_engines,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        pretrain_model_path=config.model_path,
        seed=config.seed,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=config.max_seq_length,
        enable_lora=config.lora_rank is not None,
        lora_savepath=config.lora_savepath,
    )

    # We need to finish it before we can start training
    # otherwise, when it is finished afterwards, it rewrites summary
    # and it is not correctly displayed on a sweep panel
    #
    # finish with a `failed` status code so it is continued later.
    # in the `Trainer.init`
    # wandb.finish(exit_code=1, quiet=True)

    ray.get(
        policy_models.async_fit_actor_model(
            config=config,
            vllm_engines=vllm_engines,  # , wandb_run_id=wandb_run.id
        )
    )


if __name__ == "__main__":
    main()
