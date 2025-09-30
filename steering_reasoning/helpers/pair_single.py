from dataclasses import dataclass

import numpy as np
import pyrallis
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from steering_reasoning.eval.eval import Config as EvalConfig


@dataclass
class Config(EvalConfig):
    index1: int
    index2: int
    steering_vectors_path: str


@pyrallis.wrap()
def run(config: Config):
    steering_vectors = np.load(config.steering_vectors_path)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    for layer in range(len(model.model.layers)):
        model.model.layers[layer].mlp.down_proj.bias.data = torch.zeros_like(
            model.model.layers[layer].mlp.down_proj.bias.data
        )

    model.model.layers[config.index1].mlp.down_proj.bias.data = torch.tensor(
        steering_vectors[config.index1],
        dtype=model.model.layers[config.index1].mlp.down_proj.bias.data.dtype,
        device=model.model.layers[config.index1].mlp.down_proj.bias.data.device,
    )
    model.model.layers[config.index2].mlp.down_proj.bias.data = torch.tensor(
        steering_vectors[config.index2],
        dtype=model.model.layers[config.index2].mlp.down_proj.bias.data.dtype,
        device=model.model.layers[config.index2].mlp.down_proj.bias.data.device,
    )

    model.save_pretrained(config.model_path)
    logger.info(
        f"I have set steering vectors for layers {config.index1} and {config.index2}"
    )


if __name__ == "__main__":
    run()
