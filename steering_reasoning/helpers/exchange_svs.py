from dataclasses import dataclass

import pyrallis
import torch
from loguru import logger
from transformers import AutoModelForCausalLM


@dataclass
class Config:
    model_from_path: str
    model_to_path: str


@torch.no_grad()
def get_steering_vectors(model):
    steering_vectors = []
    for layer in model.model.layers:
        steering_vector = layer.mlp.down_proj.bias.data
        steering_vectors.append(steering_vector)

    steering_vectors = torch.vstack(steering_vectors)

    return steering_vectors


@pyrallis.wrap()
def main(config: Config):
    model_from = AutoModelForCausalLM.from_pretrained(
        config.model_from_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda()
    model_to = AutoModelForCausalLM.from_pretrained(
        config.model_to_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda()

    assert len(model_from.model.layers) == len(model_to.model.layers), (
        len(model_from.model.layers),
        len(model_to.model.layers),
    )
    num_layers = len(model_from.model.layers)

    for layer in range(num_layers):
        model_to.model.layers[layer].mlp.down_proj.bias.data = model_from.model.layers[
            layer
        ].mlp.down_proj.bias.data

    model_to.save_pretrained(config.model_to_path)
    logger.info(f"Saved to {config.model_to_path}")


if __name__ == "__main__":
    main()
