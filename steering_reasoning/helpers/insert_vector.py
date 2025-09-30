from dataclasses import dataclass

import numpy as np
import pyrallis
import torch
from transformers import AutoModelForCausalLM


@dataclass
class Config:
    steer_multiply: float


@pyrallis.wrap()
def run(config: Config):
    model = AutoModelForCausalLM.from_pretrained(
        "/from_s3/model/",
        torch_dtype=torch.bfloat16,
    ).cuda()
    for layer_idx in range(len(model.model.layers)):
        model.model.layers[layer_idx].mlp.down_proj.bias.data = torch.zeros_like(
            model.model.layers[layer_idx].mlp.down_proj.bias.data
        )

    vector = np.load("/from_s3/steering_vector/to_feature.npy")
    vector = torch.from_numpy(vector).cuda().to(torch.bfloat16)

    print("steer_multiply", config.steer_multiply)
    model.model.layers[27].mlp.down_proj.bias.data = (
        config.steer_multiply * 8.57 * vector
    )

    model.save_pretrained("/from_s3/model/")


if __name__ == "__main__":
    run()
