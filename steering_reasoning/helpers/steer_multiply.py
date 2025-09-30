from dataclasses import dataclass

import pyrallis
import torch
from transformers import AutoModelForCausalLM


@dataclass
class Config:
    factor: float


@pyrallis.wrap()
def run(config: Config):
    model = AutoModelForCausalLM.from_pretrained(
        "/from_s3/model/", torch_dtype=torch.bfloat16
    )
    model = model.cuda()

    for layer_idx in range(len(model.model.layers)):
        model.model.layers[layer_idx].mlp.down_proj.bias.data *= config.factor

    model.save_pretrained("/from_s3/model/")


if __name__ == "__main__":
    run()
