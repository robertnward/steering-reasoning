from dataclasses import dataclass

import pyrallis
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


@dataclass
class Config:
    base_model: str
    trained_model: str


@pyrallis.wrap()
def run(config: Config):
    with torch.no_grad():
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
        trained_model = AutoModelForCausalLM.from_pretrained(config.trained_model)

    for (base_name, base_param), (trained_name, trained_param) in tqdm(
        zip(base_model.named_parameters(), trained_model.named_parameters())
    ):
        assert base_name == trained_name, (base_name, trained_name)

        if "down_proj" in base_name and "bias" in base_name:
            assert not torch.allclose(base_param, trained_param), base_name
        else:
            assert torch.allclose(base_param, trained_param), trained_name


if __name__ == "__main__":
    run()
