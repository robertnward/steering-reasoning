from dataclasses import dataclass

import pyrallis
import torch
from transformers import AutoModelForCausalLM


@dataclass
class Config:
    base_model: str
    trained_model: str

    model_family: str

    def __post_init__(self):
        assert self.model_family in ["Qwen2", "llama3"]


@pyrallis.wrap()
def run(config: Config):
    with torch.no_grad():
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
        trained_model = AutoModelForCausalLM.from_pretrained(config.trained_model)

    for (base_name, base_param), (trained_name, trained_param) in zip(
        base_model.named_parameters(), trained_model.named_parameters()
    ):
        assert base_name == trained_name, (base_name, trained_name)

        if "steering" in base_name:
            assert not torch.allclose(base_param, trained_param), base_name
        else:
            assert torch.allclose(base_param, trained_param), trained_name


if __name__ == "__main__":
    run()
