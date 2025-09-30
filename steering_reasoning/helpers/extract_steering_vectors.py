import os
from dataclasses import dataclass

import numpy as np
import pyrallis
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


@dataclass
class Config:
    model_path: str
    save_path: str

    def __post_init__(self):
        os.makedirs(self.save_path, exist_ok=True)


@torch.no_grad()
def get_vanilla_steering_vectors(config: Config):
    model = AutoModelForCausalLM.from_pretrained(config.model_path)

    steering_vectors = []
    for layer in model.model.layers:
        steering_vector = layer.mlp.down_proj.bias.data.cpu().numpy()
        steering_vectors.append(steering_vector)

    steering_vectors = np.vstack(steering_vectors)

    return steering_vectors


@torch.no_grad()
def get_rank_steering_vectors(config: Config):
    path = os.path.join(config.model_path, "model.safetensors")

    if os.path.isfile(path):
        state_dict = load_file(path, device="cpu")
        if "steering_matrix_A" in state_dict.keys():
            steering_matrix_A = state_dict["steering_matrix_A"].float().cpu().numpy()
            steering_matrix_B = state_dict["steering_matrix_B"].float().cpu().numpy()

            steerint_vectors = steering_matrix_A @ steering_matrix_B

            return steerint_vectors

    return


@pyrallis.wrap()
def main(config: Config):
    steering_vectors = get_rank_steering_vectors(config=config)

    if steering_vectors is None:
        steering_vectors = get_vanilla_steering_vectors(config=config)

    np.save(os.path.join(config.save_path, "steering_vectors.npy"), steering_vectors)


if __name__ == "__main__":
    main()
