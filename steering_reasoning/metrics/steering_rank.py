import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
from accelerate.utils import set_seed
from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from steering_reasoning.metrics.plots import draw_pairwise_cossims
from steering_reasoning.models.model_with_steering_rank import get_wrapper


@dataclass
class Config:
    seed: int
    model_path: str
    steering_rank: int

    savedir: str


@pyrallis.wrap()
def run(config: Config):
    set_seed(seed=config.seed, device_specific=False, deterministic=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if "Qwen2" in config.model_path:
        model_cls = Qwen2ForCausalLM
    elif "llama3" in config.model_path:
        model_cls = LlamaForCausalLM
    else:
        raise NotImplementedError(config.model_path)

    model_cls = get_wrapper(model_cls, steering_rank=config.steering_rank)

    model = model_cls.from_pretrained(
        config.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.eval().to(device)

    with torch.no_grad():
        steering_matrix_A = model.steering_matrix_A.float().cpu().numpy()
        steering_matrix_B = model.steering_matrix_B.float().cpu().numpy()
        steering_vectors = steering_matrix_A @ steering_matrix_B

        draw_pairwise_cossims(
            steering_vectors, os.path.join(config.savedir, "cossims.png")
        )
        draw_pairwise_cossims(
            steering_matrix_B, os.path.join(config.savedir, "cossims-B.png")
        )
        draw_pairwise_cossims(
            steering_matrix_A, os.path.join(config.savedir, "cossims-A.png")
        )

        if steering_matrix_A.shape[1] == 1:
            scalings = steering_matrix_A.squeeze(1) * np.linalg.norm(
                steering_matrix_B.squeeze(0)
            )
            os.makedirs(config.savedir, exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(scalings)), scalings)
            plt.xlabel("Layer")
            plt.ylabel("Scaling (A)")
            plt.title("Steering Vector Scalings for Each Layer")
            plt.tight_layout()
            plt.savefig(os.path.join(config.savedir, "scalings.png"))
            plt.close()


if __name__ == "__main__":
    run()
