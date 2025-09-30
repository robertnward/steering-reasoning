import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from steering_reasoning.utils.utils import set_logger


@dataclass
class Config:
    seed: int = 0
    model_path: str = "/from_s3/models/Qwen2.5-Math-1.5B"
    savedir: str = "plots"
    verbose: bool = False

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def calc_stable_rank(A):
    s = np.linalg.svd(A, compute_uv=False)
    return (s**2).sum() / (s[0] ** 2)


def draw_pairwise_cossims(steering_vectors, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    V = np.vstack(steering_vectors)

    # L2‑normalize rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("At least one vector has zero norm.")
    Vn = V / norms

    # Cosine similarity matrix
    sim_matrix = Vn @ Vn.T  # (N, N)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(sim_matrix, vmin=-1, vmax=1, aspect="equal")  # default colormap
    plt.colorbar(im, label="Cosine similarity")
    plt.title("Pairwise cosine similarity")
    plt.xlabel("Vector index")
    plt.ylabel("Vector index")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def draw_singular_values(steering_vectors, savepath):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    s = np.linalg.svd(np.vstack(steering_vectors), compute_uv=False)  # σ₁ ≥ σ₂ ≥ …

    # --- make the plot ---
    idx = np.arange(1, len(s) + 1)  # 1-based index
    plt.plot(idx, s, marker="o")  # a single line with dots
    plt.xlabel("Singular-value index")
    plt.ylabel("Singular value σᵢ")
    plt.title("Singular values of Steering Vector Matrix")

    # Optional: reveal tiny σᵢ better
    # plt.yscale("log")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def draw_pairwise_cossims_hist(
    steering_vectors, savedir: Optional[str] = None, savepath: Optional[str] = None
):
    V = np.vstack(steering_vectors)

    # L2‑normalize rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("At least one vector has zero norm.")
    Vn = V / norms

    # Cosine similarity matrix
    sim_matrix = Vn @ Vn.T  # (N, N)
    mask = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)
    sims = sim_matrix[mask]

    plt.figure(figsize=(12, 8))
    plt.hist(sims)
    plt.title("Pairwise cosine similarity")
    plt.xlabel("Cossim")
    plt.ylabel("Count")
    plt.tight_layout()
    assert savedir or savepath, (savedir, savepath)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "cossims_hist.png"))
    elif savepath is not None:
        plt.savefig(savepath)

    plt.close()


def draw_norms(steering_vectors, savedir: str):
    norms = [np.linalg.norm(x) for x in steering_vectors]
    plt.figure(figsize=(12, 8))
    plt.hist(norms)
    plt.title("Steering vectors' norms")
    plt.xlabel("Norm")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "norms_hist.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(norms)), norms)
    plt.title("Steering vectors' norms by layer")
    plt.xlabel("Layer idx")
    plt.ylabel("Norm")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "norms_by_layer.png"))
    plt.close()


def save_logit_lens(
    top_tokens_with_scores: List[List[Tuple[str, float]]], savepath: str
):
    for layer_idx in range(len(top_tokens_with_scores)):
        with open(savepath, "+a") as f:
            f.write(f"Logit Lens at {layer_idx}:\n")
            for token, prob in top_tokens_with_scores[layer_idx]:
                f.write(f"{token}, {prob}\n")

            f.write("=" * 100)
            f.write("\n\n")


@torch.no_grad()
def get_logit_lens(
    ranking_type: Literal["prob", "cossim"],
    model: AutoModelForCausalLM,
    steering_vectors: List[torch.Tensor],
    tokenizer: PreTrainedTokenizer,
):
    unembed_matrix = model.get_output_embeddings()
    unembed_matrix_normed = unembed_matrix.weight.data / torch.linalg.norm(
        unembed_matrix.weight.data, dim=-1, keepdims=True
    )
    tokens = []
    for steering_vector in steering_vectors:
        if ranking_type == "prob":
            token_distributions = unembed_matrix(
                torch.from_numpy(steering_vector).to(model.device).to(model.dtype)
            )
            token_distributions = torch.nn.functional.softmax(
                token_distributions, dim=-1
            )
            top_probs, top_token_ids = torch.topk(token_distributions, k=50)

            top_tokens = [
                tokenizer.decode(x, skip_special_tokens=False) for x in top_token_ids
            ]
            top_tokens_with_scores = list(zip(top_tokens, top_probs.tolist()))
        elif ranking_type == "cossim":
            steering_vector_normed = steering_vector / np.linalg.norm(steering_vector)
            cossim = unembed_matrix_normed @ torch.from_numpy(
                steering_vector_normed
            ).to(model.device).to(model.dtype)
            top_sims, top_token_ids = torch.topk(cossim, k=50)
            top_tokens = [
                tokenizer.decode(x, skip_special_tokens=False) for x in top_token_ids
            ]
            top_tokens_with_scores = list(zip(top_tokens, top_sims.tolist()))
        else:
            raise NotImplementedError(f"Unknown ranking_type {ranking_type}")

        tokens.append(top_tokens_with_scores)

    return tokens


@pyrallis.wrap()
def main(config: Config):
    set_logger(config.verbose)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)

    steering_vectors = []
    for layer in model.model.layers:
        steering_vector = (
            layer.steering_vector.steering_vector.data.squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        steering_vectors.append(steering_vector)

    draw_norms(steering_vectors=steering_vectors, savedir=config.savedir)
    draw_pairwise_cossims(
        steering_vectors=steering_vectors,
        savepath=os.path.join(config.savedir, "cossims.png"),
    )
    draw_singular_values(
        steering_vectors=steering_vectors,
        savepath=os.path.join(config.savedir, "singular_values.png"),
    )
    stable_rank = calc_stable_rank(steering_vectors)
    logger.info(f"Stable rank: {stable_rank}")

    top_tokens_with_scores = get_logit_lens(
        ranking_type="prob",
        model=model,
        steering_vectors=steering_vectors,
        tokenizer=tokenizer,
    )
    save_logit_lens(
        top_tokens_with_scores=top_tokens_with_scores,
        savepath=os.path.join(config.savedir, "logit_lens.txt"),
    )

    top_tokens_with_scores = get_logit_lens(
        ranking_type="cossim",
        model=model,
        steering_vectors=steering_vectors,
        tokenizer=tokenizer,
    )
    save_logit_lens(
        top_tokens_with_scores=top_tokens_with_scores,
        savepath=os.path.join(config.savedir, "logit_lens_cossim.txt"),
    )


if __name__ == "__main__":
    main()
