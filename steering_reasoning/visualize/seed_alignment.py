import os
import re
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
from matplotlib.ticker import MultipleLocator

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.helpers.merge_vectors_from_layers import (
    get_qwen2_5_math_7b_model_paths,
)
from steering_reasoning.visualize.utils.defs import Model, ModelGroup
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    savedir: str

    def __post_init__(self):
        self.savedir = os.path.join(self.savedir, "seed_alignment")
        os.makedirs(self.savedir, exist_ok=True)


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a, b : arrays with shape (N, d)
    returns: array with shape (N,) containing the cosine similarity
             between a[i] and b[i] for every i
    """
    # Dot-product for each pair of rows → shape (N,)
    numerators = (a * b).sum(axis=1)

    # Row-wise ℓ2 norms → shape (N,)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Guard against divide-by-zero
    denom = a_norm * b_norm
    denom[denom == 0] = 1e-12

    return numerators / denom


def plot_cosine_barplot(
    cosine_values,
    perf_diffs,
    path="cosine_perf_barplot.png",
    *,
    bar_width=0.4,
    show=False,
    cos_kwargs=None,
    diff_kwargs=None,
):
    """
    Draw a grouped bar chart of cosine similarities (left y-axis) and
    perf differences (right y-axis).

    Parameters
    ----------
    cosine_values : array-like, shape (N,)
        Row-wise cosine similarities.
    perf_diffs : array-like, shape (N,)
        Performance differences for the same layers.
    path : str, default 'cosine_perf_barplot.png'
        Output filename (any extension matplotlib supports).
    bar_width : float, default 0.4
        Width of each individual bar (for both groups).
    show : bool, default False
        If True, display the figure interactively.
    cos_kwargs : dict or None
        Extra keyword args forwarded to the cosine-similarity bars.
    diff_kwargs : dict or None
        Extra keyword args forwarded to the perf-diff bars.

    Returns
    -------
    str
        The path where the figure was saved.
    """
    # ---------- preparation ----------
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cosine_values = np.asarray(cosine_values)
    perf_diffs = np.asarray(perf_diffs)

    if cosine_values.shape != perf_diffs.shape:
        raise ValueError("`cosine_values` and `perf_diffs` must have the same length")

    n = len(cosine_values)
    x = np.arange(n)

    cos_kwargs = {} if cos_kwargs is None else cos_kwargs
    diff_kwargs = {} if diff_kwargs is None else diff_kwargs

    # ---------- plotting ----------
    fig, ax_cos = plt.subplots(figsize=(10, 4))

    # Cosine-similarity bars (left axis)
    ax_cos.bar(
        x - bar_width / 2,
        cosine_values,
        width=bar_width,
        label="Cosine similarity",
        **cos_kwargs,
    )
    ax_cos.set_xlabel("Layer")
    ax_cos.set_ylim(0, 1)
    ax_cos.set_ylabel("Cosine similarity")
    ax_cos.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_cos.set_axisbelow(True)
    ax_cos.grid(
        which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0
    )

    # Accuracy-diff bars (right axis, twin)
    ax_diff = ax_cos.twinx()
    ax_diff.bar(
        x + bar_width / 2,
        perf_diffs,
        width=bar_width,
        label="Performance diff",
        color="C1",
        **diff_kwargs,
    )
    ax_diff.set_ylabel("Performance diff")

    # Title & legend
    ax_cos.set_title("Seed alignment of steering vectors — cosine vs. perf diff")

    # Combine legends from both axes
    handles_cos, labels_cos = ax_cos.get_legend_handles_labels()
    handles_diff, labels_diff = ax_diff.get_legend_handles_labels()
    ax_cos.legend(handles_cos + handles_diff, labels_cos + labels_diff, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity for two non-zero 1-D vectors."""
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("Input vectors must be one-dimensional")
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("Vectors must have the same dimensionality")
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        raise ValueError("Vectors must be non-zero")
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def greedy_axis_removal_improve(v1: np.ndarray, v2: np.ndarray, path: str):
    if v1.shape != v2.shape or v1.ndim != 1:
        raise ValueError("v1 and v2 must be 1-D and have the same shape")

    v1 = v1.astype(np.float64, copy=False)
    v2 = v2.astype(np.float64, copy=False)

    # Pre-compute per-dimension contributions (all O(n))
    a2 = v1 * v1  # ‖v1‖² per dim
    b2 = v2 * v2  # ‖v2‖² per dim
    ab = v1 * v2  # v1·v2  per dim

    N1 = a2.sum()  # current ‖v1‖²
    N2 = b2.sum()  # current ‖v2‖²
    D = ab.sum()  # current dot product

    keep = np.ones_like(v1, dtype=bool)  # mask of remaining axes

    dims_left: List[int] = []
    sim_values: List[float] = []
    removed_order: List[int] = []

    while True:
        r = int(keep.sum())
        dims_left.append(r)
        sim_values.append(float(D / np.sqrt(N1 * N2)))

        if r == 1:  # cannot remove further
            break

        idx = np.nonzero(keep)[0]  # indices still in play

        # Vectorised “what if we drop axis i?”
        new_N1 = N1 - a2[idx]
        new_N2 = N2 - b2[idx]
        valid = (new_N1 > 0) & (new_N2 > 0)
        if not valid.any():  # degenerate case
            break

        sims = np.full_like(new_N1, -np.inf, dtype=np.float64)
        sims[valid] = (D - ab[idx][valid]) / np.sqrt(new_N1[valid] * new_N2[valid])

        best_pos = int(sims.argmax())
        best_axis = int(idx[best_pos])

        # Update the running aggregates *once* (O(1))
        N1 -= a2[best_axis]
        N2 -= b2[best_axis]
        D -= ab[best_axis]
        keep[best_axis] = False
        removed_order.append(best_axis)

    plt.figure()
    plt.plot(dims_left, sim_values)
    plt.gca().invert_xaxis()
    plt.xlabel("Dimensions remaining")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine similarity (greedy improvements by axis removal)")
    plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)

    return dims_left, sim_values, removed_order


def get_layer(string: str):
    pattern = re.compile(r"^layer_(\d+)$")

    m = pattern.match(string)
    if m:
        return int(m.group(1))
    else:
        raise Exception(f"Could not extract layer from {string}")


def get_accuracies(seed: int, savedir: str):
    os.makedirs(savedir, exist_ok=True)

    benches = [
        "AIME2025",
        "AIME_2024",
        "AMC-23",
        "MATH-500",
        "Minerva-Math",
        "OlympiadBench",
    ]

    model_paths = get_qwen2_5_math_7b_model_paths(seed=seed)
    model_paths = [
        p.replace("/steering_vectors/", "/results/").replace("steering_vectors.npy", "")
        + "temp_1.0_top_p_1.0/"
        for p in model_paths
    ]

    model_groups = [
        ModelGroup(
            title=f"Qwen2.5-Math-7B. seed-{seed}",
            models=[
                Model(
                    label=f"layer_{layer_idx}",
                    pkl_path=pkl_path,
                    name=f"Qwen2.5-Math-7B, layer {layer_idx}",
                    ckpt=None,
                )
                for layer_idx, pkl_path in enumerate(model_paths)
            ],
        )
    ]

    accuracies_dict, _, _, _, _, _ = run_for_model_groups(
        model_groups=model_groups,
        benches=benches,
        average_across_benches={
            "acc": ["MATH-500", "Minerva-Math", "OlympiadBench"],
            "avg_acc": ["AIME2025", "AIME_2024", "AMC-23"],
        },
    )

    avg_accuracies = accuracies_dict["Average"]

    tmp = [
        (get_layer(key.label), value["mean"]) for key, value in avg_accuracies.items()
    ]
    tmp = sorted(tmp, key=lambda x: x[0])
    layer_wise_accuracies = [x[1] for x in tmp]

    return np.array(layer_wise_accuracies)


@pyrallis.wrap()
def run(config: Config):
    steering_vectors_seed_0 = np.load(
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-merged/seed-0/steering_vectors.npy"
    )
    steering_vectors_seed_1 = np.load(
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-merged/seed-1/steering_vectors.npy"
    )
    steering_vectors = np.load(
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-314/steering_vectors.npy"
    )

    cossims = rowwise_cosine(steering_vectors_seed_0, steering_vectors_seed_1)
    cossims2 = rowwise_cosine(steering_vectors_seed_0, steering_vectors)

    perfs_0 = get_accuracies(seed=0, savedir="tmp")
    perfs_1 = get_accuracies(seed=1, savedir="tmp")
    assert len(perfs_0) == len(perfs_1), (len(perfs_0), len(perfs_1))
    perf_diffs = np.abs(perfs_0 - perfs_1)

    plot_cosine_barplot(
        cosine_values=cossims,
        perf_diffs=perf_diffs,
        path=os.path.join(config.savedir, "cossims.png"),
    )
    plot_cosine_barplot(
        cosine_values=cossims2,
        perf_diffs=np.zeros_like(perf_diffs),
        path=os.path.join(config.savedir, "cossims_merged_vs_vanilla.png"),
    )
    for i in range(len(steering_vectors_seed_0)):
        greedy_axis_removal_improve(
            steering_vectors_seed_0[i],
            steering_vectors_seed_1[i],
            path=os.path.join(
                config.savedir, "aligned_subspaces", f"steering_vector_{i}.png"
            ),
        )


if __name__ == "__main__":
    run()
