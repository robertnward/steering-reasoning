import math
import os
import pickle as pkl
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

# AX_LEFT = 0.12
# AX_BOTTOM = 0.14  # same 'bottom' for both figures
# AX_WIDTH = 0.82
# AX_HEIGHT = 0.66  # same height for both figures
# LEGAP = 0.02  # gap between main plot and legend
# LEG_HEIGHT = 0.14

LEFT = 0.10  # distance from left figure edge to y-axis
BOTTOM = 0.16  # distance from bottom to x-axis
WIDTH = 0.86  # width of the main plotting area
HEIGHT = 0.70  # height of the main plotting area

GAP = 0.01  # gap between main plot and legend (2nd fig)
LEG_H = 0.12  # legend axes height (2nd fig)

# FIGSIZE = (7.2, 4.8)  # use the same width for both figures
FIGSIZE = (7.2, 3.6)  # use the same width for both figures
DPI = 150


def append_to(savedir: str):
    # ── 1. DATA ────────────────────────────────────────────────────────────────
    # Means
    base_greedy_perf = 24.8
    base_greedy_append_to_perf = 35.5
    last_steering_greedy_perf = 38.7
    base_sampling_perf = 14.3
    base_sampling_append_to_perf = 25.5
    last_steering_sampling_perf = 29.4

    # Standard deviations (⇩ CHANGE THESE to your real numbers)
    base_greedy_std = 0.0
    base_greedy_append_to_std = 0.0
    last_steering_greedy_std = 0.0
    base_sampling_std = 1.7
    base_sampling_append_to_std = 0.7
    last_steering_sampling_std = 0.4

    logger.info(
        f"Greedy. Qwen + To relative: {100 * (base_greedy_append_to_perf - base_greedy_perf) / (last_steering_greedy_perf - base_greedy_perf):.1f}"
    )
    logger.info(
        f"Sampling. Qwen + To relative: {100 * (base_sampling_append_to_perf - base_sampling_perf) / (last_steering_sampling_perf - base_sampling_perf):.1f}"
    )

    qwen2_5_math_7b_means = np.array(
        [
            [
                base_greedy_perf,
                base_greedy_append_to_perf,
                last_steering_greedy_perf,
            ],
            [
                base_sampling_perf,
                base_sampling_append_to_perf,
                last_steering_sampling_perf,
            ],
        ]
    )

    qwen2_5_math_7b_stds = np.array(
        [
            [
                base_greedy_std,
                base_greedy_append_to_std,
                last_steering_greedy_std,
            ],
            [
                base_sampling_std,
                base_sampling_append_to_std,
                last_steering_sampling_std,
            ],
        ]
    )

    # Means
    base_greedy_perf = 21.5
    base_greedy_append_to_perf = 22.4
    last_steering_greedy_perf = 21.6
    base_sampling_perf = 11.7
    base_sampling_append_to_perf = 13.4
    last_steering_sampling_perf = 14.7

    # Standard deviations (⇩ CHANGE THESE to your real numbers)
    base_greedy_std = 0.0
    base_greedy_append_to_std = 0.0
    last_steering_greedy_std = 0.0
    base_sampling_std = 0.5
    base_sampling_append_to_std = 0.5
    last_steering_sampling_std = 0.5

    logger.info(
        f"Greedy. Llama + Step relative: {100 * (base_greedy_append_to_perf - base_greedy_perf) / (last_steering_greedy_perf - base_greedy_perf):.1f}"
    )
    logger.info(
        f"Sampling. Llama + Step relative: {100 * (base_sampling_append_to_perf - base_sampling_perf) / (last_steering_sampling_perf - base_sampling_perf):.1f}"
    )

    llama3_1_8b_chat_means = np.array(
        [
            [
                base_greedy_perf,
                base_greedy_append_to_perf,
                last_steering_greedy_perf,
            ],
            [
                base_sampling_perf,
                base_sampling_append_to_perf,
                last_steering_sampling_perf,
            ],
        ]
    )

    llama3_1_8b_chat_stds = np.array(
        [
            [
                base_greedy_std,
                base_greedy_append_to_std,
                last_steering_greedy_std,
            ],
            [
                base_sampling_std,
                base_sampling_append_to_std,
                last_steering_sampling_std,
            ],
        ]
    )

    groups = ["Greedy", "Sampling"]
    categories = ["Base", 'Base + "To"', "Last Steering"]

    colors = ["#999999", "#56B4E9", "#F28E2B"]

    for name, token, (means, stds) in zip(
        ["qwen", "llama"],
        ["To", "Step"],
        [
            (qwen2_5_math_7b_means, qwen2_5_math_7b_stds),
            (llama3_1_8b_chat_means, llama3_1_8b_chat_stds),
        ],
    ):
        categories = ["Base", f'Base + "{token}"', "Last Steering"]
        # ── 2. LAYOUT / AXES ───────────────────────────────────────────────────────
        x = np.arange(len(groups))
        group_width = 0.8
        bar_width = group_width / len(categories)

        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        ax = fig.add_axes([LEFT, BOTTOM, WIDTH, HEIGHT])
        ax_leg = fig.add_axes([LEFT, BOTTOM + HEIGHT + GAP, WIDTH, LEG_H])
        ax_leg.axis("off")

        # ── 3. PLOT BARS WITH ERROR BARS ──────────────────────────────────────────
        bars_by_category = []
        for i, cat in enumerate(categories):
            offsets = -group_width / 2 + i * bar_width + bar_width / 2
            bars = ax.bar(
                x + offsets,
                means[:, i],
                width=bar_width,
                yerr=stds[:, i],
                capsize=0,
                ecolor="black",
                error_kw=dict(lw=1, capthick=1),
                label=cat,
                color=colors[i],
            )
            bars_by_category.append(bars)

        # ── 4. STYLE AXES ─────────────────────────────────────────────────────────
        ax.set_ylabel("Performance")
        ax.set_xticks(x, groups)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.02)

        # ── 5. ANNOTATE EACH BAR WITH “μ ± σ” ─────────────────────────────────────
        for bars, sds in zip(bars_by_category, stds.T):
            for b, sd in zip(bars, sds):
                μ = b.get_height()
                ax.annotate(
                    f"{μ:.1f}",
                    xy=(b.get_x() + b.get_width() / 2, μ + sd),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        # ── 6. LEGEND ─────────────────────────────────────────────────────────────
        handles, labels = ax.get_legend_handles_labels()
        ax_leg.legend(
            handles,
            labels,
            loc="center",
            ncol=len(categories),
            frameon=False,
            mode="expand",
            alignment="center",
            bbox_to_anchor=(0, 0, 1, 1),
            columnspacing=2.0,
            handletextpad=0.8,
            labelspacing=0.6,
        )

        # ── 7. SAVE ───────────────────────────────────────────────────────────────
        plt.savefig(os.path.join(savedir, f"{name}_append_to_bars.pdf"), format="pdf")
        plt.close(fig)


def clean_math_text(text):
    return text.replace("$", "\\$")


def _clean_pair_sequences(
    values: Sequence[float], labels: Sequence[str]
) -> Tuple[List[float], List[str]]:
    """Return validated *values* / *labels* lists of equal length."""
    if len(values) != len(labels):
        raise ValueError("`values` and `labels` must have the same length.")
    return list(values), list(labels)


def _group_values(
    values: Sequence[float], labels: Sequence[str]
) -> Dict[str, List[float]]:
    """Group numeric *values* by *labels*, dropping NaNs/None."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for v, lab in zip(values, labels):
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        grouped[str(lab)].append(fv)
    return {k: v for k, v in grouped.items() if v}


def _compute_stat(vals: Sequence[float], metric: str) -> float:
    """Compute *metric* for *vals*.

    Parameters
    ----------
    vals : sequence of float
    metric : {'mean', 'max', 'min'}
    """
    if metric == "mean":
        return float(np.mean(vals))
    if metric == "max":
        return float(np.max(vals))
    if metric == "min":
        return float(np.min(vals))
    raise ValueError(f"Unknown metric: {metric!r}. Expected 'mean', 'max', or 'min'.")


def _sorted_labels_by(
    grouped: Mapping[str, Sequence[float]],
    *,
    sorted_by: str = "mean",
    ascending: bool = False,
) -> List[Tuple[str, float]]:
    """Return (label, stat) pairs ranked by *stat*."""
    stats: List[Tuple[str, float]] = [
        (lab, _compute_stat(vals, sorted_by)) for lab, vals in grouped.items()
    ]
    stats.sort(key=lambda x: x[1], reverse=not ascending)
    return stats


def get_stats(
    prob_diffs: np.ndarray,
    tokens: List[str],
    top_k: int,
    sorted_by: str,
    is_filtered: bool,
    ascending: bool,
):
    vals, labs = _clean_pair_sequences(prob_diffs, tokens)
    grouped = _group_values(vals, labs)
    if is_filtered:
        grouped = {k: v for k, v in grouped.items() if len(v) > 1}
    if not grouped:
        raise ValueError("No valid numeric data to plot.")

    # --- Top panel rankings (descending by *sorted_by*)
    stats_all = _sorted_labels_by(grouped, sorted_by=sorted_by, ascending=ascending)
    stats = stats_all[:top_k] if top_k else stats_all

    return stats, grouped


def token_prob_diffs(model: str, savedir: str):
    loaddir = os.path.expanduser(os.path.join("~", "Downloads"))

    all_top_diffs = np.load(os.path.join(loaddir, f"all_top_diffs_{model}.npy"))
    with open(os.path.join(loaddir, f"all_top_tokens_dst_{model}.npy"), "rb") as f:
        all_top_tokens_dst = pkl.load(f)
        all_top_tokens_dst = [
            token.replace("\n", "\\n") for token in all_top_tokens_dst
        ]
    all_positions = np.load(os.path.join(loaddir, f"all_positions_{model}.npy"))

    all_token_to_prob_diffs = np.load(
        os.path.join(loaddir, f"all_token_to_prob_diffs_{model}.npy")
    ).squeeze()

    stats, grouped = get_stats(
        prob_diffs=all_top_diffs,
        tokens=all_top_tokens_dst,
        top_k=5,
        sorted_by="max",
        is_filtered=True,
        ascending=False,
    )

    labels = [lab for lab, _ in stats]
    data = [grouped[lab] for lab in labels]

    # Extra dataset: all_token_to_prob_diffs at position 0
    mask = all_positions == 0
    extra_vals = all_token_to_prob_diffs[mask]

    # Combine so the extra boxplot appears as the last one (same axis, same style)
    labels_ext = labels + ["All tokens @ pos 0"]
    data_ext = data + [extra_vals]

    # fig = plt.figure()  # no constrained_layout
    # ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_WIDTH, AX_HEIGHT])
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)  # no constrained_layout
    ax = fig.add_axes([LEFT, BOTTOM, WIDTH, HEIGHT])

    # Single axes with all boxplots
    ax.boxplot(
        data_ext,
        widths=0.5,
        patch_artist=False,
        vert=True,
        showfliers=True,
        boxprops=dict(linewidth=2.5),
        whiskerprops=dict(linewidth=2.5),
        capprops=dict(linewidth=2.5),
        medianprops=dict(linewidth=3),
        flierprops=dict(marker="o", markersize=4, markeredgewidth=1.5),
    )

    if model == "llama":
        token = "Step"
    elif model == "qwen":
        token = "To"
    else:
        raise ValueError(f"Unknown model: {model}")

    ax.set_xticks(range(1, len(labels_ext) + 1))
    ax.set_xticklabels(
        [f'"{clean_math_text(lab)}"' for lab in labels] + [f'"{token}"\nat Pos. 0'],
        ha="center",
    )
    ax.set_ylabel("Probability Difference")

    # Vertical separator between box 4 and 5 (data coords)
    ax.axvline(x=5.5, linewidth=1.5, color="black")

    fig.savefig(
        os.path.join(savedir, f"{model}_token_prob_diffs.pdf"), format="pdf", dpi=150
    )
    plt.close(fig)


def steering_vectors_alignment(savedir: str):
    loaddir = os.path.expanduser(os.path.join("~", "Downloads"))
    steering_vectors = np.load(os.path.join(loaddir, "qwen_steering_vectors.npy"))
    steering_vectors_merged = np.load(
        os.path.join(loaddir, "qwen_steering_vectors_merged.npy")
    )

    sim_matrix = (
        steering_vectors / np.linalg.norm(steering_vectors, axis=1, keepdims=True)
    ) @ (
        steering_vectors_merged
        / np.linalg.norm(steering_vectors_merged, axis=1, keepdims=True)
    ).T

    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        sim_matrix,
        vmin=-1,
        # vmax=1,
        aspect="equal",
        cmap=sns.color_palette("vlag", as_cmap=True).copy(),
    )
    plt.colorbar(im, label="Cosine similarity")
    for (j, i), label in np.ndenumerate(sim_matrix):
        plt.text(i, j, round(label, 1), ha="center", va="center")
    plt.title("Pairwise cosine similarity")
    plt.xlabel("Steering Vector")
    plt.ylabel("Steering Vector Merged")
    plt.tight_layout()
    plt.savefig(
        os.path.join(savedir, "steering_vectors_alignment.pdf"), format="pdf", dpi=500
    )
    plt.close()


def plot_append_to_variants(savedir: str, filename: str = "bad_layers_qwen"):
    """
    Create a wide, short bar plot (orange bars) with error bars and μ ± σ annotations.

    Data (means with stds):
      - add_place_24_mlp: 25.1 (std 1.8)
      - add_place_25_input_layernorm: 35.2 (std 0.2)
      - add_place_25_self_attn: 36.2 (std 0.4)
      - add_place_25_post_attention_layernorm: 37.2 (std 0.6)
      - add_place_25_mlp: 36.3 (std 0.4)

    Args:
        savedir: Directory to save the figure files.
        filename: Base filename (without extension).
        show: If True, display the plot window before saving/closing.

    Returns:
        dict with paths to the saved PNG and PDF.
    """

    # ── 1) DATA ────────────────────────────────────────────────────────────────
    categories = [
        # "Layer 23 MLP",
        # "Layer 24 Input LN",
        # "Layer 24 Attn",
        # "Layer 24 Post-Attn LN",
        "Layer 24 MLP",
        "Layer 25 Input LN",
        "Layer 25 Attn",
        "Layer 25 Post-Attn LN",
        "Layer 25 MLP",
    ]
    # means = np.array([22.4, 28.9, 23.2, 0.0, 25.1, 35.2, 36.2, 37.2, 36.3], dtype=float)
    # stds = np.array([2.0, 1.3, 1.6, 0.0, 1.8, 0.2, 0.4, 0.6, 0.4], dtype=float)
    means = np.array([21.1, 35.2, 36.2, 37.2, 38.2], dtype=float)
    stds = np.array([2.2, 0.2, 0.4, 0.6, 0.5], dtype=float)

    # ── 2) PLOT ────────────────────────────────────────────────────────────────
    ORANGE = "#F28E2B"
    fig, ax = plt.subplots(dpi=500)  # wide & short

    x = np.arange(len(categories))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=0.7,
        capsize=0,
        ecolor="black",
        error_kw=dict(lw=1, capthick=1),
        color=ORANGE,
    )

    # ── 3) STYLE ───────────────────────────────────────────────────────────────
    ax.set_ylabel("Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)

    # ── 4) ANNOTATE μ ± σ ─────────────────────────────────────────────────────
    for b, sd in zip(bars, stds):
        mu = b.get_height()
        ax.annotate(
            f"{mu:.1f} ± {sd:.1f}",
            xy=(b.get_x() + b.get_width() / 2, mu + sd),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    # ── 5) SAVE ───────────────────────────────────────────────────────────────
    os.makedirs(savedir, exist_ok=True)
    pdf_path = os.path.join(savedir, f"{filename}.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    plt.close(fig)


def run():
    savedir = "figures"
    os.makedirs(savedir, exist_ok=True)
    with mpl.rc_context(
        {
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.fontsize": 14,
        }
    ):
        append_to(savedir=savedir)

    with mpl.rc_context(
        {
            "figure.figsize": (10, 6),
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    ):
        token_prob_diffs(model="qwen", savedir=savedir)
        token_prob_diffs(model="llama", savedir=savedir)

    with mpl.rc_context(
        {
            "figure.figsize": (10, 6),
            # "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    ):
        steering_vectors_alignment(savedir=savedir)

    with mpl.rc_context(
        {
            "figure.figsize": (10, 4),
            # "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    ):
        plot_append_to_variants(savedir=savedir)

    layer_30 = 19.9
    layer_31 = 14.5
    skip_attn = 18.2
    skip_layer = 13.9

    logger.info(
        f"llama-patch-head. layer-30: {layer_30:.1f}; skip-attn: {skip_attn:.1f}; layer-31: {layer_31:.1f}; skip-layer: {skip_layer:.1f}"
    )
    logger.info(
        f"relative llama-patch-head. skip-attn gain: {100 * (skip_attn - layer_31) / (layer_30 - layer_31):.1f}"
    )


if __name__ == "__main__":
    run()
