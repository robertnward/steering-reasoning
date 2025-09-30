import csv
import math
import os
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrallis
import torch
import vllm
from accelerate.utils import set_seed
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from vllm import RequestOutput

from steering_reasoning.train.rl.policy_model import tokenize_example
from steering_reasoning.train.rl.trainer import is_answer_present
from steering_reasoning.utils.utils import set_logger


@dataclass
class Config:
    seed: int

    dataset_path: str
    model_path: str

    num_samples: int
    batch_size: int
    max_seq_length: int
    num_generations: int
    repetition_penalty: float
    top_p: float
    top_k: int
    generation_temperature: float
    stop_token: str

    template_type: str
    append_to: bool

    prob_top_k: int
    prob_threshold: float

    boxplot_width: float

    log_first_k_samples: int

    savedir: str

    def __post_init__(self):
        if self.append_to:
            self.savedir = os.path.join(self.savedir, f"append_to_{self.append_to}")
        os.makedirs(self.savedir, exist_ok=True)

        assert self.template_type in ["qwen_math", "r1"], self.template_type


def get_steering_vectors(model):
    steering_vectors = []
    for layer in model.model.layers:
        steering_vector = layer.mlp.down_proj.bias.data
        steering_vectors.append(steering_vector)

    steering_vectors = torch.vstack(steering_vectors)

    return steering_vectors


def update_steering_vectors(model, steering_vectors: torch.Tensor):
    for layer, steering_vector in zip(model.model.layers, steering_vectors):
        layer.mlp.down_proj.bias.data = steering_vector


def get_probs(logits, generation_temperature: float):
    return torch.nn.functional.softmax(logits / generation_temperature, dim=-1)


@torch.no_grad()
def get_prob_diffs(
    model,
    token_ids: torch.Tensor,
    steering_vectors: torch.Tensor,
    zero_steering_vectors: torch.Tensor,
    generation_temperature: float,
):
    input_ids = token_ids.unsqueeze(0)
    attention_mask = torch.ones_like(token_ids).unsqueeze(0)
    position_ids = torch.arange(len(token_ids)).unsqueeze(0).to(token_ids.device)

    # Base probs
    update_steering_vectors(model=model, steering_vectors=zero_steering_vectors)
    vanilla_logits = model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    ).logits
    vanilla_probs = get_probs(
        logits=vanilla_logits, generation_temperature=generation_temperature
    ).squeeze(0)
    vanilla_entropy = torch.distributions.Categorical(probs=vanilla_probs).entropy()

    # Steered probs
    update_steering_vectors(model=model, steering_vectors=steering_vectors)
    steered_logits = model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    ).logits
    steered_probs = get_probs(
        logits=steered_logits,
        generation_temperature=generation_temperature,
    ).squeeze(0)
    steered_entropy = torch.distributions.Categorical(probs=steered_probs).entropy()

    # Prob diffs
    prob_diffs = steered_probs - vanilla_probs

    return (
        prob_diffs,
        vanilla_probs,
        steered_probs,
        vanilla_entropy,
        steered_entropy,
    )


def get_top_probs(
    vanilla_probs: torch.Tensor,
    steered_probs: torch.Tensor,
    prob_top_k: int,
    tokenizer: PreTrainedTokenizer,
):
    top_base_probs, top_base_tokens = torch.topk(vanilla_probs, k=prob_top_k, dim=-1)
    top_base_tokens = [
        [tokenizer.decode(token, skip_special_tokens=False) for token in tokens]
        for tokens in top_base_tokens
    ]
    top_steered_probs, top_steered_tokens = torch.topk(
        steered_probs, k=prob_top_k, dim=-1
    )
    top_steered_tokens = [
        [tokenizer.decode(token, skip_special_tokens=False) for token in tokens]
        for tokens in top_steered_tokens
    ]

    return top_base_probs, top_base_tokens, top_steered_probs, top_steered_tokens


@torch.no_grad()
def get_top_diffs(prob_diffs: torch.Tensor, tokenizer: PreTrainedTokenizer):
    top_prob_diffs, top_tokens = torch.max(prob_diffs, dim=-1)
    top_prob_diffs, top_tokens = top_prob_diffs.squeeze(), top_tokens.squeeze()
    top_tokens = [
        tokenizer.decode(token, skip_special_tokens=False) for token in top_tokens
    ]

    return top_prob_diffs, top_tokens


@torch.no_grad()
def get_bottom_diffs(prob_diffs: torch.Tensor, tokenizer: PreTrainedTokenizer):
    bottom_prob_diffs, bottom_tokens = torch.min(prob_diffs, dim=-1)
    bottom_prob_diffs, bottom_tokens = (
        bottom_prob_diffs.squeeze(),
        bottom_tokens.squeeze(),
    )
    bottom_tokens = [
        tokenizer.decode(token, skip_special_tokens=False) for token in bottom_tokens
    ]

    return bottom_prob_diffs, bottom_tokens


def top_k_indicator(values, k=10):
    """
    Flag each element with 1 if it ranks in the top-k largest values, else 0.

    Parameters
    ----------
    values : Sequence[float | int]
        The data to rank.
    k : int, default 10
        How many top elements to flag.

    Returns
    -------
    list[int]
        A list of 0s and 1s, same length as `values`.
    """
    n = len(values)
    if k <= 0 or n == 0:
        return [0] * n

    # Indices sorted by value, descending
    sorted_idx = sorted(range(n), key=lambda i: values[i], reverse=True)

    # Pick the first k indices (ties beyond k are *not* included;
    # you can change this by computing a threshold instead)
    top_idx = set(sorted_idx[: min(k, n)])

    # Build the indicator list
    return [1 if i in top_idx else 0 for i in range(n)]


def plot_prob_diffs_hist(prob_diffs: np.ndarray, savedir: str):
    os.makedirs(savedir, exist_ok=True)

    plt.hist(prob_diffs, bins=50)
    plt.xlabel("Probability Difference")
    plt.ylabel("Frequency")
    plt.title("Histogram of Probability Differences")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "prob_diffs.png"))
    plt.close()


def save_sorted_prob_diffs(
    prob_diffs: np.ndarray,
    tokens: List[str],
    src_tokens: List[str],
    would_be_tokens: List[str],
    reverse: bool,
    is_correct: bool,
    savedir: str,
):
    os.makedirs(savedir, exist_ok=True)

    assert len(prob_diffs) == len(tokens) == len(src_tokens) == len(would_be_tokens), (
        len(prob_diffs),
        len(tokens),
        len(src_tokens),
        len(would_be_tokens),
    )

    tokens = [token.replace("\n", "\\n") for token in tokens]
    src_tokens = [token.replace("\n", "\\n") for token in src_tokens]
    would_be_tokens = [token.replace("\n", "\\n") for token in would_be_tokens]

    zipped = sorted(
        zip(prob_diffs, tokens, src_tokens, would_be_tokens),
        key=lambda x: x[0],
        reverse=reverse,
    )

    with open(os.path.join(savedir, "sorted_prob_diffs_tokens.txt"), "w") as f:
        f.write(f"Is Correct: {is_correct}\n")
        f.write("at SrcToken instead of WouldBeToken: TopToken with ProbDiff\n")
        f.write(f"Len: {len(src_tokens)}\n\n")
        for prob_diff, token, src_token, would_be_token in zipped:
            f.write(
                f'at "{src_token}" instead of "{would_be_token}": "{token}" with prob diff {prob_diff:.4f}\n'
            )


def save_top_prob_diffs_by_position(
    prob_diffs: np.ndarray,
    tokens: List[str],
    src_tokens: List[str],
    would_be_tokens: List[str],
    is_correct: bool,
    savedir: str,
):
    os.makedirs(savedir, exist_ok=True)

    assert len(prob_diffs) == len(tokens) == len(src_tokens) == len(would_be_tokens), (
        len(prob_diffs),
        len(tokens),
        len(src_tokens),
        len(would_be_tokens),
    )

    tokens = [token.replace("\n", "\\n") for token in tokens]
    src_tokens = [token.replace("\n", "\\n") for token in src_tokens]
    would_be_tokens = [token.replace("\n", "\\n") for token in would_be_tokens]

    with open(os.path.join(savedir, "top_prob_diffs_by_position.txt"), "w") as f:
        f.write(f"Is Correct: {is_correct}\n")
        f.write("at SrcToken instead of WouldBeToken: TopToken with ProbDiff\n")
        f.write(f"Len: {len(src_tokens)}\n\n")

        for prob_diff, token, src_token, would_be_token in zip(
            prob_diffs, tokens, src_tokens, would_be_tokens
        ):
            f.write(
                f'at "{src_token}" instead of "{would_be_token}": "{token}" with prob diff {prob_diff:.4f}\n'
            )


def save_plain_text(src_tokens: List[str], is_correct: bool, savedir: str):
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, "plain_text.txt"), "w") as f:
        f.write(f"Is Correct: {is_correct}\n")
        f.write("".join([token.replace("\\n", "\n") for token in src_tokens]))


def save_top_probs_sets(
    prob_diffs: np.ndarray,
    base_probs: np.ndarray,
    base_tokens: List[str],
    steered_probs: np.ndarray,
    steered_tokens: List[str],
    src_tokens: List[str],
    prob_threshold: float,
    is_correct: bool,
    savedir: str,
):
    top_indicator = top_k_indicator(prob_diffs, k=10)

    with open(os.path.join(savedir, "top_probs_sets.txt"), "w") as f:
        f.write(f"Is Correct: {is_correct}\n")
        f.write("at SrcToken: [BaseTokens] -> [SteeredTokens]\n")
        f.write(f"Len: {len(base_tokens)}\n\n")
        for (
            src_token,
            base_probs,
            base_tokens,
            steered_probs,
            steered_tokens,
            indicator,
            top_diff,
        ) in zip(
            src_tokens,
            base_probs,
            base_tokens,
            steered_probs,
            steered_tokens,
            top_indicator,
            prob_diffs,
        ):
            base_tokens = [
                token.replace("\n", "\\n")
                for token, prob in zip(base_tokens, base_probs)
                if prob > prob_threshold
            ]
            base_tokens = '["' + '", "'.join(base_tokens) + '"]'
            base_probs = base_probs[base_probs >= prob_threshold].tolist()
            base_probs = "[" + ", ".join([f"{x:.2f}" for x in base_probs]) + "]"

            steered_tokens = [
                token.replace("\n", "\\n")
                for token, prob in zip(steered_tokens, steered_probs)
                if prob > prob_threshold
            ]
            steered_probs = steered_probs[steered_probs >= prob_threshold].tolist()
            steered_tokens = '["' + '", "'.join(steered_tokens) + '"]'
            steered_probs = "[" + ", ".join([f"{x:.2f}" for x in steered_probs]) + "]"

            line = f'at "{src_token}": {base_tokens} -> {steered_tokens} ({base_probs} -> {steered_probs})\n'

            if indicator == 1:
                line = f"[Top-10 Diff {top_diff}] " + line

            f.write(line)


def plot_prob_diffs_vs_entropy(
    prob_diffs: np.ndarray, entropy: np.ndarray, savedir: str
):
    os.makedirs(savedir, exist_ok=True)
    plt.scatter(entropy, prob_diffs)
    plt.xlabel("Entropy")
    plt.ylabel("Prob Diffs")
    plt.title("Prob Diff vs. Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "prob_diff_vs_entropy.png"))
    plt.close()


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


# ---------------------------------------------------------------------------
# Dual-panel dataset-wise plotting
# ---------------------------------------------------------------------------


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


def plot_ds_wise(
    all_top_diffs: List[float],
    all_bottom_diffs: List[float],
    all_top_tokens_dst: List[str],
    all_bottom_tokens_dst: List[str],
    all_top_tokens_src: List[str],
    all_top_tokens_would_be: List[str],
    all_positions: List[int],
    top_k: int,
    boxplot_width: float,
    sorted_by: str,
    is_filtered: bool,
    savedir: str,
):
    """Create dual-panel boxplots (top-*k* by *sorted_by*, bottom-*k* by *min*).

    The *top* panel uses the provided ``sorted_by`` statistic (e.g. *mean* or
    *max*), ranked **descending**.  The *bottom* panel is always ranked by the
    group **minimum** (*min*), ranked **ascending** to highlight worst-case
    performers.
    """
    os.makedirs(savedir, exist_ok=True)

    all_top_tokens_dst = [token.replace("\n", "\\n") for token in all_top_tokens_dst]
    all_bottom_tokens_dst = [
        token.replace("\n", "\\n") for token in all_bottom_tokens_dst
    ]
    all_top_tokens_src = [token.replace("\n", "\\n") for token in all_top_tokens_src]
    all_top_tokens_would_be = [
        token.replace("\n", "\\n") for token in all_top_tokens_would_be
    ]
    all_positions_is_first = [
        "First" if int(x) == 0 else "Other" for x in all_positions
    ]
    all_positions = [str(x) for x in all_positions]

    top_dst_plus_would_be = [
        would_be + '" -> "' + dst
        for dst, would_be in zip(all_top_tokens_dst, all_top_tokens_would_be)
    ]
    bottom_dst_plus_would_be = [
        would_be + '" -> "' + dst
        for dst, would_be in zip(all_bottom_tokens_dst, all_top_tokens_would_be)
    ]
    top_dst_plus_src = [
        src + '": "' + dst for dst, src in zip(all_top_tokens_dst, all_top_tokens_src)
    ]
    bottom_dst_plus_src = [
        src + '": "' + dst
        for dst, src in zip(all_bottom_tokens_dst, all_top_tokens_src)
    ]
    top_dst_plus_src_plus_would_be = [
        src + '": "' + would_be + '" -> "' + dst
        for dst, src, would_be in zip(
            all_top_tokens_dst, all_top_tokens_src, all_top_tokens_would_be
        )
    ]
    bottom_dst_plus_src_plus_would_be = [
        src + '": "' + would_be + '" -> "' + dst
        for dst, src, would_be in zip(
            all_bottom_tokens_dst, all_top_tokens_src, all_top_tokens_would_be
        )
    ]

    top_dst_plus_pos = [
        pos + '": "' + dst for dst, pos in zip(all_top_tokens_dst, all_positions)
    ]
    bottom_dst_plus_pos = [
        pos + '": "' + dst for dst, pos in zip(all_bottom_tokens_dst, all_positions)
    ]
    top_dst_plus_pos_plus_would_be = [
        pos + '": "' + would_be + '" -> "' + dst
        for dst, pos, would_be in zip(
            all_top_tokens_dst, all_positions, all_top_tokens_would_be
        )
    ]
    bottom_dst_plus_pos_plus_would_be = [
        pos + '": "' + would_be + '" -> "' + dst
        for dst, pos, would_be in zip(
            all_bottom_tokens_dst, all_positions, all_top_tokens_would_be
        )
    ]
    top_dst_plus_pos_is_first = [
        pos + '": "' + dst
        for dst, pos in zip(all_top_tokens_dst, all_positions_is_first)
    ]
    bottom_dst_plus_pos_is_first = [
        pos + '": "' + dst
        for dst, pos in zip(all_bottom_tokens_dst, all_positions_is_first)
    ]
    top_dst_plus_pos_is_first_plus_would_be = [
        pos + '": "' + would_be + '" -> "' + dst
        for dst, pos, would_be in zip(
            all_top_tokens_dst, all_positions_is_first, all_top_tokens_would_be
        )
    ]
    bottom_dst_plus_pos_is_first_plus_would_be = [
        pos + '": "' + would_be + '" -> "' + dst
        for dst, pos, would_be in zip(
            all_bottom_tokens_dst, all_positions_is_first, all_top_tokens_would_be
        )
    ]

    for name, (top_tokens, bottom_tokens) in zip(
        [
            "dst",
            "src",
            "would_be",
            "would_be->dst",
            "src:dst",
            "src:would_be->dst",
            "pos",
            "pos:dst",
            "pos:would_be->dst",
            "pos_is_first",
            "pos_is_first:dst",
            "pos_is_first:would_be->dst",
        ],
        [
            (all_top_tokens_dst, all_bottom_tokens_dst),
            (all_top_tokens_src, all_top_tokens_src),
            (all_top_tokens_would_be, all_top_tokens_would_be),
            (top_dst_plus_would_be, bottom_dst_plus_would_be),
            (top_dst_plus_src, bottom_dst_plus_src),
            (top_dst_plus_src_plus_would_be, bottom_dst_plus_src_plus_would_be),
            (all_positions, all_positions),
            (top_dst_plus_pos, bottom_dst_plus_pos),
            (top_dst_plus_pos_plus_would_be, bottom_dst_plus_pos_plus_would_be),
            (all_positions_is_first, all_positions_is_first),
            (top_dst_plus_pos_is_first, bottom_dst_plus_pos_is_first),
            (
                top_dst_plus_pos_is_first_plus_would_be,
                bottom_dst_plus_pos_is_first_plus_would_be,
            ),
        ],
    ):
        top_stats, top_grouped = get_stats(
            prob_diffs=all_top_diffs,
            tokens=top_tokens,
            top_k=top_k,
            sorted_by=sorted_by,
            is_filtered=is_filtered,
            ascending=False,
        )
        if len(top_stats) == 0:
            continue

        paper_savedir = os.path.join(savedir, "boxplots_paper")
        os.makedirs(paper_savedir, exist_ok=True)
        with mpl.rc_context(
            {
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
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

            _plot_boxpanel(
                ax=ax,
                stats=top_stats,
                grouped=top_grouped,
                boxplot_width=boxplot_width,
                ylabel="Probability Difference",
                title=None,
                xlabel=f"{name} token" if name != "dst" else None,
            )
            fig.savefig(
                os.path.join(paper_savedir, f"{name}_token.pdf"), format="pdf", dpi=150
            )
            plt.close(fig)

        bottom_stats, bottom_grouped = get_stats(
            prob_diffs=all_bottom_diffs,
            tokens=bottom_tokens,
            top_k=top_k,
            sorted_by=sorted_by if sorted_by == "mean" else "min",
            is_filtered=is_filtered,
            ascending=True,
        )

        if len(bottom_stats) == 0:
            continue

        # ------------------------------------------------------------------
        # Plotting - dual-row figure
        # ------------------------------------------------------------------
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(25, 10), constrained_layout=True
        )

        # -- Top panel
        _plot_boxpanel(
            ax=ax_top,
            stats=top_stats,
            grouped=top_grouped,
            boxplot_width=boxplot_width,
            ylabel="Probability Difference",
            title=f"Top {top_k} {name} tokens by {sorted_by}",
            xlabel=f"{name} token",
        )

        # -- Bottom panel
        _plot_boxpanel(
            ax=ax_bottom,
            stats=bottom_stats,
            grouped=bottom_grouped,
            boxplot_width=boxplot_width,
            ylabel="Probability Difference",
            title=f"Bottom {top_k} {name} tokens by min",
            xlabel=f"{name} token",
        )

        # ------------------------------------------------------------------
        # Save & close
        # ------------------------------------------------------------------
        fig.savefig(os.path.join(savedir, f"{name}_token.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def clean_math_text(text):
    return text.replace("$", "\\$")


def _plot_boxpanel(
    *,
    ax: plt.Axes,
    stats: List[Tuple[str, float]],
    grouped: Mapping[str, List[float]],
    boxplot_width: float,
    ylabel: str,
    title: Optional[str],
    xlabel: str,
):
    """Draw a row of narrow boxplots on *ax* in the order provided by *stats*."""
    labels = [lab for lab, _ in stats]
    data = [grouped[lab] for lab in labels]
    if title is None:
        kwargs = dict(
            boxprops=dict(linewidth=2.5),
            whiskerprops=dict(linewidth=2.5),
            capprops=dict(linewidth=2.5),
            medianprops=dict(linewidth=3),
            flierprops=dict(marker="o", markersize=4, markeredgewidth=1.5),
        )
    else:
        kwargs = {}

    ax.boxplot(
        data,
        widths=boxplot_width,
        patch_artist=False,
        vert=True,
        showfliers=True,
        **kwargs,
    )
    ax.set_xticks(range(1, len(labels) + 1))
    if title is not None:
        xticklabels = [
            f'"{clean_math_text(lab)}" (#{len(data_)})'
            for lab, data_ in zip(labels, data)
        ]
    else:
        xticklabels = [f'"{clean_math_text(lab)}"' for lab in labels]
    ax.set_xticklabels(
        xticklabels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def frac_reinforce_top(
    diffs: List[float],
    tokens_dst: List[str],
    tokens_would_be: List[str],
    tokens_src: List[str],
    savedir: str,
):
    os.makedirs(savedir, exist_ok=True)

    sames = [dst == would_be for dst, would_be in zip(tokens_dst, tokens_would_be)]
    frac_same = np.mean(sames)

    with open(os.path.join(savedir, "frac_same.txt"), "w") as f:
        f.write(f"Frac same: {frac_same:.3f}")

    save_sorted_prob_diffs(
        prob_diffs=[diff for diff, is_same in zip(diffs, sames) if is_same],
        tokens=[diff for diff, is_same in zip(tokens_dst, sames) if is_same],
        src_tokens=[diff for diff, is_same in zip(tokens_src, sames) if is_same],
        would_be_tokens=[
            diff for diff, is_same in zip(tokens_would_be, sames) if is_same
        ],
        reverse=True,
        is_correct=None,
        savedir=os.path.join(savedir, "same"),
    )
    save_sorted_prob_diffs(
        prob_diffs=[diff for diff, is_same in zip(diffs, sames) if not is_same],
        tokens=[diff for diff, is_same in zip(tokens_dst, sames) if not is_same],
        src_tokens=[diff for diff, is_same in zip(tokens_src, sames) if not is_same],
        would_be_tokens=[
            diff for diff, is_same in zip(tokens_would_be, sames) if not is_same
        ],
        reverse=True,
        is_correct=None,
        savedir=os.path.join(savedir, "non_same"),
    )


def init_probs_vs_prob_diff(
    prob_diffs: np.ndarray,
    vanilla_probs: np.ndarray,
    steered_probs: np.ndarray,
    savedir: str,
):
    os.makedirs(savedir, exist_ok=True)

    max_prob_diffs = prob_diffs.max(-1)
    indices = prob_diffs.argmax(-1)
    max_vanilla_probs = vanilla_probs[np.arange(vanilla_probs.shape[0]), indices]
    max_steered_probs = steered_probs[np.arange(steered_probs.shape[0]), indices]

    plt.scatter(max_vanilla_probs, max_prob_diffs, label="Vanilla")
    plt.xlabel("Vanilla Probs")
    plt.ylabel("Prob Diffs")
    plt.title("Vanilla Probs vs. Prob Diffs")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "vanilla_probs_vs_prob_diffs.png"))
    plt.close()

    plt.scatter(max_steered_probs, max_prob_diffs, label="Vanilla")
    plt.xlabel("Steered Probs")
    plt.ylabel("Prob Diffs")
    plt.title("Steered Probs vs. Prob Diffs")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "steered_probs_vs_prob_diffs.png"))
    plt.close()


def token_to_vis_position(
    prob_diffs: List[int],
    positions: List[int],
    savepath: str,
) -> None:
    assert len(positions) == len(prob_diffs), "positions length must match prob_diffs"
    prob_diffs = np.asarray(prob_diffs)
    positions = np.asarray(positions)

    first_prob_diffs = prob_diffs[positions == 0]
    other_prob_diffs = prob_diffs[positions != 0]

    # Drop NaNs to avoid matplotlib stats warnings
    first = first_prob_diffs[~np.isnan(first_prob_diffs)]
    other = other_prob_diffs[~np.isnan(other_prob_diffs)]

    if len(first) == 0 or len(other) == 0:
        raise ValueError(
            f"Both groups need data (got n0={len(first)}, n≠0={len(other)})."
        )

    with mpl.rc_context(
        {
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
        fig, ax = plt.subplots(figsize=(5, 6), dpi=150, constrained_layout=True)
        ax.boxplot(
            [first, other],
            labels=["pos = 0", r"pos $\neq$ 0"],
            boxprops=dict(linewidth=2.5),
            whiskerprops=dict(linewidth=2.5),
            capprops=dict(linewidth=2.5),
            medianprops=dict(linewidth=3),
            flierprops=dict(marker="o", markersize=4, markeredgewidth=1.5),
        )
        ax.set_ylabel("Probability Difference")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        plt.close(fig)


def get_for_nikita(prob_diffs, vanilla_probs, steered_probs, token_ids):
    next_token_diffs = prob_diffs[:-1][torch.arange(len(token_ids[1:])), token_ids[1:]]
    max_vanilla = vanilla_probs.argmax(dim=-1)
    max_steered = steered_probs.argmax(dim=-1)

    return next_token_diffs, max_vanilla, max_steered


def tsv_for_nikita(tokens, p_s, p_b, token_diffs, savedir):
    data = pd.DataFrame.from_records(
        {
            "tokens": [x.replace("\n", r"\n") for x in tokens],
            "argmax_steered": [x.replace("\n", r"\n") for x in p_s],
            "argmax_base": [x.replace("\n", r"\n") for x in p_b],
            "p_diff": [np.float32(0)] + list(token_diffs),
        }
    )
    columns = ["tokens", "argmax_steered", "argmax_base", "p_diff"]
    data = data[columns]
    data[columns].to_csv(
        os.path.join(savedir, "tokens.tsv"),
        index=False,
        sep="\t",
        quoting=csv.QUOTE_ALL,
    )


@pyrallis.wrap()
def run(config: Config):
    torch.set_grad_enabled(False)

    set_seed(seed=config.seed, device_specific=False, deterministic=False)
    set_logger(verbose=False)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = model.cuda().eval()

    steering_vectors = get_steering_vectors(model=model)
    zero_steering_vectors = torch.zeros_like(steering_vectors)

    vllm_actor = vllm.LLM(
        model=config.model_path,
        trust_remote_code=True,
        seed=config.seed,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=config.max_seq_length,
        max_seq_len_to_capture=config.max_seq_length * 2,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    stop_token_id = tokenizer.encode(config.stop_token, add_special_tokens=False)
    assert len(stop_token_id) == 1, (
        f"Stop token must be a single token, instead {stop_token_id}"
    )
    stop_token_id = stop_token_id[0]

    sampling_params = vllm.SamplingParams(
        n=config.num_generations,
        repetition_penalty=config.repetition_penalty,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.generation_temperature,
        stop_token_ids=[stop_token_id],
        seed=config.seed,
        logprobs=0,
    )

    ds = load_dataset(config.dataset_path)["train"]
    ds = ds.shuffle(seed=config.seed)
    ds = ds.filter(lambda x: len(x["answer"]) > 0)
    ds = ds.select(list(range(config.num_samples)))
    ds = ds.map(
        tokenize_example(
            tokenizer=tokenizer,
            stop_token=config.stop_token,
            max_seq_length=config.max_seq_length,
            template_type=config.template_type,
            append_to=config.append_to,
        ),
        batched=False,
        num_proc=None,
        remove_columns=["problem", "answer", "solution"],
    )

    if "llama" in config.model_path:
        token_to_idx = tokenizer.encode("Step", add_special_tokens=False)
    elif "Qwen" in config.model_path:
        token_to_idx = tokenizer.encode("To", add_special_tokens=False)
    else:
        raise ValueError(f"Unknown model: {config.model_path}")

    all_top_diffs = []
    all_top_tokens_dst = []
    all_bottom_diffs = []
    all_bottom_tokens_dst = []
    all_tokens_src = []
    all_tokens_would_be = []
    all_tokens_max_steered = []
    all_vanilla_entropies = []
    all_steered_entropies = []
    all_positions = []
    all_is_corrects = []
    all_is_eos = []
    all_token_to_prob_diffs = []
    for idx in trange(0, len(ds), config.batch_size, desc="Sample"):
        prompt_token_ids = ds["prompt_input_ids"][idx : idx + config.batch_size]
        prompt_attention_mask = ds["prompt_attention_mask"][
            idx : idx + config.batch_size
        ]
        answer_token_ids = ds["answer_input_ids"][idx : idx + config.batch_size]

        max_gen_tokens = [
            config.max_seq_length - len(prompt_attention_mask[i])
            for i in range(len(prompt_attention_mask))
        ]
        sampling_params.max_tokens = max(max_gen_tokens)

        request_outputs: List[RequestOutput] = vllm_actor.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        for prompt_idx, request_output in enumerate(request_outputs):
            offset_idx = idx + prompt_idx
            assert len(request_output.outputs) == 1, len(request_output.outpus)
            generation = request_output.outputs[0]
            generation_token_ids = generation.token_ids[: max_gen_tokens[prompt_idx]]
            generation_offset = len(prompt_token_ids[prompt_idx]) - 1

            is_correct, _ = is_answer_present(
                processing_class=tokenizer,
                answer_token_ids=answer_token_ids[prompt_idx],
                generation_token_ids=generation_token_ids,
                template_type=config.template_type,
            )
            is_eos = (generation_token_ids[-1] == stop_token_id) or (
                generation_token_ids[-1] == tokenizer.eos_token_id
            )
            all_is_corrects.append(is_correct)
            all_is_eos.append(is_eos)
            if not is_eos:
                continue

            token_ids = torch.tensor(
                prompt_token_ids[prompt_idx] + generation_token_ids,
                device="cuda",
                dtype=torch.long,
            )
            true_tokens = [
                tokenizer.decode(token, skip_special_tokens=False)
                for token in token_ids
            ]

            (
                prob_diffs,
                vanilla_probs,
                steered_probs,
                vanilla_entropy,
                steered_entropy,
            ) = get_prob_diffs(
                model=model,
                token_ids=token_ids,
                steering_vectors=steering_vectors,
                zero_steering_vectors=zero_steering_vectors,
                generation_temperature=config.generation_temperature,
            )
            next_token_diffs, max_vanilla, max_steered = get_for_nikita(
                prob_diffs=prob_diffs,
                vanilla_probs=vanilla_probs,
                steered_probs=steered_probs,
                token_ids=token_ids,
            )

            top_base_probs, top_base_tokens, top_steered_probs, top_steered_tokens = (
                get_top_probs(
                    vanilla_probs=vanilla_probs,
                    steered_probs=steered_probs,
                    prob_top_k=config.prob_top_k,
                    tokenizer=tokenizer,
                )
            )
            top_prob_diffs, top_tokens = get_top_diffs(
                prob_diffs=prob_diffs, tokenizer=tokenizer
            )
            bottom_prob_diffs, bottom_tokens = get_bottom_diffs(
                prob_diffs=prob_diffs, tokenizer=tokenizer
            )

            would_be_tokens = [toks[0] for toks in top_base_tokens]
            max_steered_tokens = [toks[0] for toks in top_steered_tokens]

            if offset_idx < config.log_first_k_samples:
                savedir = os.path.join(config.savedir, f"sample_{offset_idx}", "dump")
                os.makedirs(savedir, exist_ok=True)

                max_vanilla_tokens_ = [
                    tokenizer.decode(toks, skip_special_tokens=False)
                    for toks in max_vanilla
                ]

                max_steered_tokens_ = [
                    tokenizer.decode(toks, skip_special_tokens=False)
                    for toks in max_steered
                ]

                with open(os.path.join(savedir, "true_tokens.pkl"), "+wb") as f:
                    pkl.dump(true_tokens, f)
                np.save(
                    os.path.join(savedir, "next_token_diffs.npy"),
                    next_token_diffs.float().cpu().numpy(),
                )
                np.save(
                    os.path.join(savedir, "max_vanilla.npy"), max_vanilla.cpu().numpy()
                )
                np.save(
                    os.path.join(savedir, "max_steered.npy"), max_steered.cpu().numpy()
                )

                with open(os.path.join(savedir, "max_vanilla_tokens.pkl"), "+wb") as f:
                    pkl.dump(max_vanilla_tokens_, f)
                with open(os.path.join(savedir, "max_steered_tokens.pkl"), "+wb") as f:
                    pkl.dump(max_steered_tokens_, f)

                tsv_for_nikita(
                    tokens=true_tokens,
                    p_s=max_steered_tokens_,
                    p_b=max_vanilla_tokens_,
                    token_diffs=next_token_diffs.float().cpu().numpy(),
                    savedir=savedir,
                )

                for name, offset in zip(
                    ["all_positions", "generation_positions"], [0, generation_offset]
                ):
                    for side_name, (side_prob_diffs, side_tokens) in zip(
                        ["top", "bottom"],
                        [
                            (top_prob_diffs, top_tokens),
                            (bottom_prob_diffs, bottom_tokens),
                        ],
                    ):
                        savedir = os.path.join(
                            config.savedir, f"sample_{offset_idx}", name, side_name
                        )
                        os.makedirs(savedir, exist_ok=True)
                        plot_prob_diffs_hist(
                            prob_diffs=side_prob_diffs.float().cpu().numpy()[offset:],
                            savedir=savedir,
                        )
                        save_sorted_prob_diffs(
                            prob_diffs=side_prob_diffs.float().cpu().numpy()[offset:],
                            tokens=side_tokens[offset:],
                            src_tokens=true_tokens[offset:],
                            would_be_tokens=would_be_tokens[offset:],
                            reverse=side_name == "top",
                            is_correct=is_correct,
                            savedir=savedir,
                        )
                        save_top_prob_diffs_by_position(
                            prob_diffs=side_prob_diffs.float().cpu().numpy()[offset:],
                            tokens=side_tokens[offset:],
                            src_tokens=true_tokens[offset:],
                            would_be_tokens=would_be_tokens[offset:],
                            is_correct=is_correct,
                            savedir=savedir,
                        )

                    savedir = os.path.join(config.savedir, f"sample_{offset_idx}", name)

                    save_plain_text(
                        src_tokens=true_tokens[offset:],
                        is_correct=is_correct,
                        savedir=savedir,
                    )

                    init_probs_vs_prob_diff(
                        prob_diffs=prob_diffs.float().cpu().numpy()[offset:],
                        vanilla_probs=vanilla_probs.float().cpu().numpy()[offset:],
                        steered_probs=steered_probs.float().cpu().numpy()[offset:],
                        savedir=savedir,
                    )
                    save_top_probs_sets(
                        prob_diffs=side_prob_diffs.float().cpu().numpy()[offset:],
                        base_probs=top_base_probs.float().cpu().numpy()[offset:],
                        base_tokens=top_base_tokens[offset:],
                        steered_probs=top_steered_probs.float().cpu().numpy()[offset:],
                        steered_tokens=top_steered_tokens[offset:],
                        src_tokens=true_tokens[offset:],
                        prob_threshold=config.prob_threshold,
                        is_correct=is_correct,
                        savedir=savedir,
                    )

            all_top_diffs.extend(
                top_prob_diffs.float().cpu().tolist()[generation_offset:]
            )
            all_top_tokens_dst.extend(top_tokens[generation_offset:])
            all_tokens_src.extend(true_tokens[generation_offset:])
            all_tokens_would_be.extend(would_be_tokens[generation_offset:])
            all_tokens_max_steered.extend(max_steered_tokens[generation_offset:])
            all_bottom_diffs.extend(
                bottom_prob_diffs.float().cpu().tolist()[generation_offset:]
            )
            all_bottom_tokens_dst.extend(bottom_tokens[generation_offset:])
            all_vanilla_entropies.extend(
                vanilla_entropy.float().cpu().tolist()[generation_offset:]
            )
            all_steered_entropies.extend(
                steered_entropy.float().cpu().tolist()[generation_offset:]
            )
            all_positions.extend(
                range(len(top_prob_diffs.float().cpu().tolist()[generation_offset:]))
            )

            assert prob_diffs.dim() == 2, prob_diffs.dim()
            all_token_to_prob_diffs.extend(
                prob_diffs[:, token_to_idx].float().cpu().tolist()[generation_offset:]
            )

    for sorted_by in ["mean", "max"]:
        for is_filtered in [True, False]:
            plot_ds_wise(
                all_top_diffs=all_top_diffs,
                all_top_tokens_dst=all_top_tokens_dst,
                all_top_tokens_src=all_tokens_src,
                all_top_tokens_would_be=all_tokens_would_be,
                all_bottom_diffs=all_bottom_diffs,
                all_bottom_tokens_dst=all_bottom_tokens_dst,
                all_positions=all_positions,
                top_k=config.prob_top_k,
                boxplot_width=config.boxplot_width,
                sorted_by=sorted_by,
                is_filtered=is_filtered,
                savedir=os.path.join(
                    config.savedir,
                    "generation_positions",
                    f"is_filtered_{is_filtered}",
                    f"prob_diff_by. {sorted_by}_sorted",
                ),
            )

    token_to_vis_position(
        prob_diffs=all_token_to_prob_diffs,
        positions=all_positions,
        savepath=os.path.join(
            config.savedir, "generation_positions", "token_to_prob_diff.pdf"
        ),
    )

    frac_reinforce_top(
        diffs=all_top_diffs,
        tokens_dst=all_top_tokens_dst,
        tokens_would_be=all_tokens_would_be,
        tokens_src=all_tokens_src,
        savedir=os.path.join(
            config.savedir, "generation_positions", "frac_reinforce_top"
        ),
    )
    frac_reinforce_top(
        diffs=all_top_diffs,
        tokens_dst=all_top_tokens_dst,
        tokens_would_be=all_tokens_max_steered,
        tokens_src=all_tokens_src,
        savedir=os.path.join(config.savedir, "generation_positions", "frac_change_top"),
    )

    plot_prob_diffs_vs_entropy(
        prob_diffs=all_top_diffs,
        entropy=all_vanilla_entropies,
        savedir=os.path.join(config.savedir, "generation_positions", "top"),
    )
    plot_prob_diffs_vs_entropy(
        prob_diffs=all_bottom_diffs,
        entropy=all_vanilla_entropies,
        savedir=os.path.join(config.savedir, "generation_positions", "bottom"),
    )

    with open(os.path.join(config.savedir, "all_is_corrects.txt"), "+w") as f:
        num_correct = sum(all_is_corrects)
        f.write(f"Num correct: {num_correct}/{len(all_is_corrects)}\n")
        num_eos = sum(all_is_eos)
        f.write(f"Num eos: {num_eos}/{len(all_is_eos)}\n")

    pkl_savedir = os.path.join(config.savedir, "pkls")
    os.makedirs(pkl_savedir, exist_ok=True)
    np.save(os.path.join(pkl_savedir, "all_top_diffs.npy"), np.asarray(all_top_diffs))
    with open(os.path.join(pkl_savedir, "all_top_tokens_dst.npy"), "+wb") as f:
        pkl.dump(all_top_tokens_dst, f)
    np.save(
        os.path.join(pkl_savedir, "all_bottom_diffs.npy"), np.asarray(all_bottom_diffs)
    )
    with open(os.path.join(pkl_savedir, "all_bottom_tokens_dst.npy"), "+wb") as f:
        pkl.dump(all_bottom_tokens_dst, f)
    with open(os.path.join(pkl_savedir, "all_tokens_src.pkl"), "+wb") as f:
        pkl.dump(all_tokens_src, f)
    with open(os.path.join(pkl_savedir, "all_tokens_would_be.pkl"), "+wb") as f:
        pkl.dump(all_tokens_would_be, f)
    with open(os.path.join(pkl_savedir, "all_tokens_max_steered.pkl"), "+wb") as f:
        pkl.dump(all_tokens_max_steered, f)
    np.save(os.path.join(pkl_savedir, "all_positions.npy"), np.asarray(all_positions))
    np.save(
        os.path.join(pkl_savedir, "all_token_to_prob_diffs.npy"),
        np.asarray(all_token_to_prob_diffs),
    )


if __name__ == "__main__":
    run()
