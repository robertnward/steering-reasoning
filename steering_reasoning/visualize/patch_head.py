import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pyrallis

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.common import COLORS, dict_to_benchmark_df
from steering_reasoning.visualize.utils.defs import Model, ModelGroup
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    seed: int

    loaddir: str
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def get_qwen2_5_math_7b_model_groups_patch_heads() -> tuple[
    list[ModelGroup], list[Model]
]:
    models = []
    for proj_type in [
        "head",
        "group",
        "q_proj",
        "k_proj",
        "v_proj",
        "all_v_proj",
        "inverse_head",
        "inverse_group",
        "inverse_q_proj",
        "inverse_k_proj",
        "inverse_v_proj",
        "inverse_all_v_proj",
        "all_attn",
        "all_layer",
        "skip_attn",
        "skip_layer",
        "no_patch",
        "base",
    ]:
        if proj_type in ["q_proj", "inverse_q_proj", "head", "inverse_head"]:
            num_heads = 28
        elif proj_type in [
            "k_proj",
            "v_proj",
            "inverse_k_proj",
            "inverse_v_proj",
            "group",
            "inverse_group",
        ]:
            num_heads = 4
        elif proj_type in [
            "all_attn",
            "skip_attn",
            "all_layer",
            "skip_layer",
            "no_patch",
            "base",
            "all_v_proj",
            "inverse_all_v_proj",
        ]:
            num_heads = 1
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        for head_idx in range(num_heads):
            pkl_path = f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/patch-{proj_type}-{head_idx}/append_to_False/temp_1.0_top_p_1.0"
            if os.path.exists(pkl_path):
                models.append(
                    Model(
                        label=f"patch-{proj_type}-{head_idx}",
                        pkl_path=pkl_path,
                        name=f"patch-{proj_type}-{head_idx}",
                        ckpt=None,
                    )
                )

    other_models = [
        # Model(
        #     label="Base Model",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_0.0_top_p_1.0",
        #     name="Base Model",
        #     ckpt=None,
        # ),
        # Model(
        #     label=r"Base Model - $\tau=1$",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_1.0_top_p_1.0",
        #     name=r"Base Model - $\tau=1$",
        #     ckpt=None,
        # ),
        Model(
            label="Skip Attn",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/patch-all_attn-0/append_to_False/temp_1.0_top_p_1.0",
            name="Skip Attn",
            ckpt=None,
        ),
        Model(
            label="Skip Layer",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/patch-all_layer-0/append_to_False/temp_1.0_top_p_1.0",
            name="Skip Layer",
            ckpt=None,
        ),
        Model(
            label="Steering-Layer-26",
            # pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-0_lr-0.1/checkpoint-314/temp_1.0_top_p_1.0",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/patch-no_patch-0/append_to_False/temp_1.0_top_p_1.0",
            name="Steering-Layer-26",
            ckpt=None,
        ),
        Model(
            label="Steering-Layer-27",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-0_lr-0.1/checkpoint-265/temp_1.0_top_p_1.0",
            name="Steering-Layer-27",
            ckpt=None,
        ),
    ]
    model_groups = [
        ModelGroup("Qwen2.5-Math-7B. Patch Heads", models=models + other_models)
    ]
    return model_groups, other_models


def get_llama3_1_8b_chat_model_groups_patch_heads(
    old_setup: bool,
) -> tuple[list[ModelGroup], list[Model]]:
    models = []
    for proj_type in [
        "head",
        "group",
        "group3_and",
        "group4_and",
        "group6_and",
        "q_proj",
        "k_proj",
        "v_proj",
        "all_v_proj",
        "inverse_head",
        "inverse_group",
        "inverse_group7_and",
        "inverse_group7_and1_and",
        "inverse_group7_and6_and",
        "inverse_group7_and1_and2_and",
        "inverse_group7_and1_and5_and",
        "inverse_q_proj",
        "inverse_k_proj",
        "inverse_v_proj",
        "inverse_all_v_proj",
        "all_attn",
        "all_layer",
        "skip_attn",
        "skip_layer",
        "no_patch",
        "no_patch_imp",
        "base",
    ]:
        patch_paths = (
            ["all", "residual", "mlp"]
            if proj_type in ["head", "inverse_head"]
            else ["all"]
        )
        if proj_type in ["q_proj", "inverse_q_proj", "head", "inverse_head"]:
            num_heads = 32
        elif proj_type in [
            "k_proj",
            "v_proj",
            "inverse_k_proj",
            "inverse_v_proj",
            "group",
            "group3_and",
            "group4_and",
            "group6_and",
            "inverse_group",
            "inverse_group7_and",
            "inverse_group7_and1_and",
            "inverse_group7_and6_and",
            "inverse_group7_and1_and2_and",
            "inverse_group7_and1_and5_and",
        ]:
            num_heads = 8
        elif proj_type in [
            "all_attn",
            "all_layer",
            "skip_attn",
            "skip_layer",
            "no_patch",
            "no_patch_imp",
            "base",
            "all_v_proj",
            "inverse_all_v_proj",
        ]:
            num_heads = 1
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        for patch_path in patch_paths:
            for head_idx in range(num_heads):
                if not old_setup:
                    pkl_path = f"/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-{proj_type}-{head_idx}-{patch_path}/transformers_backend/temp_1.0_top_p_1.0"
                    if os.path.exists(pkl_path):
                        models.append(
                            Model(
                                label=f"patch-{proj_type}_{patch_path}-{head_idx}",
                                pkl_path=pkl_path,
                                name=f"patch-{proj_type}_{patch_path}-{head_idx}",
                                ckpt=None,
                            )
                        )
                if old_setup:
                    pkl_path = f"/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-{proj_type}-{head_idx}-all/temp_1.0_top_p_1.0"
                    if os.path.exists(pkl_path):
                        models.append(
                            Model(
                                label=f"patch-{proj_type}-{head_idx}",
                                pkl_path=pkl_path,
                                name=f"patch-{proj_type}-{head_idx}",
                                ckpt=None,
                            )
                        )

    if old_setup:
        other_models = [
            # Model(
            #     label="Base Model",
            #     pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
            #     name="Base Model",
            #     ckpt=None,
            # ),
            # Model(
            #     label=r"Base Model. $\tau=1$",
            #     pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
            #     name=r"Base Model. $\tau=1$",
            #     ckpt=None,
            # ),
            Model(
                label="Skip Attn",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-all_attn-0/temp_1.0_top_p_1.0",
                name="Skip Attn",
                ckpt=None,
            ),
            Model(
                label="Skip Layer",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-all_layer-0/temp_1.0_top_p_1.0",
                name="Skip Layer",
                ckpt=None,
            ),
            Model(
                label="Steering-Layer-30",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/append_to_False/temp_1.0_top_p_1.0",
                name="Steering-Layer-30",
                ckpt=None,
            ),
            Model(
                label="Steering-Layer-31",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-31/seed-0_lr-0.005/checkpoint-159/append_to_False/temp_1.0_top_p_1.0",
                name="Steering-Layer-31",
                ckpt=None,
            ),
        ]
    else:
        other_models = [
            # Model(
            #     label="Base Model",
            #     pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
            #     name="Base Model",
            #     ckpt=None,
            # ),
            # Model(
            #     label=r"Base Model. $\tau=1$",
            #     pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
            #     name=r"Base Model. $\tau=1$",
            #     ckpt=None,
            # ),
            Model(
                label="Skip Attn",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-skip_attn-0-all/transformers_backend/temp_1.0_top_p_1.0",
                name="Skip Attn",
                ckpt=None,
            ),
            Model(
                label="Skip Layer",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-skip_layer-0-all/transformers_backend/temp_1.0_top_p_1.0",
                name="Skip Layer",
                ckpt=None,
            ),
            Model(
                label="Steering-Layer-30",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/transformers_backend/temp_1.0_top_p_1.0",
                name="Steering-Layer-30",
                ckpt=None,
            ),
            Model(
                label="Steering-Layer-31",
                pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-31/seed-0_lr-0.005/checkpoint-159/transformers_backend/temp_1.0_top_p_1.0",
                name="Steering-Layer-31",
                ckpt=None,
            ),
        ]
    other_models = [
        x for x in other_models if x.pkl_path is not None and os.path.exists(x.pkl_path)
    ]
    model_groups = [
        ModelGroup("llama3.1-8b-chat. Patch Heads", models=models + other_models)
    ]
    return model_groups, other_models


def get_llama3_1_8b_chat_model_temp_0_groups_patch_heads() -> tuple[
    list[ModelGroup], list[Model]
]:
    models = []
    for proj_type in [
        "head",
        "group",
        "group3_and",
        "group4_and",
        "group6_and",
        "q_proj",
        "k_proj",
        "v_proj",
        "all_v_proj",
        "inverse_head",
        "inverse_group",
        "inverse_group7_and",
        "inverse_group7_and1_and",
        "inverse_group7_and6_and",
        "inverse_group7_and1_and2_and",
        "inverse_group7_and1_and5_and",
        "inverse_q_proj",
        "inverse_k_proj",
        "inverse_v_proj",
        "inverse_all_v_proj",
        "all_attn",
        "all_layer",
        "skip_attn",
        "skip_layer",
        "no_patch",
        "no_patch_imp",
        "base",
    ]:
        patch_paths = (
            ["all", "residual", "mlp"]
            if proj_type in ["head", "inverse_head"]
            else ["all"]
        )
        if proj_type in ["q_proj", "inverse_q_proj", "head", "inverse_head"]:
            num_heads = 32
        elif proj_type in [
            "k_proj",
            "v_proj",
            "inverse_k_proj",
            "inverse_v_proj",
            "group",
            "group3_and",
            "group4_and",
            "group6_and",
            "inverse_group",
            "inverse_group7_and",
            "inverse_group7_and1_and",
            "inverse_group7_and6_and",
            "inverse_group7_and1_and2_and",
            "inverse_group7_and1_and5_and",
        ]:
            num_heads = 8
        elif proj_type in [
            "all_attn",
            "all_layer",
            "skip_attn",
            "skip_layer",
            "no_patch",
            "no_patch_imp",
            "all_v_proj",
            "inverse_all_v_proj",
            "base",
        ]:
            num_heads = 1
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        for patch_path in patch_paths:
            for head_idx in range(num_heads):
                pkl_path = f"/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-{proj_type}-{head_idx}-{patch_path}/transformers_backend/temp_0.0_top_p_1.0"
                if os.path.exists(pkl_path):
                    models.append(
                        Model(
                            label=f"patch-{proj_type}_{patch_path}-{head_idx}",
                            pkl_path=pkl_path,
                            name=f"patch-{proj_type}_{patch_path}-{head_idx}",
                            ckpt=None,
                        )
                    )

    other_models = [
        # Model(
        #     label="Base Model",
        #     pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
        #     name="Base Model",
        #     ckpt=None,
        # ),
        # Model(
        #     label=r"Base Model. $\tau=1$",
        #     pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
        #     name=r"Base Model. $\tau=1$",
        #     ckpt=None,
        # ),
        Model(
            label="Skip Attn",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-skip_attn-0-all/transformers_backend/temp_0.0_top_p_1.0",
            name="Skip Attn",
            ckpt=None,
        ),
        Model(
            label="Skip Layer",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/patch-skip_layer-0-all/transformers_backend/temp_0.0_top_p_1.0",
            name="Skip Layer",
            ckpt=None,
        ),
        Model(
            label="Steering-Layer-30",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-106/transformers_backend/temp_0.0_top_p_1.0",
            name="Steering-Layer-30",
            ckpt=None,
        ),
        Model(
            label="Steering-Layer-31",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-31/seed-0_lr-0.005/checkpoint-159/transformers_backend/temp_0.0_top_p_1.0",
            name="Steering-Layer-31",
            ckpt=None,
        ),
    ]
    other_models = [
        x for x in other_models if x.pkl_path is not None and os.path.exists(x.pkl_path)
    ]
    model_groups = [
        ModelGroup("llama3.1-8b-chat. Patch Heads", models=models + other_models)
    ]
    return model_groups, other_models


def split_by_proj_type(
    data: Dict[str, Dict[Model, Dict[str, float]]],
    other_models: List[Model],
) -> Tuple[
    Dict[str, Dict[str, Dict[Model, Dict[str, float]]]],  # ← patched models
    Dict[str, Dict[Model, Dict[str, float]]],  # ← baselines
]:
    """
    Separate *patched* models (which must follow the
    ``patch-{proj_type}-{head_idx}`` naming convention) from *other* “baseline”
    models whose labels do **not** match the pattern or which are listed
    explicitly in *other_models*.

    Returns
    -------
    patched_split : dict
        ``{proj_type → {bench → {model → metrics}}}``
    baselines     : dict
        ``{bench → {model → metrics}}``
    """
    baseline_set: Set["Model"] = set(other_models)

    patched_split: Dict[str, Dict[str, Dict["Model", Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    baselines: Dict[str, Dict["Model", Dict[str, float]]] = defaultdict(dict)

    for bench, model_dict in data.items():
        for model, metrics in model_dict.items():
            # Explicit baseline?
            if model in baseline_set:
                baselines[bench][model] = metrics
                continue

            # Does it *look* like a patch?
            try:
                prefix, proj_type, _ = model.label.split("-", 2)
            except ValueError:
                # Not a patch → treat as baseline
                baselines[bench][model] = metrics
                continue

            if prefix != "patch":
                baselines[bench][model] = metrics
                continue

            # Valid patch → keep under its proj_type
            patched_split[proj_type][bench][model] = metrics

    # Turn the defaultdict nest back into normal dicts
    patched_split = {
        p: {b: dict(models) for b, models in outer.items()}
        for p, outer in patched_split.items()
    }
    baselines = {b: dict(models) for b, models in baselines.items()}
    return patched_split, baselines


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Unchanged helper - still only for patched models
# ──────────────────────────────────────────────────────────────────────────────
def _extract_sorted_points(
    model_metrics: Dict["Model", Dict[str, float]],
) -> Tuple[List[int], List[float], List[float]]:
    def head_idx(m: "Model") -> int:
        return int(m.label.rsplit("-", maxsplit=1)[-1])

    sorted_items = sorted(model_metrics.items(), key=lambda kv: head_idx(kv[0]))

    xs, means, stds = [], [], []
    for model, stats in sorted_items:
        xs.append(head_idx(model))
        means.append(stats["mean"])
        stds.append(stats["std"])
    return xs, means, stds


def vis_split_dict_paper(
    avg_metrics: Dict[Model, Dict[str, float]],
    baseline_metrics: Dict[Model, Dict[str, float]],
    proj_type: str,
    model: str,
    savedir: str,
) -> None:
    """
    Render a single steering-head plot *for the already averaged numbers*.

    Parameters
    ----------
    avg_metrics
        ``split_dict[proj_type]["Average"]`` - maps patch-*head* models
        ➜ ``{"mean": …, "std": …}``.
    baseline_metrics
        ``baselines["Average"]`` - same shape but for the baseline models.
    proj_type
        Title for the plot.
    savedir
        Output directory (created if necessary).
    """
    os.makedirs(savedir, exist_ok=True)

    with mpl.rc_context(
        {
            "figure.figsize": (5, 5),
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
        fig, ax = plt.subplots(constrained_layout=True)

        # -- patched heads --
        xs, means, stds = _extract_sorted_points(avg_metrics)
        if xs:
            ax.plot(xs, means, marker="o", linewidth=2.5, color="#F28E2B")
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.25,
                color="#F28E2B",
            )
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter("{x:.0f}")

        # -- baseline(s) --
        for base_model, stats in baseline_metrics.items():
            yref = stats["mean"]
            yref_std = stats["std"]
            ax.axhline(yref, linestyle="--", linewidth=2, color="black", zorder=1)
            ax.fill_between(
                xs, yref - yref_std, yref + yref_std, alpha=0.1, color="black", zorder=0
            )
            if model == "llama3.1-8b-chat. temp=1":
                if base_model.label == "Steering-Layer-30":
                    label = r"$\mathrm{s}_{30}$"
                    ytext = 10
                elif base_model.label == "Steering-Layer-31":
                    label = r"$\mathrm{s}_{31}$"
                    ytext = 10
                elif base_model.label == "Skip Attn":
                    label = "Skip-Attn"
                    ytext = -30
                elif base_model.label == "Skip Layer":
                    label = "Skip-Layer"
                    ytext = 7
                else:
                    raise ValueError(f"Unknown baseline model: {base_model.label}")
            elif model == "llama3.1-8b-chat. temp=0":
                if base_model.label == "Steering-Layer-30":
                    label = r"$\mathrm{s}_{30}$"
                    ytext = 15
                elif base_model.label == "Steering-Layer-31":
                    label = r"$\mathrm{s}_{31}$"
                    ytext = 15
                elif base_model.label == "Skip Attn":
                    label = "Skip-Attn"
                    ytext = -30
                elif base_model.label == "Skip Layer":
                    label = "Skip-Layer"
                    ytext = -30
                else:
                    raise ValueError(f"Unknown baseline model: {base_model.label}")
            if model == "Qwen2.5-Math-7B. temp=1":
                if base_model.label == "Steering-Layer-26":
                    label = r"$\mathrm{s}_{26}$"
                    ytext = 15
                elif base_model.label == "Steering-Layer-27":
                    label = r"$\mathrm{s}_{27}$"
                    ytext = 15
                elif base_model.label == "Skip Attn":
                    label = "Skip-Attn"
                    ytext = -20
                elif base_model.label == "Skip Layer":
                    label = "Skip-Layer"
                    ytext = 15
                else:
                    raise ValueError(f"Unknown baseline model: {base_model.label}")

            ax.annotate(
                label,
                xy=(0.995, yref),
                xycoords=("axes fraction", "data"),
                xytext=(0, ytext),
                textcoords="offset points",
                ha="right",
                va="bottom",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                ),
                zorder=5,
            )

        ax.set_xlabel("Head")
        ax.set_ylabel("Mean Acc.")
        if proj_type == "head_all":
            ax.set_title("Skip-Head")
        elif proj_type == "inverse_head_all":
            ax.set_title("Steer-Head")
        else:
            ax.set_title(proj_type)

        fig.savefig(
            os.path.join(savedir, f"{proj_type}.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.close(fig)


def vis_split_dict_paper_multi(
    avg_metrics_list: List[Dict[Model, Dict[str, float]]],
    baseline_metrics: Dict[Model, Dict[str, float]],
    proj_types: List[str],
    model: str,
    savedir: str,
    name: str = "combined",
    outfile: str | None = None,
) -> None:
    """
    Side-by-side version of *vis_split_dict_paper* that shares the same baseline.

    Parameters
    ----------
    avg_metrics_list
        List of ``split_dict[proj_type]["Average"]`` dictionaries - one per panel.
    baseline_metrics
        ``baselines["Average"]`` - shared across all panels.
    proj_types
        Titles for the panels (same length/order as ``avg_metrics_list``).
    savedir
        Directory where the PDF will be written.
    name
        Base file name (used if ``outfile`` is ``None``).
    outfile
        Optional explicit file name (without extension).
    """
    if len(avg_metrics_list) == 0:
        raise ValueError("avg_metrics_list must contain at least one element")
    if len(avg_metrics_list) != len(proj_types):
        raise ValueError("proj_types must align with avg_metrics_list")

    ncols = len(avg_metrics_list)
    os.makedirs(savedir, exist_ok=True)

    with mpl.rc_context(
        {
            "figure.figsize": (5 * ncols, 5),
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
        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            sharey=True,
            constrained_layout=True,
        )
        if ncols == 1:  # keep iterable
            axes = [axes]

        for ax, avg_metrics, title in zip(axes, avg_metrics_list, proj_types):
            xs, means, stds = _extract_sorted_points(avg_metrics)
            if xs:
                ax.plot(xs, means, marker="o", linewidth=2.5, color="#F28E2B")
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25,
                    color="#F28E2B",
                )
                ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                ax.xaxis.set_major_formatter("{x:.0f}")

            # shared baseline(s)
            for base_model, stats in baseline_metrics.items():
                yref = stats["mean"]
                yref_std = stats["std"]
                ax.axhline(yref, linestyle="--", linewidth=2, color="black", zorder=1)
                ax.fill_between(
                    xs,
                    yref - yref_std,
                    yref + yref_std,
                    alpha=0.1,
                    color="black",
                    zorder=0,
                )
                if model == "llama3.1-8b-chat. temp=1":
                    if base_model.label == "Steering-Layer-30":
                        label = r"$\mathrm{s}_{30}$"
                        ytext = 10
                    elif base_model.label == "Steering-Layer-31":
                        label = r"$\mathrm{s}_{31}$"
                        ytext = 10
                    elif base_model.label == "Skip Attn":
                        label = "Skip-Attn"
                        ytext = -25
                    elif base_model.label == "Skip Layer":
                        label = "Skip-Layer"
                        ytext = 7
                    else:
                        raise ValueError(f"Unknown baseline model: {base_model.label}")
                elif model == "llama3.1-8b-chat. temp=0":
                    if base_model.label == "Steering-Layer-30":
                        label = r"$\mathrm{s}_{30}$"
                        ytext = 15
                    elif base_model.label == "Steering-Layer-31":
                        label = r"$\mathrm{s}_{31}$"
                        ytext = 15
                    elif base_model.label == "Skip Attn":
                        label = "Skip-Attn"
                        ytext = -30
                    elif base_model.label == "Skip Layer":
                        label = "Skip-Layer"
                        ytext = -30
                    else:
                        raise ValueError(f"Unknown baseline model: {base_model.label}")
                if model == "Qwen2.5-Math-7B. temp=1":
                    if base_model.label == "Steering-Layer-26":
                        label = r"$\mathrm{s}_{26}$"
                        ytext = 15
                    elif base_model.label == "Steering-Layer-27":
                        label = r"$\mathrm{s}_{27}$"
                        ytext = 15
                    elif base_model.label == "Skip Attn":
                        label = "Skip-Attn"
                        ytext = -40
                    elif base_model.label == "Skip Layer":
                        label = "Skip-Layer"
                        ytext = 15
                    else:
                        raise ValueError(f"Unknown baseline model: {base_model.label}")
                ax.annotate(
                    label,
                    xy=(0.995, yref),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, ytext),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.9,
                    ),
                    zorder=5,
                )

            ax.set_title(title)
            ax.set_xlabel("Head")
        axes[0].set_ylabel("Mean Acc.")  # only left-most panel needs a y-label

        filename = outfile or f"{name}_steering_heads.pdf"
        fig.savefig(os.path.join(savedir, filename), format="pdf", bbox_inches="tight")
        plt.close(fig)


def visualize_split_dict(  # noqa: C901 - still within reasonable complexity
    split_dict: Dict[str, Dict[str, Dict[Model, Dict[str, float]]]],
    baselines: Dict[str, Dict[Model, Dict[str, float]]],
    model: str,
    savedir: str,
    sharey: bool = False,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.0),
) -> None:
    """
    Plot per-head accuracy curves for every benchmark *including* the pre-computed
    “Average” benchmark that now ships with ``split_dict``.

    The function no longer:
      • builds global patch / baseline accumulators
      • inserts its own “Averaged” subplot
      • spawns the separate paper-style summary figures

    Parameters
    ----------
    split_dict
        First-level keys are projection types (“q_proj”, …); the second level now
        **must** contain an “Average” entry in addition to the individual
        benchmarks.
    baselines
        Same structure as before: ``baselines[bench][model] -> {"mean", "std"}``.
    savedir
        Output directory.
    sharey
        Forwarded to ``plt.subplots``.
    figsize_per_panel
        Physical size of *one* panel (width, height in inch).
    """
    os.makedirs(savedir, exist_ok=True)

    for proj_type, benches in split_dict.items():
        n_total_panels = len(benches)  # “Average” already included
        ncols = min(3, n_total_panels)
        nrows = math.ceil(n_total_panels / ncols)

        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            sharex=False,
            sharey=sharey,
            squeeze=False,
        )
        axes = axes.ravel()

        # ── one subplot per benchmark (incl. “Average”) ───────────────────
        for ax, (bench, model_metrics) in zip(axes, benches.items()):
            # Patched heads
            xs, means, stds = _extract_sorted_points(model_metrics)
            if xs:
                ax.plot(xs, means, marker="o")
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25,
                )
                ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)

            # Baseline horizontal lines (if any)
            for idx, (base_model, stats) in enumerate(baselines.get(bench, {}).items()):
                m = stats["mean"]
                hl = ax.axhline(
                    y=m,
                    color=COLORS[idx],
                    linestyle=":",
                    linewidth=1.5,
                    label=getattr(base_model, "label", str(base_model)),
                )
                ax.fill_between(
                    ax.get_xlim(),
                    [m - stats["std"]] * 2,
                    [m + stats["std"]] * 2,
                    alpha=0.15,
                    color=hl.get_color(),
                )

            ax.set_title(bench, fontsize="medium", pad=4)
            ax.set_ylabel("mean")
            ax.legend(loc="best", fontsize="x-small")

        # Hide any unused axes when the grid is larger than needed
        for ax in axes[len(benches) :]:
            ax.set_visible(False)

        fig.suptitle(f"Projection type: {proj_type}", fontsize="large")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(savedir, f"{proj_type}.png"))
        plt.close(fig)

        vis_split_dict_paper(
            avg_metrics=split_dict[proj_type]["Average"],
            baseline_metrics=baselines["Average"],
            proj_type=proj_type,
            model=model,
            savedir=os.path.join(savedir, "paper"),
        )

    # three-panel version
    if "q_proj" in split_dict and "k_proj" in split_dict and "v_proj" in split_dict:
        vis_split_dict_paper_multi(
            avg_metrics_list=[
                split_dict["q_proj"]["Average"],
                split_dict["k_proj"]["Average"],
                split_dict["v_proj"]["Average"],
            ],
            baseline_metrics=baselines["Average"],
            name="combined",
            proj_types=["Steer Q-Proj", "Steer K-Proj", "Steer V-Proj"],
            model=model,
            savedir=os.path.join(savedir, "paper"),
        )
    if (
        "inverse_q_proj" in split_dict
        and "inverse_k_proj" in split_dict
        and "inverse_v_proj" in split_dict
    ):
        vis_split_dict_paper_multi(
            avg_metrics_list=[
                split_dict["inverse_q_proj"]["Average"],
                split_dict["inverse_k_proj"]["Average"],
                split_dict["inverse_v_proj"]["Average"],
            ],
            baseline_metrics=baselines["Average"],
            proj_types=["Steer Q-Proj", "Steer K-Proj", "Steer V-Proj"],
            name="inverse_combined",
            model=model,
            savedir=os.path.join(savedir, "paper"),
        )


@pyrallis.wrap()
def run(config: Config):
    init_savedir = os.path.join(config.savedir, "patch_head")

    benches = [
        "AIME2025",
        "AIME_2024",
        "AMC-23",
        "MATH-500",
        "Minerva-Math",
        "OlympiadBench",
    ]

    for model_groups_name, (model_groups, other_models) in zip(
        [
            "llama3.1-8b-chat. temp=1",
            "llama3.1-8b-chat. temp=0",
            "Qwen2.5-Math-7B. temp=1",
        ],
        [
            get_llama3_1_8b_chat_model_groups_patch_heads(
                old_setup="patch_head2" not in init_savedir
            ),
            get_llama3_1_8b_chat_model_temp_0_groups_patch_heads(),
            get_qwen2_5_math_7b_model_groups_patch_heads(),
        ],
    ):
        config.savedir = os.path.join(init_savedir, model_groups_name)
        os.makedirs(config.savedir, exist_ok=True)
        # Final plots and tables
        (
            accuracies_dict,
            avg_accuracies_dict,
            all_lens_dict,
            all_correct_lens_dict,
            all_max_correct_lens_dict,
            all_incorrect_lens_dict,
        ) = run_for_model_groups(
            model_groups=model_groups,
            benches=benches,
            average_across_benches={
                "acc": ["MATH-500", "Minerva-Math", "OlympiadBench"],
                "avg_acc": ["AIME2025", "AIME_2024", "AMC-23"],
            },
        )
        accuracies_dict = {
            "AIME2025": avg_accuracies_dict["AIME2025"],
            "AIME_2024": avg_accuracies_dict["AIME_2024"],
            "AMC-23": avg_accuracies_dict["AMC-23"],
            "MATH-500": accuracies_dict["MATH-500"],
            "Minerva-Math": accuracies_dict["Minerva-Math"],
            "OlympiadBench": accuracies_dict["OlympiadBench"],
            "Average": accuracies_dict["Average"],
        }
        savedir = os.path.join(config.savedir, "tables")
        os.makedirs(savedir, exist_ok=True)
        for name, values in zip(
            ["accuracies", "lens", "correct_lens", "incorrect_lens"],
            [
                accuracies_dict,
                all_lens_dict,
                all_correct_lens_dict,
                all_incorrect_lens_dict,
            ],
        ):
            dict_to_benchmark_df(
                results=values, csv_path=os.path.join(savedir, f"{name}.csv")
            )

            by_proj_type, baselines = split_by_proj_type(
                data=values, other_models=other_models
            )
            visualize_split_dict(
                split_dict=by_proj_type,
                baselines=baselines,
                model=model_groups_name,
                savedir=os.path.join(config.savedir, "proj_types", name),
            )


if __name__ == "__main__":
    run()
