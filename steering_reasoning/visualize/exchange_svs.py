import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyrallis

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.common import dict_to_benchmark_df
from steering_reasoning.visualize.utils.defs import Model, ModelGroup
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    seed: int

    loaddir: str
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def get_models_helper(
    model_to_names: List[str],
    model_from_names: List[str],
    base_models: List[Model],
    temp: float,
) -> Tuple[List[ModelGroup], List[Model]]:
    models = base_models

    for model_to in model_to_names:
        for model_from in model_from_names:
            pkl_path = (
                f"/from_s3/results/{model_to}/from/{model_from}/temp_{temp}_top_p_1.0"
            )
            if os.path.exists(pkl_path):
                models.append(
                    Model(
                        label=model_from.split("/")[0],
                        pkl_path=pkl_path,
                        name=model_to,
                        ckpt=None,
                    )
                )

    model_groups = [ModelGroup("Qwen2.5-Math-7B. Pair Layers", models=models)]

    return model_groups, []


def get_qwen_1_5b_models() -> Tuple[List[ModelGroup], List[Model]]:
    model_to_names = ["Qwen2.5-1.5B", "Qwen2.5-Math-1.5B", "Qwen2.5-1.5B-Instruct"]
    model_from_names = [
        "Qwen2.5-1.5B/deepscaler/steering/seed-0_lr-0.001/checkpoint-314",
        "Qwen2.5-Math-1.5B/deepscaler/steering/seed-0_lr-0.001/checkpoint-265",
        "Qwen2.5-1.5B-Instruct/deepscaler/steering/seed-0_lr-0.001/checkpoint-314",
    ]
    base_models = [
        Model(
            name="Qwen2.5-Math-1.5B",
            pkl_path="/from_s3/results/Qwen2.5-Math-1.5B/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
        Model(
            name="Qwen2.5-1.5B",
            pkl_path="/from_s3/results/Qwen2.5-1.5B/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
        Model(
            name="Qwen2.5-1.5B-Instruct",
            pkl_path="/from_s3/results/Qwen2.5-1.5B-Instruct/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
    ]

    return get_models_helper(
        model_to_names, model_from_names, base_models=base_models, temp=1.0
    )


def get_qwen_7b_models() -> Tuple[List[ModelGroup], List[Model]]:
    model_to_names = ["Qwen2.5-7B", "Qwen2.5-Math-7B", "Qwen2.5-7B-Instruct"]
    model_from_names = [
        "Qwen2.5-7B/deepscaler/steering/seed-0/checkpoint-159",
        "Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-314",
        "Qwen2.5-7B-Instruct/deepscaler/steering/seed-0_lr-0.001/checkpoint-159",
    ]
    base_models = [
        Model(
            name="Qwen2.5-Math-7B",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
        Model(
            name="Qwen2.5-7B",
            pkl_path="/from_s3/results/Qwen2.5-7B/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
        Model(
            name="Qwen2.5-7B-Instruct",
            pkl_path="/from_s3/results/Qwen2.5-7B-Instruct/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
    ]

    return get_models_helper(
        model_to_names, model_from_names, base_models=base_models, temp=1.0
    )


def get_llama_8b_models() -> Tuple[List[ModelGroup], List[Model]]:
    model_to_names = ["llama3.1-8b-chat", "llama3.1-8b"]
    model_from_names = [
        "llama3.1-8b-chat/deepscaler/steering/seed-0/checkpoint-265",
        "llama3.1-8b/deepscaler/steering/seed-0_lr-0.0003_last/checkpoint-314",
    ]
    base_models = [
        Model(
            name="llama3.1-8b",
            pkl_path="/from_s3/results/llama3.1-8b/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
        Model(
            name="llama3.1-8b-chat",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
    ]

    return get_models_helper(
        model_to_names, model_from_names, base_models=base_models, temp=1.0
    )


def get_llama_8b_temp_0_models() -> Tuple[List[ModelGroup], List[Model]]:
    model_to_names = ["llama3.1-8b-chat", "llama3.1-8b"]
    model_from_names = [
        "llama3.1-8b-chat/deepscaler/steering/seed-0/checkpoint-265",
        "llama3.1-8b/deepscaler/steering/seed-0_lr-0.0003_last/checkpoint-314",
    ]
    base_models = [
        Model(
            name="llama3.1-8b",
            pkl_path="/from_s3/results/llama3.1-8b/base/temp_0.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
        Model(
            name="llama3.1-8b-chat",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
            label="Base",
            ckpt=None,
        ),
    ]

    return get_models_helper(
        model_to_names, model_from_names, base_models=base_models, temp=0.0
    )


def _natural_key(s: str):
    # e.g., "Qwen2.5-7B-Instruct" -> ["qwen", 2, 5, "-b-", "instruct"]
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def plot_model_pairs_pcolormesh(
    scores_by_model: dict,
    value_key: str = "mean",
    to_order: Iterable[str] | None = None,  # if None, inferred automatically
    from_order: Iterable[str] | None = None,  # if None, inferred automatically
    annotate: bool = True,
    drop_missing_name: bool = True,
    savepath: str | os.PathLike = ".",
    dpi: int = 200,
    figsize: tuple[float, float] = (6.5, 4.5),
    normalize: bool = False,  # <-- NEW
    base_col_name: str = "Base",  # <-- NEW: which column is the "min" anchor
):
    """
    Build a (model_to × model_from) grid using pcolormesh, auto-infer orders, and save.

    Auto-inference:
      - Rows (to_order) := unique Model.label values (natural-sorted).
      - Cols (from_order) := unique Model.name values (natural-sorted), unless
        drop_missing_name=False and some names are falsy, in which case they are
        represented by the string "<base>".

    Normalization (if normalize=True):
      - Per row, the cell in column `base_col_name` is treated as the min,
      - and the cell where column name == that row's label is treated as the max.
      - Values (and std, if present) are scaled as (v - min) / (max - min).
      - If anchors are missing/NaN or max == min, that row is left unnormalized.
    """
    # Gather unique labels/names
    to_names_set = set()
    from_names_set = set()
    items = []

    for m, stats in scores_by_model.items():
        assert m.name is not None, m

        items.append((m, stats))
        to_names_set.add(m.name)
        from_names_set.add(m.label)

    # ---- Enforce the same canonical order for rows and columns ----
    if to_order is not None:
        canonical = [x for x in to_order]
    elif from_order is not None:
        canonical = [x for x in from_order]
    else:
        canonical = sorted(to_names_set.union(from_names_set), key=_natural_key)

    # Rows/cols follow the same relative order, filtered to existing labels/names
    to_names = [x for x in canonical if x in to_names_set]
    from_names = [x for x in canonical if x in from_names_set]

    # Build matrices
    Z = np.full((len(to_names), len(from_names)), np.nan, dtype=float)
    S = np.full_like(Z, np.nan)

    for m, stats in items:
        row_name = m.name
        assert m.name is not None, m
        col_name = m.label
        try:
            i = to_names.index(row_name)
            j = from_names.index(col_name)
        except ValueError:
            # Key not in inferred/provided order; skip
            continue
        Z[i, j] = float(stats[value_key])
        if "std" in stats and stats["std"] is not None:
            S[i, j] = float(stats["std"])

    # ---- Optional per-row normalization ----
    if normalize and len(to_names) > 0 and len(from_names) > 0:
        try:
            j_base = from_names.index(base_col_name)
        except ValueError:
            j_base = None

        for i, row_label in enumerate(to_names):
            # anchors
            min_val = None
            if j_base is not None and np.isfinite(Z[i, j_base]):
                min_val = Z[i, j_base]

            try:
                j_diag = from_names.index(
                    row_label
                )  # column with same name as row label
            except ValueError:
                j_diag = None

            max_val = None
            if j_diag is not None and np.isfinite(Z[i, j_diag]):
                max_val = Z[i, j_diag]

            # if anchors are invalid, skip normalization for this row
            if (
                min_val is None
                or max_val is None
                or not np.isfinite(min_val)
                or not np.isfinite(max_val)
                or max_val == min_val
            ):
                continue

            scale = max_val - min_val
            # normalize row i
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    Z[i, j] = (Z[i, j] - min_val) / scale
                if np.isfinite(S[i, j]):
                    S[i, j] = S[i, j] / scale

    # Grid coordinates (edges for pcolormesh)
    x = np.arange(len(from_names) + 1)
    y = np.arange(len(to_names) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(x, y, Z, shading="flat")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"{value_key}{' (row-normalized)' if normalize else ''}")

    # Center tick labels at cell centers
    ax.set_xticks(x[:-1] + 0.5, from_names, rotation=45, ha="right")
    ax.set_yticks(y[:-1] + 0.5, to_names)
    ax.set_xlabel("model_from (Model.name)")
    ax.set_ylabel("model_to (Model.label)")

    # Annotate cells
    if annotate:
        for i in range(len(to_names)):
            for j in range(len(from_names)):
                v = Z[i, j]
                if np.isfinite(v):
                    txt = f"{v:.2f}"
                    if np.isfinite(S[i, j]):
                        txt += f"\n±{S[i, j]:.2f}"
                    ax.text(j + 0.5, i + 0.5, txt, va="center", ha="center")

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, format="pdf", dpi=dpi, bbox_inches="tight")

    # Return the resolved orders too (useful for debugging/reuse)
    return fig, ax, savepath, to_names, from_names


@pyrallis.wrap()
def run(config: Config):
    init_savedir = os.path.join(config.savedir, "exchange-svs")

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
            "qwens_7b",
            "qwens_1.5b",
            "llama_8b",
            "llama_8b_temp_0",
        ],
        [
            get_qwen_7b_models(),
            get_qwen_1_5b_models(),
            get_llama_8b_models(),
            get_llama_8b_temp_0_models(),
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
        savedir = os.path.join(config.savedir, "tables", "temp_1.0_top_p_1.0")
        os.makedirs(savedir, exist_ok=True)
        dict_to_benchmark_df(
            results=accuracies_dict, csv_path=os.path.join(savedir, "accuracies.csv")
        )
        dict_to_benchmark_df(
            results=avg_accuracies_dict,
            csv_path=os.path.join(savedir, "accuracies.csv"),
        )
        dict_to_benchmark_df(
            results=all_lens_dict, csv_path=os.path.join(savedir, "lens.csv")
        )
        dict_to_benchmark_df(
            results=all_correct_lens_dict, csv_path=os.path.join(savedir, "correct.csv")
        )
        dict_to_benchmark_df(
            results=all_max_correct_lens_dict,
            csv_path=os.path.join(savedir, "max_correct.csv"),
        )
        dict_to_benchmark_df(
            results=all_incorrect_lens_dict,
            csv_path=os.path.join(savedir, "incorrect.csv"),
        )

        accuracies_dict = {
            "Average": accuracies_dict["Average"],
            "AIME2025": avg_accuracies_dict["AIME2025"],
            "AIME_2024": avg_accuracies_dict["AIME_2024"],
            "AMC-23": avg_accuracies_dict["AMC-23"],
            "MATH-500": accuracies_dict["MATH-500"],
            "Minerva-Math": accuracies_dict["Minerva-Math"],
            "OlympiadBench": accuracies_dict["OlympiadBench"],
        }

        for bench in accuracies_dict.keys():
            plot_model_pairs_pcolormesh(
                accuracies_dict[bench],
                value_key="mean",
                normalize=False,
                savepath=os.path.join(config.savedir, "unnorm", f"{bench}.pdf"),
            )
            plot_model_pairs_pcolormesh(
                accuracies_dict[bench],
                value_key="mean",
                normalize=True,
                savepath=os.path.join(config.savedir, "norm", f"{bench}.pdf"),
            )


if __name__ == "__main__":
    run()
