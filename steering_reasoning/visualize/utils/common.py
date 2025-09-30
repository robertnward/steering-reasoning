import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.defs import Model, SetupMapping

COLORS = ["red", "green", "purple", "orange", "brown", "yellow"]


def save_best_ckpts(
    best_models: List[Model],
    savedir: str | Path,
):
    def extract_layer_from_name(model: Model) -> int:
        pattern = re.compile(r"^steering-layer-(\d+)$")

        m = pattern.match(model.name)
        if m:
            return int(m.group(1))
        else:
            return -1

    best_models_ = sorted(best_models, key=extract_layer_from_name)
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "best-ckpts.txt"), "w") as f:
        for model in best_models_:
            f.write(f"{model.name}. Checkpoint: {model.ckpt}\n")


def select_best_ckpt_per_name(
    avg_scores: Dict[Model, Dict[str, float]],
) -> Dict[Model, float]:
    # Track best (model, score) per name
    best: dict[str, Tuple[Model, float]] = defaultdict(lambda: (None, float("-inf")))  # type: ignore

    for model, score in avg_scores.items():
        _, current_best_score = best[model.name]
        if score["mean"] > current_best_score or np.isnan(score["mean"]):
            best[model.name] = (model, score["mean"])

    # Re-shape into the requested `{Model: score}` mapping
    return {model: score for model, score in best.values()}


def relative_to_full_tune(
    df: pd.DataFrame,
    value_col: str = "Average",
    config_level: int = 1,
    anchor: str = "Full-Tune",
    base: str = "Base",
    new_col1: str = "Relative (%)",
    new_col2: str = "Relative Improvement (%)",
    decimals: int = 1,
) -> pd.DataFrame:
    """
    Add a column with values expressed as a percent of the anchor row (e.g. 'Full-Tune')
    inside each model group.

    Parameters
    ----------
    df : DataFrame
        MultiIndex rows: level 0 = model name, level 1 = configuration.
    value_col : str, default 'Average'
        Column whose values you want to relativize.
    config_level : int, default 1
        The index level that contains the configuration labels.
    anchor : str, default 'Full-Tune'
        Which configuration serves as the 100 % reference.
    new_col : str, default 'Relative (%)'
        Name of the column to store the percentage values.
    decimals : int, default 2
        How many decimals to keep after rounding.

    Returns
    -------
    DataFrame
        Original frame with an extra column ``new_col``.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex index.")

    # Make a working copy so we don’t mutate the caller’s DataFrame
    out = df.copy()

    def calc_relative(series):
        anchor_value = series.loc[series.index.get_level_values(config_level) == anchor]
        if len(anchor_value) != 0:
            return (series / anchor_value.iloc[0]) * 100
        else:
            return np.nan

    # Group by the model name (level 0) …
    out[new_col1] = (
        out.groupby(level=0)[value_col].transform(calc_relative).round(decimals)
    )

    def calc_relative_improvement(series):
        anchor_value = series.loc[series.index.get_level_values(config_level) == anchor]
        base_value = series.loc[series.index.get_level_values(config_level) == base]
        if len(anchor_value) != 0 and len(base_value) != 0:
            return (
                (series - base_value.iloc[0])
                / (anchor_value.iloc[0] - base_value.iloc[0])
            ) * 100
        else:
            return np.nan

    out[new_col2] = (
        out.groupby(level=0)[value_col]
        .transform(calc_relative_improvement)
        .round(decimals)
    )
    return out


# Compile patterns once
_P_QWEN = re.compile(r"^Qwen2\.5-(\d+(?:\.\d+)?)B$", re.I)
_P_QWEN_MATH = re.compile(r"^Qwen2\.5-Math-(\d+(?:\.\d+)?)B$", re.I)
_P_LLAMA = re.compile(r"^llama3\.1-(\d+(?:\.\d+)?)b-chat$", re.I)


def _classify(model_name: str):
    """
    Return (group, n) where lower group sorts earlier, then by numeric n.
    Unmatched models go to the end (group=3) with n=+inf.
    """
    m = _P_QWEN.match(model_name)
    if m:
        return (0, float(m.group(1)))
    m = _P_QWEN_MATH.match(model_name)
    if m:
        return (1, float(m.group(1)))
    m = _P_LLAMA.match(model_name)
    if m:
        return (2, float(m.group(1)))
    return (3, float("inf"))


def sort_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a pandas DataFrame whose index's first level is model name
    and second level is setup. Keeps the original setup order within
    each model.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "Expected a MultiIndex with (model, setup) as the first two levels."
        )

    # Remember the original first occurrence position of each model
    model_level = df.index.get_level_values(0)
    first_pos = {}
    for i, name in enumerate(model_level):
        if name not in first_pos:
            first_pos[name] = i

    # Work on a flat frame to build stable sort keys
    tmp = df.reset_index()
    model_col, setup_col = tmp.columns[:2]  # names of the two index columns
    tmp["_row_pos"] = np.arange(
        len(tmp)
    )  # for stability within identical model+setup groups

    # Derive group and N
    gn = tmp[model_col].map(_classify)
    tmp["_group"] = gn.map(lambda t: t[0])
    tmp["_n"] = gn.map(lambda t: t[1])

    # Tie-break equally ranked models by their first appearance in the original DF
    tmp["_model_pos"] = tmp[model_col].map(first_pos)

    # Sort: group -> N -> first appearance of model -> original row order.
    # Using mergesort to preserve stability for setups within a model.
    tmp = tmp.sort_values(
        by=["_group", "_n", "_model_pos", "_row_pos"], kind="mergesort"
    )

    # Restore original structure
    tmp = tmp.drop(columns=["_group", "_n", "_model_pos", "_row_pos"])
    return tmp.set_index([model_col, setup_col])


def _dict_to_benchmark_df(
    results: Dict[str, Dict[Model, Dict[str, float]]], add_std: bool, csv_path=None
):
    """
    Convert a results dict with keys
        {benchmark: {(model_name, setup): {'mean': x, 'std': y}}}
    into a pandas DataFrame with a **MultiIndex** on the rows
    (level-0 = model_name, level-1 = setup).

    Parameters
    ----------
    results_dict : dict
        Benchmark results as described above.
    csv_path : str, optional
        If given, the DataFrame is saved to this location.

    Returns
    -------
    pd.DataFrame
        Rows -> (Model, Setup) multi-index
        Columns -> benchmarks
        Values  -> "mean ± std" (two-decimal precision)
    """
    data: Dict[Tuple[str, str], Dict[str, str]] = {}
    for bench, model_dict in results.items():
        for model, stats in model_dict.items():
            mname, setup = model.name, model.label
            data.setdefault((mname, setup), {})
            mean = stats.get("mean", float("nan"))
            std = stats.get("std", float("nan"))
            if add_std:
                if np.isnan(std) or np.isnan(mean):
                    data[(mname, setup)][bench] = ""
                else:
                    data[(mname, setup)][bench] = f"{mean:.1f} $\pm$ {std:.1f}"
            else:
                if np.isnan(mean):
                    data[(mname, setup)][bench] = ""
                else:
                    data[(mname, setup)][bench] = f"{mean:.1f}"

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Model", "Setup"])
    df = df.reindex(sorted(df.columns), axis=1)  # column order = alphabetical

    df = sort_models(df)

    rename = {
        "AIME2025 AVG@32": "AIME25 AVG@32",
        "AIME_2024 AVG@32": "AIME24 AVG@32",
        "AIME2025 PASS@1": "AIME25 PASS@1",
        "AIME_2024 PASS@1": "AIME24 PASS@1",
        "AMC-23 AVG@32": "AMC23 AVG@32",
        "AMC-23 PASS@1": "AMC23 PASS@1",
        "MATH-500": "MATH500",
        "Minerva-Math": "MinervaMath",
        "OlympiadBench": "OlympiadBench",
        "Average": "Avg.",
    }

    if not add_std and "Average" in df.columns:
        # df_with_rel = relative_to_full_tune(df.astype(float)).round(1).astype(str)
        df_with_rel = (
            df.replace("", np.nan)  # original DataFrame  # 1️⃣ treat "" as missing
            .astype(float)  # 2️⃣ safe cast to float
            .pipe(relative_to_full_tune)  # 3️⃣ your function
            .round(1)  # 4️⃣ round
            .fillna("")  # 5️⃣ turn NaNs back to ""
            .astype(str)  # 6️⃣ final string dtype
        )
        df = df.rename(columns=rename)
        df_with_rel.to_csv(csv_path.replace(".csv", "_with_rel.csv"))

    df = df.rename(columns=rename)
    df.to_csv(csv_path)

    return df


def dict_to_benchmark_df(
    results: Dict[str, Dict[Model, Dict[str, float]]], csv_path=None
):
    _dict_to_benchmark_df(
        results=results, add_std=True, csv_path=csv_path.replace(".csv", "_std.csv")
    )
    _dict_to_benchmark_df(results=results, add_std=False, csv_path=csv_path)


def plot_benchmark_curves_avg_paper(
    xs_plain,  # unchanged
    ys_plain,  # unchanged
    ys_std_plain,  # unchanged
    other_models,  # unchanged
    baseline_means: dict,  # NEW - flat {Model: float}
    baseline_stds: dict,  # NEW - flat {Model: float}
    xlabel: str,
    savedir,
):
    """Minimal figure for the paper: only plain steering-layer + baseline lines."""
    with mpl.rc_context(
        {
            "figure.figsize": (9, 3.5),
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

        # Orange steering-layer curve
        ax.plot(
            xs_plain, ys_plain, marker="o", linewidth=2.5, color="#F28E2B", zorder=3
        )
        plt.fill_between(
            xs_plain,
            np.array(ys_plain) - np.array(ys_std_plain),
            np.array(ys_plain) + np.array(ys_std_plain),
            alpha=0.2,
            color="#F28E2B",
            zorder=2,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Acc.")

        # Baseline h-lines + labels (only if present in baseline_means)
        for idx, other_model in enumerate(other_models[:3]):
            yref = baseline_means.get(other_model)
            yref_std = baseline_stds.get(other_model)
            if yref is None:
                continue  # skip baselines we don’t have
            ax.axhline(yref, linestyle="--", linewidth=2, color="black", zorder=1)
            if yref_std != 0:
                ax.fill_between(
                    xs_plain,
                    yref - yref_std,
                    yref + yref_std,
                    alpha=0.1,
                    color="black",
                    zorder=0,
                )
            ax.annotate(
                other_model.label,
                xy=(0.995, yref),
                xycoords=("axes fraction", "data"),
                xytext=(0, 6),
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

        # Add 5 % headroom so top label never clips
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.05)

        fig.tight_layout()
        (Path(savedir) / "paper-average-across-benches.pdf").parent.mkdir(
            parents=True, exist_ok=True
        )
        fig.savefig(
            Path(savedir) / "paper-average-across-benches.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_benchmark_curves(
    data: Dict[str, Dict[Model, Dict[str, float]]],
    savedir: str | Path,
    other_models: List[Model],
    regex_pattern: str,
    xlabel: str,
    file_format: str = "png",
) -> None:
    """
    Plot (i) per-benchmark steering-layer curves and (ii) an overall
    “Average across benches” curve that is **already supplied**
    in ``data["Average"]``.  Baselines are read directly from that entry.
    """
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    def slugify(text: str) -> str:
        text = re.sub(r"\s+", "-", text.strip().lower())
        return re.sub(r"[^a-z0-9\-_]+", "", text)

    # ---------------------------- per-benchmark plots --------------------- #
    for bench, results in data.items():
        if bench == "Average":
            continue  # handled later

        layer_pts: list[tuple[int, float, float, bool]] = []
        for model, stats in results.items():
            m = re.match(regex_pattern, model.name)
            if not m:
                continue
            n = float(m.group(1))
            onto = "on to" in model.name
            layer_pts.append((n, stats["mean"], stats.get("std", 0.0), onto))

        if not layer_pts:
            continue

        layer_pts.sort(key=lambda t: t[0])
        xs, ys, stds, ontos = zip(*layer_pts)

        plt.figure()

        # plain steering-layer
        xs_p = [x for x, o in zip(xs, ontos) if not o]
        ys_p = [y for y, o in zip(ys, ontos) if not o]
        std_p = [s for s, o in zip(stds, ontos) if not o]
        plt.plot(xs_p, ys_p, marker="o", label="steering-layer", zorder=3)
        plt.fill_between(
            xs_p,
            np.array(ys_p) - np.array(std_p),
            np.array(ys_p) + np.array(std_p),
            alpha=0.2,
            zorder=2,
        )

        # append-to steering-layer
        xs_o = [x for x, o in zip(xs, ontos) if o]
        ys_o = [y for y, o in zip(ys, ontos) if o]
        std_o = [s for s, o in zip(stds, ontos) if o]
        if xs_o:
            plt.plot(xs_o, ys_o, marker="s", label="steering-layer append to", zorder=3)
            plt.fill_between(
                xs_o,
                np.array(ys_o) - np.array(std_o),
                np.array(ys_o) + np.array(std_o),
                alpha=0.2,
                zorder=2,
            )

        # baseline h-lines
        for idx, other_model in enumerate(other_models):
            if other_model in results:
                plt.axhline(
                    results[other_model]["mean"],
                    color=COLORS[idx],
                    ls="--",
                    lw=1,
                    label=other_model.label,
                )

        plt.title(f"{bench} performance")
        plt.xlabel("Layer")
        plt.ylabel("Mean Acc.")
        plt.legend()
        plt.tight_layout()

        plt.savefig(savedir / f"{slugify(bench)}.{file_format}", format=file_format)
        plt.close()

    # ---------------------- “Average across benches” plot ----------------- #
    if "Average" not in data:
        raise KeyError('Expected an "Average" entry in `data`.')

    avg_results = data["Average"]

    # Build steering-layer points
    layer_pts_avg: list[tuple[int, float, float, bool]] = []
    for model, stats in avg_results.items():
        m = re.match(regex_pattern, model.name)
        if not m:
            continue
        n = float(m.group(1))
        onto = "on to" in model.name
        layer_pts_avg.append((n, stats["mean"], stats["std"], onto))

    if layer_pts_avg:
        layer_pts_avg.sort(key=lambda t: t[0])
        xs_avg, ys_avg, ys_std, ontos_avg = zip(*layer_pts_avg)

        xs_plain = [x for x, o in zip(xs_avg, ontos_avg) if not o]
        ys_plain = [y for y, o in zip(ys_avg, ontos_avg) if not o]
        ys_std_plain = [y for y, o in zip(ys_std, ontos_avg) if not o]

        plt.figure()

        # average curves
        if xs_plain:
            plt.plot(xs_plain, ys_plain, marker="o", label="steering-layer (Average)")
            plt.fill_between(
                xs_plain,
                np.array(ys_plain) - np.array(ys_std_plain),
                np.array(ys_plain) + np.array(ys_std_plain),
                alpha=0.2,
            )

        xs_onto = [x for x, o in zip(xs_avg, ontos_avg) if o]
        ys_onto = [y for y, o in zip(ys_avg, ontos_avg) if o]
        ys_std_onto = [y for y, o in zip(ys_std, ontos_avg) if o]
        if xs_onto:
            plt.plot(
                xs_onto, ys_onto, marker="s", label="steering-layer append to (Average)"
            )
            plt.fill_between(
                xs_onto,
                np.array(ys_onto) - np.array(ys_std_onto),
                np.array(ys_onto) + np.array(ys_std_onto),
                alpha=0.2,
            )

        # baseline average lines
        for idx, other_model in enumerate(other_models):
            if other_model in avg_results:
                plt.axhline(
                    avg_results[other_model]["mean"],
                    color=COLORS[idx],
                    ls="--",
                    lw=1,
                    label=f"{other_model.label} (Average)",
                )

        plt.title("Average performance across benches")
        plt.xlabel("Layer")
        plt.ylabel("Mean Acc.")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            savedir / f"average-across-benches.{file_format}", format=file_format
        )
        plt.close()

        # ---------------- paper-style simplified figure ------------------- #
        baseline_means = {
            m: avg_results[m]["mean"] for m in other_models if m in avg_results
        }
        baseline_stds = {
            m: avg_results[m]["std"] for m in other_models if m in avg_results
        }
        plot_benchmark_curves_avg_paper(
            xs_plain=xs_plain,
            ys_plain=ys_plain,
            ys_std_plain=ys_std_plain,
            other_models=other_models,
            baseline_means=baseline_means,  # NEW simpler arg
            baseline_stds=baseline_stds,  # NEW simpler arg
            xlabel=xlabel,
            savedir=savedir,
        )


def plot_benchmark_panels(
    data: Dict[str, Dict[Model, Dict[str, float]]],
    savedir: str | Path,
    other_models: List[Model],
    regex_pattern: str,
    *,
    file_format: str = "png",
    max_cols: int = 3,
    panel_size: tuple[int, int] = (4, 3),
) -> None:
    """
    Draw one multi-panel figure: every benchmark **plus** a final panel that
    shows the *Average across benches* supplied in ``data["Average"]``.
    """
    if "Average" not in data:
        raise KeyError('Expected an "Average" entry in `data`.')

    # ----------------------- bookkeeping & layout ------------------------- #
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # All real benchmarks (exclude the synthetic "Average")
    benches = [b for b in data if b != "Average"]
    n_benchmarks = len(benches)
    n_total_panels = n_benchmarks + 1  # +1 for the Average panel

    ncols = min(max_cols, n_total_panels)
    nrows = math.ceil(n_total_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()

    # ----------------------------- per-bench ------------------------------ #
    for ax, bench in zip(axes, benches):
        results = data[bench]

        # collect steering-layer points
        layer_pts: list[tuple[int, float, float, bool]] = []
        for model, stats in results.items():
            m = re.match(regex_pattern, model.name)
            if not m:
                continue
            n = float(m.group(1))
            onto = "on to" in model.name
            layer_pts.append((n, stats["mean"], stats.get("std", 0.0), onto))

        if not layer_pts:
            ax.set_visible(False)
            continue

        layer_pts.sort(key=lambda t: t[0])
        xs, ys, stds, ontos = zip(*layer_pts)
        xs, ys, stds = np.asarray(xs), np.asarray(ys), np.asarray(stds)  # type: ignore

        # plain steering-layer
        mask_plain = ~np.asarray(ontos, dtype=bool)
        ax.plot(xs[mask_plain], ys[mask_plain], marker="o", label="steering-layer")
        ax.fill_between(
            xs[mask_plain],
            ys[mask_plain] - stds[mask_plain],
            ys[mask_plain] + stds[mask_plain],
            alpha=0.2,
        )

        # append-to steering-layer
        mask_onto = np.asarray(ontos, dtype=bool)
        if mask_onto.any():
            ax.plot(
                xs[mask_onto],
                ys[mask_onto],
                marker="s",
                label="steering-layer append to",
            )
            ax.fill_between(
                xs[mask_onto],
                ys[mask_onto] - stds[mask_onto],
                ys[mask_onto] + stds[mask_onto],
                alpha=0.2,
            )

        # baseline reference lines
        for idx, other_model in enumerate(other_models):
            if other_model in results:
                ax.axhline(
                    results[other_model]["mean"],
                    color=COLORS[idx],
                    ls="--",
                    lw=1,
                    label=other_model.label,
                )

        ax.set_title(bench)
        ax.set_xlabel("Layer N")
        ax.set_ylabel("Mean score")
        ax.legend(fontsize="small")

    # -------------------------- Average panel ----------------------------- #
    avg_ax = axes[n_benchmarks]
    avg_results = data["Average"]

    # steering-layer curves
    avg_layer_pts: list[tuple[int, float, float, bool]] = []
    for model, stats in avg_results.items():
        m = re.match(regex_pattern, model.name)
        if not m:
            continue
        n = float(m.group(1))
        onto = "on to" in model.name
        avg_layer_pts.append((n, stats["mean"], stats.get("std", 0.0), onto))

    if avg_layer_pts:
        avg_layer_pts.sort(key=lambda t: t[0])
        xs, ys, stds, ontos = zip(*avg_layer_pts)
        xs, ys, stds = np.asarray(xs), np.asarray(ys), np.asarray(stds)  # type: ignore

        mask_plain = ~np.asarray(ontos, dtype=bool)
        avg_ax.plot(
            xs[mask_plain], ys[mask_plain], marker="o", label="Avg steering-layer"
        )

        mask_onto = np.asarray(ontos, dtype=bool)
        if mask_onto.any():
            avg_ax.plot(
                xs[mask_onto],
                ys[mask_onto],
                marker="s",
                label="Avg steering-layer append to",
            )

        # baseline averages (already averaged by provider of `data`)
        for idx, other_model in enumerate(other_models):
            if other_model in avg_results:
                avg_ax.axhline(
                    avg_results[other_model]["mean"],
                    color=COLORS[idx],
                    ls="--",
                    lw=1,
                    label=other_model.label,
                )

        avg_ax.set_title("Average across benchmarks")
        avg_ax.set_xlabel("Layer N")
        avg_ax.set_ylabel("Mean score")
        avg_ax.legend(fontsize="small")
    else:
        avg_ax.set_visible(False)

    # --------------------------- tidy-up ---------------------------------- #
    for ax in axes[n_total_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    out_path = Path(savedir) / f"all_benchmarks.{file_format}"
    fig.savefig(out_path, format=file_format, dpi=150)
    plt.close(fig)


def name_to_label(
    benches: Dict[str, Dict[Model, Dict[str, float]]], setup_mapping: SetupMapping
) -> Dict[str, Dict[Model, Dict[str, float]]]:
    new_benches = {}
    for bench, values in benches.items():
        new_dict = {}
        for key, value in list(values.items()):
            if "|" in key.name:
                key = Model(
                    label=setup_mapping[key.name.split("|")[1]],
                    name=key.name.split("|")[0],
                    pkl_path=key.pkl_path,
                    ckpt=key.ckpt,
                )
            new_dict[key] = value

        new_benches[bench] = new_dict

    return new_benches


def filter_benchmarks_for_models(
    data: Dict[str, Dict[Model, Dict[str, float]]],
    selected_models: List[Model],
) -> Dict[str, Dict[Model, Dict[str, float]]]:
    """
    Drop any model not in `selected_models`; drop entire benchmarks that
    would become empty after that.
    """
    keep: set[Model] = set(selected_models)
    filtered: Dict[str, Dict[Model, Dict[str, float]]] = {}

    for bench, results in data.items():
        subset = {m: stats for m, stats in results.items() if m in keep}
        if subset:
            filtered[bench] = subset

    return filtered
