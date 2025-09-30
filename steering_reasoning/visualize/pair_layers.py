import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import seaborn as sns
from loguru import logger
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.common import (
    plot_benchmark_curves,
    plot_benchmark_panels,
)
from steering_reasoning.visualize.utils.defs import Model, ModelGroup
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    seed: int

    loaddir: str
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def get_qwen2_5_math_7b_model_groups_pair_layer() -> Tuple[
    List[ModelGroup], List[Model]
]:
    models = []
    for i in range(28):
        for j in range(i, 28):
            pkl_path = f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-{i}-{j}/append_to_False/temp_1.0_top_p_1.0"
            if os.path.exists(pkl_path):
                models.append(
                    Model(
                        label=f"steering-layer-{i}-{j}",
                        pkl_path=pkl_path,
                        name=f"steering-layer-{i}-{j}",
                        ckpt=None,
                    )
                )

    model_groups = [ModelGroup("Qwen2.5-Math-7B. Pair Layers", models=models)]

    return model_groups, []


def get_llama3_1_8b_chat_model_groups_pair_layer() -> Tuple[
    List[ModelGroup], List[Model]
]:
    models = []
    for i in range(32):
        for j in range(i, 32):
            pkl_path = f"/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-{i}-{j}/temp_1.0_top_p_1.0"
            if os.path.exists(pkl_path):
                models.append(
                    Model(
                        label=f"steering-layer-{i}-{j}",
                        pkl_path=pkl_path,
                        name=f"steering-layer-{i}-{j}",
                        ckpt=None,
                    )
                )

    model_groups = [ModelGroup("LLaMa3.1-8B-It. Pair Layers", models=models)]

    return model_groups, []


def _parse_ij(label: str) -> Tuple[int, int]:
    """Extract (i, j) from a label of the form ``steering-layer-{i}-{j}``.

    Raises
    ------
    ValueError
        If the label does not match the expected pattern.
    """
    _LABEL_RE = re.compile(r"steering-layer-(\d+)-(\d+)")

    m = _LABEL_RE.fullmatch(label)
    if not m:
        raise ValueError(
            f"Label does not match 'steering-layer-{{i}}-{{j}}': {label!r}"
        )
    return int(m.group(1)), int(m.group(2))


def _sanitize_filename(name: str) -> str:
    """Turn an arbitrary benchmark name into a safe filename with .png suffix."""

    # collapse spaces and unsafe chars to underscore
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_ ")
    if not base:
        base = "benchmark"
    return base


def plot_steering_layer_benchmarks(
    results: Dict[str, Dict[Model, Dict[str, float]]],
    savedir: Path | str,
    steering_perf: float,
    *,
    annotate_std: bool = False,
    cmap: str = "dark:salmon",
    dpi: int = 200,
) -> None:
    """Plot *N×N* heat‑maps for every benchmark plus their pre‑computed **Average**.

    The caller is responsible for inserting an *"Average"* entry inside ``results``
    that already contains the cell‑wise mean across benchmarks.

    Parameters
    ----------
    results
        ``benchmark_name → {model → {"mean": float, "std": float}}``.
    savedir
        Output directory (created if needed).
    annotate_std
        Write ``±std`` beneath each mean when available.
    cmap, dpi
        Forwarded to Matplotlib.
    """

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build grids once
    # ------------------------------------------------------------------
    N = _infer_grid_size(results)
    grids = {name: _build_grid(stats, N) for name, stats in results.items()}

    # ------------------------------------------------------------------
    # Per‑benchmark plots (includes the provided "Average")
    # ------------------------------------------------------------------

    cmap_obj = sns.color_palette("YlOrBr", as_cmap=True).copy()
    # cmap_obj_sliced = LinearSegmentedColormap.from_list(
    #     "OrBr", cmap_obj(np.linspace(0.5, 1, 256))
    # )
    cmap_obj_sliced = LinearSegmentedColormap.from_list(
        "YlOr", cmap_obj(np.linspace(0.0, 0.5, 256))
    )

    for bench, grid in grids.items():
        _plot_grid(
            grid,
            title="Average across benchmarks" if bench == "Average" else bench,
            filename=_sanitize_filename(bench),
            savedir=savedir,
            stats_map=results[bench],
            annotate_std=annotate_std,
            annotate_color="black",
            cmap=cmap_obj_sliced,
            dpi=dpi,
            N=N,
        )
        os.makedirs(os.path.join(savedir, "pkls"), exist_ok=True)
        np.save(os.path.join(savedir, "pkls", f"{bench}.npy"), grid)

    # ------------------------------------------------------------------
    # Relative view (average grid normalised by its diagonal)
    # ------------------------------------------------------------------
    if "Average" in grids:
        avg = grids["Average"].copy()
        diag = np.diag(avg)

        # fabricate a stats map from the relative grid so we can reuse _annotate
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_improvement_prev = (avg - diag) / (steering_perf - diag)
            rel_improvement_next = (avg - diag[:, None]) / (
                steering_perf - diag[:, None]
            )

            pair_max = np.maximum(diag[:, None], diag[None, :])
            rel_improvement_max = (avg - pair_max) / (steering_perf - pair_max)

        rel_improvement_prev_stats: Dict[Model, Dict[str, float]] = {}
        rel_improvement_next_stats: Dict[Model, Dict[str, float]] = {}
        rel_improvement_max_stats: Dict[Model, Dict[str, float]] = {}
        for i in range(N):
            for j in range(N):
                val_prev = rel_improvement_prev[i, j]
                val_next = rel_improvement_next[i, j]
                val_max = rel_improvement_max[i, j]
                if not math.isnan(val_prev):
                    rel_improvement_prev_stats[
                        Model(
                            label=f"steering-layer-{i}-{j}",
                            pkl_path=None,
                            name="",
                            ckpt=None,
                        )
                    ] = {"mean": float(val_prev)}
                if not math.isnan(val_next):
                    rel_improvement_next_stats[
                        Model(
                            f"steering-layer-{i}-{j}",
                            pkl_path=None,
                            name="",
                            ckpt=None,
                        )
                    ] = {"mean": float(val_next)}
                if not math.isnan(val_max):
                    rel_improvement_max_stats[
                        Model(
                            f"steering-layer-{i}-{j}",
                            pkl_path=None,
                            name="",
                            ckpt=None,
                        )
                    ] = {"mean": float(val_max)}

        # cmap_obj = sns.color_palette(cmap, as_cmap=True).copy()
        # cmap_obj = LinearSegmentedColormap.from_list(
        #     "black_to_salmon",  # name
        #     ["#000000", "#F28E2B"],  # start  →  end
        # )
        cmap_obj = LinearSegmentedColormap.from_list(
            "blue_white_orange",
            [(0.0, "#56B4E9"), (0.5, "white"), (1.0, "#F28E2B")],
        )
        # cmap_obj.set_bad(color="white", alpha=0)
        _plot_grid_paper(
            rel_improvement_prev,
            title="Relative Improvement average (normalised by diagonal)",
            filename=_sanitize_filename(
                "relative_improvement_average_across_benchmarks_prev"
            ),
            norm=TwoSlopeNorm(
                vmin=np.nanmin(rel_improvement_prev),
                vcenter=0.0,
                vmax=1,
            ),
            savedir=savedir,
            # stats_map=rel_improvement_prev_stats,  # annotate with numbers, no std
            stats_map=None,  # annotate with numbers, no std
            annotate_std=False,
            cmap=cmap_obj,
            dpi=dpi,
            N=N,
        )
        _plot_grid_paper(
            rel_improvement_next,
            title="Relative Improvement average (normalised by diagonal)",
            filename=_sanitize_filename(
                "relative_improvement_average_across_benchmarks_next"
            ),
            norm=TwoSlopeNorm(
                vmin=np.nanmin(rel_improvement_next),
                vcenter=0.0,
                vmax=1,
            ),
            savedir=savedir,
            # stats_map=rel_improvement_next_stats,  # annotate with numbers, no std
            stats_map=None,  # annotate with numbers, no std
            annotate_std=False,
            cmap=cmap_obj,
            dpi=dpi,
            N=N,
        )
        _plot_grid_paper(
            rel_improvement_max,
            title="Relative Improvement average (normalised by diagonal)",
            filename=_sanitize_filename(
                "relative_improvement_average_across_benchmarks_max"
            ),
            norm=TwoSlopeNorm(
                vmin=np.nanmin(rel_improvement_max),
                vcenter=0.0,
                vmax=1,
            ),
            savedir=savedir,
            # stats_map=rel_improvement_max_stats,  # annotate with numbers, no std
            stats_map=None,  # annotate with numbers, no std
            annotate_std=False,
            cmap=cmap_obj,
            dpi=dpi,
            N=N,
        )

        # _plot_grid_paper_demo(
        #     rel_improvement_masked,
        #     title="Demo grid",
        #     filename="demo_grid",
        #     savedir=savedir,
        #     stats_map=None,
        #     annotate_std=False,
        #     cmap=cmap_obj,
        #     dpi=300,
        #     N=N,
        #     norm=Normalize(vmin=0, vmax=1),
        # )


def add_x_on_quadmesh(mesh, mask, frac=0.5, **kw):
    """
    Draw X marks centered in cells where mask==True.
    Works for both shading='flat' (edges) and 'nearest' (centers).
    frac ≤ 1 controls size; 0.5 = half the cell.
    """
    if isinstance(mesh, QuadMesh):
        pc = mesh
    else:  # assume Axes
        qms = [c for c in mesh.collections if isinstance(c, QuadMesh)]
        if not qms:
            raise ValueError("No QuadMesh found on the provided Axes.")
        pc = qms[-1]
    coords = pc._coordinates  # (My,Nx,2) : private but stable
    My, Nx, _ = coords.shape
    M, N = mask.shape

    if (My, Nx) == (M + 1, N + 1):  # edges provided
        xe = coords[0, :, 0]
        ye = coords[:, 0, 1]
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        w = xe[1:] - xe[:-1]
        h = ye[1:] - ye[:-1]
    elif (My, Nx) == (M, N):  # centers provided (nearest)
        xc = coords[0, :, 0]
        yc = coords[:, 0, 1]
        dx = np.diff(xc)
        dy = np.diff(yc)
        w = np.r_[dx[0], 0.5 * (dx[:-1] + dx[1:]), dx[-1]]
        h = np.r_[dy[0], 0.5 * (dy[:-1] + dy[1:]), dy[-1]]
    else:
        raise ValueError("mask shape doesn't match QuadMesh")

    rr, cc = np.where(mask)
    segs = []
    for r, c in zip(rr, cc):
        L = 0.5 * frac * min(w[c], h[r])
        x, y = xc[c], yc[r]
        segs += [[(x - L, y - L), (x + L, y + L)], [(x - L, y + L), (x + L, y - L)]]

    lc = LineCollection(segs, **kw)
    pc.axes.add_collection(lc)
    return lc


def _plot_grid_paper(
    grid: np.ndarray,
    *,
    title: str,
    filename: str,
    savedir: Path,
    stats_map: Optional[Dict[Model, Dict[str, float]]],
    annotate_std: bool,
    cmap,
    dpi: int,
    N: int,
    norm=None,
    tick_step: int = 2,
    show_title: bool = False,
    cbar_size: str = "3%",  # width of the colour‑bar (e.g. "3%" or 0.15 in)
    cbar_pad: float = 0.02,
    marker_threshold: float = 0.99,
) -> None:
    """Draw a square heat‑map ready for journal publication.

    The function keeps the caller‑provided *cmap* intact but tightens
    layout, moves the colour‑bar below the plot (default), and hides the
    redundant lower‑left triangle if *grid* is a masked array.  Tick
    density can be controlled with *tick_step*.
    """

    # --- Matplotlib context --------------------------------------------------
    with mpl.rc_context(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.compression": 0,
            "savefig.dpi": dpi,
        }
    ):
        # Figure size - clamp to single‑column width (≈ 85 mm)
        fig_side = max(3.35, N * 0.25)  # inches
        fig, ax = plt.subplots(figsize=(fig_side, fig_side), constrained_layout=False)

        # --------------------------- heat‑map ---------------------------------
        x_edges = np.arange(grid.shape[1] + 1) - 0.5
        y_edges = np.arange(grid.shape[0] + 1) - 0.5

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            cmap=cmap,
            norm=norm,
            shading="flat",
            antialiased=False,
        )
        ax.set_aspect("equal")
        ax.set(
            xlabel="Layer",
            ylabel="Layer",
            xticks=np.arange(0, N, tick_step),
            yticks=np.arange(0, N, tick_step),
            xlim=(-0.5, N - 0.5),
            ylim=(N - 0.5, -0.5),
        )
        # high-contrast double stroke that still fits inside each cell
        add_x_on_quadmesh(
            mesh,
            mask=grid > marker_threshold,
            frac=0.5,
            colors="white",
            linewidths=4,
            zorder=5,
        )
        add_x_on_quadmesh(
            mesh,
            mask=grid > marker_threshold,
            frac=0.5,
            colors="black",
            linewidths=2,
            zorder=6,
        )
        ax.set_xticklabels([str(t) for t in range(0, N, tick_step)])
        ax.set_yticklabels([str(t) for t in range(0, N, tick_step)])

        if show_title:
            ax.set_title(title, pad=6)

        # ------------------------ colour‑bar ---------------------------------
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        # Always make the colour‑bar *vertical* and full‑height
        cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
        # cbar.set_label("Relative Improvement", fontsize=14)
        # cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Relative Improvement")
        # cbar.set_ticks(
        #     [
        #         -np.nanmin(grid),
        #         -np.nanmin(grid) / 2,
        #         0,
        #         np.nanmax(grid) / 2,
        #         np.nanmax(grid),
        #     ]
        # )
        # cbar.set_ticklabels(
        #     [
        #         f"{t:g}"
        #         for t in [
        #             -np.nanmin(grid),
        #             -np.nanmin(grid) / 2,
        #             0,
        #             np.nanmax(grid) / 2,
        #             np.nanmax(grid),
        #         ]
        #     ]
        # )

        # ---------------------- cell annotations -----------------------------
        if stats_map is not None:
            for (i, j), cell_val in np.ndenumerate(grid):
                if np.ma.is_masked(cell_val):
                    continue
                mean_val = stats_map.get(
                    Model(
                        label=f"steering-layer-{i}-{j}",
                        pkl_path=None,
                        name="",
                        ckpt=None,
                    ),
                    {},
                ).get("mean", np.nan)
                if np.isnan(mean_val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black" if cell_val < 0.5 else "white",
                )

        # ----------------------------- save ----------------------------------
        savepath = savedir / f"{filename}.pdf"
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _infer_grid_size(results: Dict[str, Dict[Model, Dict[str, float]]]) -> int:
    """Return *N* such that all steering-layer indices ``i,j < N``."""

    N = 0
    for stats_map in results.values():
        for model in stats_map:
            try:
                i, j = _parse_ij(model.label)
            except ValueError:  # skip non‑steering‑layer labels
                continue
            N = max(N, i + 1, j + 1)
    if N == 0:
        raise ValueError("No valid 'steering-layer-i-j' labels found.")
    logger.info(f"global_N={N}")
    return N


def _build_grid(stats_map: Dict[Model, Dict[str, float]], N: int) -> np.ndarray:
    grid = np.full((N, N), np.nan, dtype=float)
    for model, stats in stats_map.items():
        try:
            i, j = _parse_ij(model.label)
        except ValueError:
            continue
        grid[i, j] = float(stats.get("mean", np.nan))
    return grid


def _plot_grid(
    grid: np.ndarray,
    *,
    title: str,
    filename: str,
    savedir: Path,
    stats_map: Optional[Dict[Model, Dict[str, float]]],
    annotate_std: bool,
    cmap: str,
    dpi: int,
    N: int,
    norm=None,
    annotate_color: str = "black",
) -> None:
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
            "pdf.compression": 0,
            "savefig.dpi": dpi,
        }
    ):
        fig, ax = plt.subplots(figsize=(max(4.0, N * 0.6), max(3.6, N * 0.6)))
        x = np.arange(grid.shape[1] + 1) - 0.5  # columns - 0.5 … N-0.5
        y = np.arange(grid.shape[0] + 1) - 0.5  # rows    - 0.5 … N-0.5

        mesh = ax.pcolormesh(
            x,
            y,
            grid,
            cmap=cmap,
            shading="flat",  # no interpolation between cells
            antialiased=False,
        )
        ax.set_aspect("equal")

        if stats_map is not None:
            _annotate(ax, stats_map, annotate_std, annotate_color=annotate_color)

        ax.set(
            xlabel="j",
            ylabel="i",
            xticks=range(N),
            yticks=range(N),
            xlim=(-0.5, N - 0.5),
            ylim=(N - 0.5, -0.5),
        )
        ax.set_xticklabels(map(str, range(N)))
        ax.set_yticklabels(map(str, range(N)))
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="mean")
        fig.tight_layout()
        fig.savefig(
            savedir / f"{filename}.pdf", dpi=dpi, format="pdf", bbox_inches="tight"
        )
        plt.close(fig)


def _annotate(
    ax: plt.Axes,
    stats_map: Dict[Model, Dict[str, float]],
    annotate_std: bool,
    annotate_color: str,
) -> None:
    """Write mean (and optionally std) inside *ax* grid cells."""

    for model, stats in stats_map.items():
        try:
            i, j = _parse_ij(model.label)
        except ValueError:
            continue
        m = float(stats.get("mean", math.nan))
        if math.isnan(m):
            continue
        s = stats.get("std")
        txt = (
            f"{m:.3f}"
            if not (annotate_std and s is not None and not math.isnan(s))
            else f"{m:.3f}\n±{float(s):.3f}"
        )
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=annotate_color)


@pyrallis.wrap()
def run(config: Config):
    init_savedir = os.path.join(config.savedir, "pair-layers")

    benches = [
        "AIME2025",
        "AIME_2024",
        "AMC-23",
        "MATH-500",
        "Minerva-Math",
        "OlympiadBench",
    ]

    for model_groups_name, (model_groups, other_models), steering_perf in zip(
        ["Qwen2.5-Math-7B", "llama3.1-8b-chat"],
        [
            get_qwen2_5_math_7b_model_groups_pair_layer(),
            get_llama3_1_8b_chat_model_groups_pair_layer(),
        ],
        [42.9, 25.8],
    ):
        config.savedir = os.path.join(init_savedir, model_groups_name)
        os.makedirs(config.savedir, exist_ok=True)
        # Final plots and tables
        accuracies_dict, avg_accuracies_dict, _, _, _, _ = run_for_model_groups(
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

        savedir = os.path.join(config.savedir, "best-ckpts")
        os.makedirs(savedir, exist_ok=True)
        filtered_accuracies = {}
        for bench, values in accuracies_dict.items():
            inner_dict = {}
            for model, inner_values in values.items():
                if "steering-layer-" in model.label and int(
                    model.label.rsplit("-")[-1]
                ) == int(model.label.rsplit("-")[-2]):
                    inner_dict[model] = inner_values

            filtered_accuracies[bench] = inner_dict

        plot_benchmark_curves(
            data=filtered_accuracies,
            savedir=os.path.join(savedir, "accuracies"),
            other_models=other_models,
            regex_pattern=r"^steering-layer-(\d+)",
            xlabel="Layer",
        )

        plot_benchmark_panels(
            data=filtered_accuracies,
            savedir=os.path.join(savedir, "accuracies"),
            other_models=other_models,
            regex_pattern=r"^steering-layer-(\d+)",
        )

        plot_steering_layer_benchmarks(
            results=accuracies_dict, savedir=savedir, steering_perf=steering_perf
        )


if __name__ == "__main__":
    run()
