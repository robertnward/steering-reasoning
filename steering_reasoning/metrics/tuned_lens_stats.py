import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from steering_reasoning.metrics.plots import (
    draw_pairwise_cossims,
    draw_pairwise_cossims_hist,
)
from steering_reasoning.visualize.seed_alignment import rowwise_cosine


@dataclass
class SVDReport:
    # --- core shape / norms ---
    shape: Tuple[int, int]
    rank: int
    spectral_norm: float  # ||A||_2 = σ_max
    smallest_singular_value: float  # σ_min
    frobenius_norm: float  # ||A||_F
    nuclear_norm: float  # ||A||_* = Σ σ_i
    norm_1: float  # ||A||_1 (max column sum)
    norm_inf: float  # ||A||_∞ (max row sum)

    # --- conditioning ---
    condition_number_2: float  # κ2  = σ_max / σ_min
    condition_number_1: Optional[float]  # κ1  = ||A||_1  ||A^{-1}||_1
    condition_number_inf: Optional[float]  # κ∞  = ||A||_∞  ||A^{-1}||_∞

    # --- effective dimensionality ---
    stable_rank: float  # sr  = ||A||_F^2 / ||A||_2^2
    entropy_effective_rank: float  # r_eff = exp( -Σ p_i log p_i ), p_i ∝ σ_i^2

    # --- determinant / volume (square only) ---
    log_abs_det: Optional[float]  # log |det A|
    det_sign: Optional[int]  # sign(det A) ∈ {-1,0,1}

    # --- scaling / anisotropy diagnostics ---
    row_l2_min: float
    row_l2_max: float
    row_l2_mean: float
    row_l2_cv: float
    col_l2_min: float
    col_l2_max: float
    col_l2_mean: float
    col_l2_cv: float

    # --- leverage / coherence diagnostics (compact SVD) ---
    row_leverage_max: float  # max_i ||U[i,:]||^2
    col_leverage_max: float  # max_j ||V[j,:]||^2
    row_coherence: float  # μ(U) = (m/r)*max leverage
    col_coherence: float  # μ(V) = (n/r)*max leverage

    # --- polar R diagnostics (square only) ---
    polar_R_orthogonality_err_fro: float  # ||RᵀR - I||_F
    polar_R_orthogonality_err_2: float  # ||RᵀR - I||_2
    polar_R_det: float  # det(R) = ±1
    polar_geodesic_distance: float  # ||log R||_F / sqrt(2)
    polar_eig_plus_one: int  # count of eigenvalues ≈ +1
    polar_eig_minus_one: int  # count of eigenvalues ≈ -1
    polar_angle_mean: float  # mean θ (radians; principal branch)
    polar_angle_std: float  # std of θ (radians)
    polar_angle_min: float  # min θ
    polar_angle_max: float  # max θ


def svd_analyze(
    A: np.ndarray,
    visualize: bool = True,
    random_map_samples: int = 400,
    title_prefix: str = "SVD Analysis",
    histogram_bins: int = 50,
    rng: Optional[np.random.Generator] = None,
    savedir: Optional[str] = None,  # figures go to os.path.join(savedir, "svd_stats")
    show_plots: bool = True,  # False = save only, don't display
    angle_hist_bins: int = 72,  # bins for eigen-angle histogram
    angle_tol: float = 1e-6,  # radians; classify ±1 via |θ| or |π- |θ|| < tol
    radius_tol: float = 1e-6,  # classify ±1 also requires ||λ|-1| < radius_tol
) -> Dict[str, Any]:
    """
    Full SVD + diagnostics + visualizations for a *square* (n x n) matrix A.
    ALWAYS computes the right polar decomposition A = R S and produces a polar plot.

    Visuals (when visualize=True), saved under {savedir}/svd_stats if `savedir` is set:
      01) Singular values (scree)
      01b) Log singular values
      02) Cumulative energy (σ²) w/ 90/95/99% cut lines
      03) Histogram of ||A x|| for random unit vectors x
      04) Condition numbers (bar chart)
      05) Row ℓ2 norms histogram
      06) Column ℓ2 norms histogram
      07) Row leverage scores histogram (‖U[i,:]‖²)
      08) Column leverage scores histogram (‖V[j,:]‖²)
      09) Eigenvalues of R on unit circle (polar plot)
      10) Histogram of eigen-angles θ = arg(λ) for R

    Returns:
      dict with SVD factors, polar factors (R,S), SVDReport, energy stats,
      energy thresholds (k for 90/95/99%), and saved figure paths.
    """
    A = np.asarray(A)
    m, n = A.shape
    assert m == n, (
        "svd_analyze expects a square matrix (n x n) to analyze R's spectrum."
    )

    # ---------- SVD ----------
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = s.size
    sigma_max = float(s[0]) if r else 0.0
    sigma_min = float(s[-1]) if r else 0.0

    # ---------- Polar decomposition (right): A = R S ----------
    R = U @ Vt  # orthogonal
    S = (Vt.T * s) @ Vt  # = V diag(s) Vᵀ (symmetric PD)

    # Orthogonality diagnostics
    RtR = R.T @ R
    I = np.eye(n)
    RtR_mI = RtR - I
    polar_R_orth_fro = float(np.linalg.norm(RtR_mI, "fro"))
    polar_R_orth_2 = float(np.linalg.norm(RtR_mI, 2))

    # det(R) = sign(det(A)) (since det(S) > 0)
    sign, _ = np.linalg.slogdet(A)
    polar_R_det = float(sign)

    # ---------- Core norms / ranks / conditioning ----------
    fro = float(np.linalg.norm(A, "fro"))
    nuc = float(np.sum(s))
    rank = int(np.linalg.matrix_rank(A))
    cond2 = float(np.inf) if sigma_min == 0 else float(sigma_max / sigma_min)
    norm1 = float(np.linalg.norm(A, 1))
    norminf = float(np.linalg.norm(A, np.inf))
    # other condition numbers
    try:
        cond1 = float(np.linalg.cond(A, 1))
    except np.linalg.LinAlgError:
        cond1 = float("inf")
    try:
        condinf = float(np.linalg.cond(A, np.inf))
    except np.linalg.LinAlgError:
        condinf = float("inf")

    # ---------- Energy & effective ranks ----------
    energy = s**2
    energy_total = float(np.sum(energy))
    if energy_total > 0.0:
        energy_ratio = energy / energy_total
        energy_cum = np.cumsum(energy_ratio)
        p = energy_ratio[energy_ratio > 0]  # avoid 0*log0
        H = float(-np.sum(p * np.log(p)))  # nat log
        r_eff = float(np.exp(H))
    else:
        energy_ratio = energy
        energy_cum = energy
        r_eff = 0.0
    stable_rank = float((fro**2) / (sigma_max**2)) if sigma_max > 0 else 0.0

    # Energy thresholds: k for 90/95/99%
    def k_for(thr: float) -> int:
        if energy_total == 0:
            return 0
        return int(np.searchsorted(energy_cum, thr, side="left") + 1)

    k90, k95, k99 = k_for(0.90), k_for(0.95), k_for(0.99)

    # ---------- Row/column scaling diagnostics ----------
    row_l2 = np.linalg.norm(A, axis=1)
    col_l2 = np.linalg.norm(A, axis=0)

    def stats(x: np.ndarray):
        mn = float(np.min(x)) if x.size else 0.0
        mx = float(np.max(x)) if x.size else 0.0
        mean = float(np.mean(x)) if x.size else 0.0
        cv = float(np.std(x) / mean) if (x.size and mean != 0.0) else 0.0
        return mn, mx, mean, cv

    row_min, row_max, row_mean, row_cv = stats(row_l2)
    col_min, col_max, col_mean, col_cv = stats(col_l2)

    # ---------- Leverage / coherence ----------
    row_lev = np.sum(U**2, axis=1) if U.size else np.array([])
    col_lev = np.sum((Vt.T) ** 2, axis=1) if Vt.size else np.array([])
    row_lev_max = float(np.max(row_lev)) if row_lev.size else 0.0
    col_lev_max = float(np.max(col_lev)) if col_lev.size else 0.0
    row_coh = float((m / r) * row_lev_max) if r > 0 else 0.0
    col_coh = float((n / r) * col_lev_max) if r > 0 else 0.0

    # ---------- R eigen-analysis (ALWAYS) ----------
    eig_R = np.linalg.eigvals(R)  # complex unit-modulus (± rounding)
    radii = np.abs(eig_R)
    angles = np.arctan2(eig_R.imag, eig_R.real)  # principal branch (-π, π]
    # counts of ±1 (need radius ~1 and angle ~ 0 or π)
    plus_one = int(
        np.sum((np.abs(radii - 1.0) <= radius_tol) & (np.abs(angles) <= angle_tol))
    )
    pi_dist = np.minimum(np.abs(angles - np.pi), np.abs(angles + np.pi))
    minus_one = int(
        np.sum((np.abs(radii - 1.0) <= radius_tol) & (pi_dist <= angle_tol))
    )
    # geodesic distance on O(n): ||log R||_F / sqrt(2)
    try:
        # principal matrix log via eigen-decomposition
        w, V = np.linalg.eig(R.astype(complex))
        L = np.diag(np.log(w))  # principal log of eigenvalues
        V_inv = np.linalg.inv(V)
        logR = V @ L @ V_inv
        polar_geo = float(np.linalg.norm(logR, "fro") / np.sqrt(2.0))
    except np.linalg.LinAlgError:
        polar_geo = float("nan")

    # angle summary
    ang_mean = float(np.mean(angles))
    ang_std = float(np.std(angles))
    ang_min = float(np.min(angles))
    ang_max = float(np.max(angles))

    # ---------- Determinant info ----------
    log_abs_det = None
    det_sign = None
    sign, logdet = np.linalg.slogdet(A)
    det_sign = int(sign)
    log_abs_det = float(logdet) if sign != 0 else float("-inf")

    # ---------- Build report ----------
    report = SVDReport(
        shape=(m, n),
        rank=rank,
        spectral_norm=sigma_max,
        smallest_singular_value=sigma_min,
        frobenius_norm=fro,
        nuclear_norm=nuc,
        norm_1=norm1,
        norm_inf=norminf,
        condition_number_2=cond2,
        condition_number_1=cond1,
        condition_number_inf=condinf,
        stable_rank=stable_rank,
        entropy_effective_rank=r_eff,
        log_abs_det=log_abs_det,
        det_sign=det_sign,
        row_l2_min=row_min,
        row_l2_max=row_max,
        row_l2_mean=row_mean,
        row_l2_cv=row_cv,
        col_l2_min=col_min,
        col_l2_max=col_max,
        col_l2_mean=col_mean,
        col_l2_cv=col_cv,
        row_leverage_max=row_lev_max,
        col_leverage_max=col_lev_max,
        row_coherence=row_coh,
        col_coherence=col_coh,
        polar_R_orthogonality_err_fro=polar_R_orth_fro,
        polar_R_orthogonality_err_2=polar_R_orth_2,
        polar_R_det=polar_R_det,
        polar_geodesic_distance=polar_geo,
        polar_eig_plus_one=plus_one,
        polar_eig_minus_one=minus_one,
        polar_angle_mean=ang_mean,
        polar_angle_std=ang_std,
        polar_angle_min=ang_min,
        polar_angle_max=ang_max,
    )

    results: Dict[str, Any] = {
        "U": U,
        "S": s,
        "Vt": Vt,
        "polar_R": R,
        "polar_S": S,
        "report": report,
        "energy_ratio": energy_ratio,
        "energy_cumulative": energy_cum,
        "energy_k_thresholds": {"k90": k90, "k95": k95, "k99": k99},
        "row_leverage": row_lev,
        "col_leverage": col_lev,
        "row_l2": row_l2,
        "col_l2": col_l2,
        "R_eigvals": eig_R,
        "R_angles": angles,
        "figure_paths": {},
    }

    # ---------- Save directory ----------
    save_root = None
    if savedir is not None:
        save_root = os.path.join(savedir)
        os.makedirs(save_root, exist_ok=True)

    if save_root is not None:
        with open(os.path.join(save_root, "report.json"), "+w") as f:
            json.dump(asdict(report), f, indent=4)

    def maybe_save(fig: plt.Figure, filename: str):
        if save_root is not None:
            path = os.path.join(save_root, filename)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            results["figure_paths"][filename] = path

    # ---------- Visualizations ----------
    if visualize:
        # 01) Singular values (scree)
        fig1 = plt.figure()
        plt.plot(np.arange(1, r + 1), s, marker="o")
        plt.title(f"{title_prefix}: Singular Values")
        plt.xlabel("Index")
        plt.ylabel("Singular value")
        plt.grid(True)
        maybe_save(fig1, "01_singular_values.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig1)

        # 01b) Log singular values
        fig1b = plt.figure()
        plt.plot(np.arange(1, r + 1), np.log(s + 1e-16), marker="o")
        plt.title(f"{title_prefix}: Log Singular Values")
        plt.xlabel("Index")
        plt.ylabel("log(σ)")
        plt.grid(True)
        maybe_save(fig1b, "01b_log_singular_values.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig1b)

        # 02) Cumulative energy with cut lines
        fig2 = plt.figure()
        plt.plot(np.arange(1, r + 1), energy_cum, marker="o")
        for thr, kk in [(0.90, k90), (0.95, k95), (0.99, k99)]:
            if kk > 0:
                plt.axvline(kk, linestyle="--", linewidth=1)
        plt.title(f"{title_prefix}: Cumulative Energy (σ²)")
        plt.xlabel("Components")
        plt.ylabel("Cumulative fraction")
        plt.ylim(0, 1.05)
        plt.grid(True)
        maybe_save(fig2, "02_cumulative_energy.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig2)

        # 03) Distribution of ||A x|| (random unit x)
        if random_map_samples > 0:
            gen = rng if rng is not None else np.random.default_rng()
            X = gen.standard_normal(size=(n, random_map_samples))
            X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
            norms = np.linalg.norm(A @ X, axis=0)
            fig3 = plt.figure()
            plt.hist(norms, bins=histogram_bins)
            plt.title(f"{title_prefix}: Distribution of ||A x|| (x ~ unit sphere)")
            plt.xlabel("||A x||")
            plt.ylabel("Count")
            plt.grid(True)
            maybe_save(fig3, "03_norm_histogram.png")
            if show_plots:
                plt.show()
            else:
                plt.close(fig3)

        # 04) Condition numbers (bar chart)
        fig4 = plt.figure()
        labels = ["κ₂", "κ₁", "κ_∞"]
        values = [cond2, cond1, condinf]
        plt.bar(labels, values)
        plt.title(f"{title_prefix}: Condition Numbers")
        plt.ylabel("value")
        plt.grid(True, axis="y")
        maybe_save(fig4, "04_condition_numbers.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig4)

        # 05) Row l2 norms histogram
        fig5 = plt.figure()
        plt.hist(row_l2, bins=histogram_bins)
        plt.title(f"{title_prefix}: Row ℓ2 Norms")
        plt.xlabel("||row_i||₂")
        plt.ylabel("Count")
        plt.grid(True)
        maybe_save(fig5, "05_row_l2_hist.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig5)

        # 06) Column l2 norms histogram
        fig6 = plt.figure()
        plt.hist(col_l2, bins=histogram_bins)
        plt.title(f"{title_prefix}: Column ℓ2 Norms")
        plt.xlabel("||col_j||₂")
        plt.ylabel("Count")
        plt.grid(True)
        maybe_save(fig6, "06_col_l2_hist.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig6)

        # 07) Row leverage histogram
        fig7 = plt.figure()
        plt.hist(row_lev, bins=histogram_bins)
        plt.title(f"{title_prefix}: Row Leverage ‖U[i,:]‖²")
        plt.xlabel("row leverage")
        plt.ylabel("Count")
        plt.grid(True)
        maybe_save(fig7, "07_row_leverage_hist.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig7)

        # 08) Column leverage histogram
        fig8 = plt.figure()
        plt.hist(col_lev, bins=histogram_bins)
        plt.title(f"{title_prefix}: Column Leverage ‖V[j,:]‖²")
        plt.xlabel("column leverage")
        plt.ylabel("Count")
        plt.grid(True)
        maybe_save(fig8, "08_col_leverage_hist.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig8)

        # 09) Eigenvalues of R on unit circle (ALWAYS)
        fig9 = plt.figure()
        ax = plt.gca()
        ax.scatter(eig_R.real, eig_R.imag, s=8)
        t = np.linspace(0, 2 * np.pi, 512)
        ax.plot(np.cos(t), np.sin(t), linewidth=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(f"{title_prefix}: Eigenvalues of R (unit circle)")
        ax.grid(True)
        maybe_save(fig9, "09_R_eigs_unit_circle.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig9)

        # 10) Histogram of eigen-angles θ
        fig10 = plt.figure()
        plt.hist(angles, bins=angle_hist_bins, range=(-np.pi, np.pi))
        plt.title(f"{title_prefix}: R Eigen-Angles θ (radians)")
        plt.xlabel("θ (−π, π]")
        plt.ylabel("Count")
        plt.grid(True)
        plt.yscale("log")
        maybe_save(fig10, "10_R_angle_hist.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig10)

    return results


def run():
    savedir = "results"
    os.makedirs(savedir, exist_ok=True)

    all_tuned_lens_paths = [
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-0/lr-0.001/projection_layer0.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-1/lr-0.001/projection_layer1.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-2/lr-0.001/projection_layer2.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-3/lr-0.001/projection_layer3.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-4/lr-0.001/projection_layer4.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-5/lr-0.001/projection_layer5.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-6/lr-0.001/projection_layer6.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-7/lr-0.001/projection_layer7.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-8/lr-0.001/projection_layer8.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-9/lr-0.001/projection_layer9.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-10/lr-0.001/projection_layer10.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-11/lr-0.001/projection_layer11.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-12/lr-0.001/projection_layer12.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-13/lr-0.001/projection_layer13.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-14/lr-0.001/projection_layer14.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-15/lr-0.001/projection_layer15.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-16/lr-0.001/projection_layer16.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-17/lr-0.001/projection_layer17.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-18/lr-0.001/projection_layer18.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-19/lr-0.001/projection_layer19.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-20/lr-0.001/projection_layer20.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-21/lr-0.001/projection_layer21.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-22/lr-0.001/projection_layer22.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-23/lr-0.001/projection_layer23.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-24/lr-0.001/projection_layer24.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-25/lr-0.001/projection_layer25.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-26/lr-0.001/projection_layer26.pt",
        "/from_s3/tuned_lens/Qwen2.5-Math-7B/deepscaler_greedy/tuned_lens_layer-27/lr-0.001/projection_layer27.pt",
    ]
    all_steering_vectors_paths = [
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-0/seed-0_lr-0.003/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-1/seed-0_lr-0.007/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-2/seed-0_lr-0.005/checkpoint-212/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-3/seed-0_lr-0.007/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-4/seed-0_lr-0.007/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-5/seed-0_lr-0.007/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-6/seed-0_lr-0.007/checkpoint-212/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-7/seed-0_lr-0.007/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-8/seed-0_lr-0.01/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-9/seed-0_lr-0.01/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-10/seed-0_lr-0.01/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-11/seed-0_lr-0.01/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-12/seed-0_lr-0.01/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-13/seed-0_lr-0.01/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-14/seed-0_lr-0.01/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-0_lr-0.01/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-16/seed-0_lr-0.01/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-17/seed-0_lr-0.01/checkpoint-159/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-18/seed-0_lr-0.03/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-19/seed-0_lr-0.05/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-20/seed-0_lr-0.05/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-21/seed-0_lr-0.1/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-22/seed-0_lr-0.1/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-23/seed-0_lr-0.01/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-24/seed-0_lr-0.01/checkpoint-265/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-25/seed-0_lr-0.1/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-0_lr-0.1/checkpoint-314/steering_vectors.npy",
        "/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-0_lr-0.1/checkpoint-265/steering_vectors.npy",
    ]
    all_tuned_lens = [torch.load(x) for x in all_tuned_lens_paths]
    all_steering_vectors = [
        np.load(x)[idx] for idx, x in enumerate(all_steering_vectors_paths)
    ]
    all_projected_steering_vectors = [
        tn["proj.weight"]
        @ torch.from_numpy(sv).to(tn["proj.weight"].device).to(tn["proj.weight"].dtype)
        + tn["proj.bias"]
        for tn, sv in zip(all_tuned_lens, all_steering_vectors)
    ]

    # for idx, tuned_lens in enumerate(all_tuned_lens):
    #     svd_analyze(
    #         A=tuned_lens["proj.weight"].float().cpu().numpy(),
    #         random_map_samples=1_000,
    #         savedir=os.path.join(savedir, f"tuned_lens_{idx}"),
    #     )

    local_savedir = os.path.join(savedir, "cossims")
    os.makedirs(local_savedir, exist_ok=True)

    cossims = rowwise_cosine(
        a=np.stack(all_steering_vectors),
        b=torch.stack(all_projected_steering_vectors).float().cpu().numpy(),
    )
    plt.bar(x=range(len(cossims)), height=cossims)
    plt.ylim(-1, 1)
    plt.xlabel("Layer")
    plt.ylabel("Cossim")
    plt.title("Cossim between steering vectors and projected steering vectors")
    plt.savefig(
        os.path.join(
            local_savedir, "steering_vectors_vs_projected_steering_vectors.png"
        )
    )
    plt.close()

    draw_pairwise_cossims_hist(
        steering_vectors=np.stack(all_steering_vectors),
        savepath=os.path.join(local_savedir, "pairwise_cossims_vanilla_hist.png"),
    )
    draw_pairwise_cossims_hist(
        steering_vectors=torch.stack(all_projected_steering_vectors)
        .float()
        .cpu()
        .numpy(),
        savepath=os.path.join(local_savedir, "pairwise_cossims_projected_hist.png"),
    )
    draw_pairwise_cossims(
        steering_vectors=np.stack(all_steering_vectors),
        savepath=os.path.join(local_savedir, "pairwise_cossims_vanilla.png"),
    )
    draw_pairwise_cossims(
        steering_vectors=torch.stack(all_projected_steering_vectors)
        .float()
        .cpu()
        .numpy(),
        savepath=os.path.join(local_savedir, "pairwise_cossims_projected.png"),
    )


if __name__ == "__main__":
    run()
