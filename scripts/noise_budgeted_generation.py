#!/usr/bin/env python3
"""Toy generation test for measurement-noise-budgeted diffusion sampling.

This experiment is intentionally small and numpy-only. It does not train a QNN.
Instead, it uses an oracle denoiser for a bounded zero-sum Gaussian mixture and
reads that denoiser through finite-shot Born probabilities. The purpose is to
test whether the covariance contract also improves the sample-quality/shot-cost
tradeoff, not only a one-step covariance identity.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def projection(n: int) -> np.ndarray:
    one = np.ones((n, 1))
    return np.eye(n) - one @ one.T / n


def project_zero_sum(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def multinomial_cov(p: np.ndarray) -> np.ndarray:
    return np.diag(p) - np.outer(p, p)


def relative_fro_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro") / max(np.linalg.norm(b, "fro"), 1e-12))


def psd_sqrt(cov: np.ndarray) -> tuple[np.ndarray, bool, float]:
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
    min_eval = float(vals.min())
    clipped = bool(min_eval < -1e-9)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(np.sqrt(vals)), clipped, min_eval


def schedule(steps: int, beta_max: float, beta_min: float) -> list[tuple[float, float, float, float]]:
    betas = np.linspace(beta_max, beta_min, steps + 1)
    out = []
    for i in range(steps):
        beta_t = float(betas[i])
        beta_next = float(betas[i + 1])
        alpha_t = math.sqrt(max(1.0 - beta_t**2, 1e-12))
        alpha_next = math.sqrt(max(1.0 - beta_next**2, 1e-12))
        out.append((alpha_t, beta_t, alpha_next, beta_next))
    return out


def compute_ddim_params(
    alpha_t: float, beta_t: float, alpha_next: float, beta_next: float, eta: float,
) -> tuple[float, float, float]:
    """Compute DDIM parameters: sigma_t, dir_coeff, c_t.

    When eta=1.0 the returned (sigma_t, dir_coeff) reproduce the DDPM noise
    structure in a covariance-equivalent sense.  When eta=0 the sampler is
    deterministic (sigma_t=0, dir_coeff=beta_next).

    Returns
    -------
    sigma_t : float
        DDIM noise injection scale.
    dir_coeff : float
        Coefficient on chi_hat in the DDIM update.
    c_t : float
        Denoiser gain alpha_next * beta_t / alpha_t (same as DDPM).
    """
    c_t = alpha_next * beta_t / alpha_t
    if eta > 0.0 and beta_t > beta_next:
        sigma_t = eta * (beta_next / beta_t) * math.sqrt(beta_t**2 - beta_next**2) / alpha_next
    else:
        sigma_t = 0.0
    dir_coeff = math.sqrt(max(beta_next**2 - sigma_t**2, 0.0))
    return sigma_t, dir_coeff, c_t


def mixture_centers(n: int) -> np.ndarray:
    base = np.array([0.55, 0.40, 0.20, 0.05, -0.05, -0.20, -0.40, -0.55])
    if n != 8:
        xs = np.linspace(-1.0, 1.0, n)
        base = 0.55 * xs
    c1 = project_zero_sum(base)
    c2 = project_zero_sum(np.roll(-base, 2))
    c3 = project_zero_sum(np.array([0.45, -0.45, 0.35, -0.35, 0.20, -0.20, 0.05, -0.05]) if n == 8 else 0.35 * np.sin(np.arange(n)))
    return np.vstack([c1, c2, c3])


def sample_target(rng: np.random.Generator, count: int, centers: np.ndarray, sigma0: float, xi_sqrt: np.ndarray) -> np.ndarray:
    idx = rng.integers(0, len(centers), size=count)
    noise = rng.normal(size=(count, centers.shape[1])) @ xi_sqrt.T
    return centers[idx] + sigma0 * noise


def oracle_noise_prediction(
    x_t: np.ndarray,
    *,
    centers: np.ndarray,
    sigma0: float,
    alpha_t: float,
    beta_t: float,
) -> np.ndarray:
    """Return E[W | X_t=x_t] for a zero-sum Gaussian mixture."""

    var_t = alpha_t**2 * sigma0**2 + beta_t**2
    diffs = x_t[:, None, :] - alpha_t * centers[None, :, :]
    logits = -0.5 * np.sum(diffs * diffs, axis=-1) / max(var_t, 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    posterior_x0_per_component = centers[None, :, :] + (alpha_t * sigma0**2 / var_t) * diffs
    posterior_x0 = np.sum(weights[:, :, None] * posterior_x0_per_component, axis=1)
    return project_zero_sum((x_t - alpha_t * posterior_x0) / max(beta_t, 1e-12))


def chi_to_p(chi: np.ndarray, a_bound: float) -> tuple[np.ndarray, int]:
    n = chi.shape[-1]
    p = chi / (n * a_bound) + 1.0 / n
    bad = np.logical_or(p < 0.0, p > 1.0).any(axis=1)
    if np.any(bad):
        # This is outside the clean theorem. Keep the experiment running but
        # report the count so the manuscript can state the limitation.
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum(axis=1, keepdims=True)
    return p, int(np.count_nonzero(bad))


def empirical_born_readout(
    rng: np.random.Generator,
    p: np.ndarray,
    shots: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(p)
    for i, (pi, mi) in enumerate(zip(p, shots, strict=True)):
        out[i] = rng.multinomial(int(mi), pi) / int(mi)
    return out


def min_shots_psd(
    p: np.ndarray,
    *,
    a_bound: float,
    c_t: float,
    beta_next: float,
    sigma_t: float | None = None,
    dir_coeff: float | None = None,
) -> np.ndarray:
    """Minimum shots for PSD residual covariance.

    DDPM (sigma_t is None):
        M_t >= (AN * c_t / beta_next)^2 * lambda_max(Sigma_t)

    DDIM (sigma_t > 0):
        M_t >= (AN * (dir_coeff - c_t) / sigma_t)^2 * lambda_max(Sigma_t)
    """
    n = p.shape[1]
    vals = []
    if sigma_t is not None and dir_coeff is not None and sigma_t > 1e-10:
        # DDIM formula: (dir_coeff - c_t) / sigma_t replaces c_t / beta_next
        coeff = ((dir_coeff - c_t) * a_bound * n / max(sigma_t, 1e-12)) ** 2
    else:
        # DDPM formula
        coeff = (a_bound * n * c_t / max(beta_next, 1e-12)) ** 2
    for pi in p:
        vals.append(math.ceil(coeff * float(np.linalg.eigvalsh(multinomial_cov(pi)).max())))
    return np.asarray(vals, dtype=int)


def sample_residual_noise(
    rng: np.random.Generator,
    p: np.ndarray,
    shots: np.ndarray,
    *,
    xi: np.ndarray,
    a_bound: float,
    c_t: float,
    beta_next: float,
    mode: str,
    sigma_t: float | None = None,
    dir_coeff: float | None = None,
) -> tuple[np.ndarray, int, float]:
    """Sample residual noise from R_t.

    DDPM (sigma_t is None):
        gamma = (AN * c_t)^2 / (M * beta_next^2)
        R_t = Xi - gamma * Sigma_t

    DDIM (sigma_t > 0):
        gamma = (AN * (dir_coeff - c_t))^2 / (M * sigma_t^2)
        R_t = Xi - gamma * Sigma_t

    The returned noise is sqrt(R_t) @ z, which will be scaled externally by
    beta_next (DDPM) or sigma_t (DDIM).
    """
    n = p.shape[1]
    out = np.empty_like(p)
    clips = 0
    min_eval_sum = 0.0
    for i, (pi, mi) in enumerate(zip(p, shots, strict=True)):
        sigma = multinomial_cov(pi)
        if mode == "naive_low":
            residual = xi
        elif mode == "sigma_squared_bug":
            # Bug variant: Sigma^2 instead of Sigma (kept for regression testing)
            if sigma_t is not None and dir_coeff is not None and sigma_t > 1e-10:
                gamma = ((dir_coeff - c_t) * a_bound * n) ** 2 / (int(mi) * sigma_t**2)
            else:
                gamma = (a_bound * n * c_t) ** 2 / (int(mi) * beta_next**2)
            residual = xi - gamma * (sigma @ sigma)
        else:
            # Default (adaptive, high_shot, uniform, fixed_low): correct formula
            if sigma_t is not None and dir_coeff is not None and sigma_t > 1e-10:
                # DDIM formula
                gamma = ((dir_coeff - c_t) * a_bound * n) ** 2 / (int(mi) * sigma_t**2)
            else:
                # DDPM formula
                gamma = (a_bound * n * c_t) ** 2 / (int(mi) * beta_next**2)
            residual = xi - gamma * sigma
        sqrt_residual, clipped, min_eval = psd_sqrt(residual)
        clips += int(clipped)
        min_eval_sum += min_eval
        out[i] = sqrt_residual @ rng.normal(size=n)
    return out, clips, min_eval_sum / len(p)


def run_sampler(
    rng: np.random.Generator,
    *,
    mode: str,
    sample_count: int,
    centers: np.ndarray,
    sigma0: float,
    xi: np.ndarray,
    xi_sqrt: np.ndarray,
    a_bound: float,
    steps: int,
    beta_max: float,
    beta_min: float,
    m_min: int,
    m_low: int,
    m_high: int,
    m_uniform: int | None = None,
    eta: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    n = centers.shape[1]
    x = rng.normal(size=(sample_count, n)) @ xi_sqrt.T
    total_shots = np.zeros(sample_count, dtype=float)
    psd_clips = 0
    affine_violations = 0
    min_eval_acc = []
    adaptive_trace = []

    for step_id, (alpha_t, beta_t, alpha_next, beta_next) in enumerate(
        schedule(steps, beta_max, beta_min), start=1
    ):
        # Compute DDIM parameters (eta=1.0 → DDPM-equivalent, eta=0 → deterministic)
        sigma_t, dir_coeff, c_t = compute_ddim_params(
            alpha_t, beta_t, alpha_next, beta_next, eta,
        )

        chi = oracle_noise_prediction(x, centers=centers, sigma0=sigma0, alpha_t=alpha_t, beta_t=beta_t)
        p, bad = chi_to_p(chi, a_bound)
        affine_violations += bad
        required = min_shots_psd(
            p, a_bound=a_bound, c_t=c_t, beta_next=beta_next,
            sigma_t=sigma_t, dir_coeff=dir_coeff,
        )

        if mode == "adaptive":
            shots = np.maximum(m_min, np.ceil(1.10 * required)).astype(int)
        elif mode == "high_shot":
            shots = np.full(sample_count, m_high, dtype=int)
        elif mode == "uniform_same_budget":
            if m_uniform is None:
                raise ValueError("uniform_same_budget requires m_uniform")
            shots = np.full(sample_count, m_uniform, dtype=int)
        else:
            shots = np.full(sample_count, m_low, dtype=int)

        shots = np.maximum(shots, 2)
        p_hat = empirical_born_readout(rng, p, shots)
        chi_hat = project_zero_sum(a_bound * n * (p_hat - 1.0 / n))
        residual, clips, min_eval = sample_residual_noise(
            rng,
            p,
            shots,
            xi=xi,
            a_bound=a_bound,
            c_t=c_t,
            beta_next=beta_next,
            mode=mode,
            sigma_t=sigma_t,
            dir_coeff=dir_coeff,
        )

        # DDIM reverse update
        if eta > 0.0:
            # DDIM: x_{next} = alpha_next * pred_x0 + dir_coeff * chi_hat + sigma_t * residual
            pred_x0 = (x - beta_t * chi_hat) / alpha_t
            x = alpha_next * pred_x0 + dir_coeff * chi_hat + sigma_t * residual
        else:
            # DDIM eta=0: deterministic, no noise injection
            # x_{next} = alpha_next * pred_x0 + beta_next * chi_hat
            pred_x0 = (x - beta_t * chi_hat) / alpha_t
            x = alpha_next * pred_x0 + beta_next * chi_hat

        total_shots += shots
        psd_clips += clips
        min_eval_acc.append(min_eval)
        adaptive_trace.append((step_id, float(np.mean(shots)), float(np.mean(required))))

    stats = {
        "shots_per_sample": float(np.mean(total_shots)),
        "psd_clips": float(psd_clips),
        "affine_violations": float(affine_violations),
        "mean_residual_min_eval": float(np.mean(min_eval_acc)),
        "trace": adaptive_trace,
    }
    return x, stats


def median_bandwidth(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_pairs: int = 2000) -> float:
    z = np.vstack([x, y])
    count = len(z)
    idx_a = rng.integers(0, count, size=max_pairs)
    idx_b = rng.integers(0, count, size=max_pairs)
    dist2 = np.sum((z[idx_a] - z[idx_b]) ** 2, axis=1)
    med = float(np.median(dist2[dist2 > 1e-12]))
    return max(med, 1e-3)


def mmd2_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None, :] - b[None, :, :]
        return float(np.mean(np.exp(-gamma * np.sum(diff * diff, axis=-1))))

    return kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y)


def write_quality_svg(rows: list[dict[str, float | str]], out: Path) -> None:
    width, height = 740, 420
    margin = 58
    colors = {
        "high_shot": "#009E73",
        "adaptive": "#0072B2",
        "uniform_same_budget": "#E69F00",
        "fixed_low": "#D55E00",
        "naive_low": "#CC79A7",
        "sigma_squared_bug": "#7A3E9D",
    }
    xs = [float(r["shots_per_sample_mean"] if "shots_per_sample_mean" in r else r["shots_per_sample"]) for r in rows]
    ys = [float(r["mmd2_mean"] if "mmd2_mean" in r else r["mmd2"]) for r in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.15, 1e-4)

    def sx(x: float) -> float:
        return margin + (x - x_min + x_pad) / (x_max - x_min + 2 * x_pad) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - y_min + y_pad) / (y_max - y_min + 2 * y_pad) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="16">Toy generation quality vs finite-shot cost</text>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">average shots per generated sample</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">MMD^2 to target mixture</text>',
    ]
    # axis ticks
    n_xticks = 5
    for i in range(n_xticks):
        x_val = (x_min - x_pad) + (x_max - x_min + 2 * x_pad) * i / (n_xticks - 1)
        x_pos = sx(x_val)
        parts.append(f'<line x1="{x_pos:.1f}" y1="{height-margin}" x2="{x_pos:.1f}" y2="{height-margin+4}" stroke="#222"/>')
        if x_val >= 1000:
            parts.append(f'<text x="{x_pos:.1f}" y="{height-margin+16}" text-anchor="middle" font-family="sans-serif" font-size="9">{x_val/1000:.1f}k</text>')
        else:
            parts.append(f'<text x="{x_pos:.1f}" y="{height-margin+16}" text-anchor="middle" font-family="sans-serif" font-size="9">{x_val:.0f}</text>')
    n_yticks = 5
    for i in range(n_yticks):
        y_val = (y_min - y_pad) + (y_max - y_min + 2 * y_pad) * i / (n_yticks - 1)
        y_pos = sy(y_val)
        parts.append(f'<line x1="{margin-4}" y1="{y_pos:.1f}" x2="{margin}" y2="{y_pos:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{margin-6}" y="{y_pos+3:.1f}" text-anchor="end" font-family="sans-serif" font-size="9">{y_val:.4f}</text>')
    for row in rows:
        mode = str(row["mode"])
        x = float(row["shots_per_sample_mean"] if "shots_per_sample_mean" in row else row["shots_per_sample"])
        y = float(row["mmd2_mean"] if "mmd2_mean" in row else row["mmd2"])
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="6" fill="{colors.get(mode, "#333")}"/>')
        parts.append(f'<text x="{sx(x)+9:.2f}" y="{sy(y)+4:.2f}" font-family="sans-serif" font-size="11">{mode}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def summarize_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    modes = []
    for row in rows:
        mode = str(row["mode"])
        if mode not in modes:
            modes.append(mode)
    metrics = [
        "shots_per_sample",
        "mmd2",
        "mean_l2",
        "cov_rel_error",
        "psd_clips",
        "affine_violations",
        "mean_residual_min_eval",
    ]
    summary: list[dict[str, float | str]] = []
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        out: dict[str, float | str] = {"mode": mode, "num_seeds": float(len(selected))}
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in selected], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(vals))
            out[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def run_once(args: argparse.Namespace, seed: int) -> tuple[list[dict[str, float | str]], list[tuple[int, float, float]]]:
    rng = np.random.default_rng(seed)
    xi = projection(args.n)
    xi_sqrt = psd_sqrt(xi)[0]
    centers = mixture_centers(args.n)
    target = sample_target(rng, args.samples, centers, args.sigma0, xi_sqrt)
    bandwidth = median_bandwidth(target, target, rng)
    gamma = 1.0 / bandwidth

    rows: list[dict[str, float | str]] = []
    adaptive_trace: list[tuple[int, float, float]] = []

    def evaluate_mode(mode: str, local_seed_offset: int, m_uniform: int | None = None) -> dict[str, float | str]:
        local_rng = np.random.default_rng(seed + local_seed_offset)
        samples, stats = run_sampler(
            local_rng,
            mode=mode,
            sample_count=args.samples,
            centers=centers,
            sigma0=args.sigma0,
            xi=xi,
            xi_sqrt=xi_sqrt,
            a_bound=args.a_bound,
            steps=args.steps,
            beta_max=args.beta_max,
            beta_min=args.beta_min,
            m_min=args.m_min,
            m_low=args.m_low,
            m_high=args.m_high,
            m_uniform=m_uniform,
            eta=args.eta,
        )
        mean_l2 = float(np.linalg.norm(samples.mean(axis=0) - target.mean(axis=0)))
        cov_rel = relative_fro_error(np.cov(samples, rowvar=False, bias=True), np.cov(target, rowvar=False, bias=True))
        row = {
            "seed": float(seed),
            "mode": mode,
            "shots_per_sample": stats["shots_per_sample"],
            "mmd2": mmd2_rbf(samples, target, gamma),
            "mean_l2": mean_l2,
            "cov_rel_error": cov_rel,
            "psd_clips": stats["psd_clips"],
            "affine_violations": stats["affine_violations"],
            "mean_residual_min_eval": stats["mean_residual_min_eval"],
        }
        if mode == "adaptive":
            nonlocal adaptive_trace
            adaptive_trace = stats["trace"]  # type: ignore[assignment]
        return row

    high = evaluate_mode("high_shot", 101)
    adaptive = evaluate_mode("adaptive", 202)
    uniform_shots = int(math.ceil(float(adaptive["shots_per_sample"]) / args.steps))
    rows.extend(
        [
            high,
            adaptive,
            evaluate_mode("uniform_same_budget", 303, m_uniform=uniform_shots),
            evaluate_mode("fixed_low", 404),
            evaluate_mode("naive_low", 505),
            evaluate_mode("sigma_squared_bug", 606),
        ]
    )
    return rows, adaptive_trace


def write_trace_svg(trace: list[tuple[int, float, float]], out: Path) -> None:
    width, height = 740, 340
    margin = 52
    max_x = max(t[0] for t in trace)
    max_y = max(max(t[1], t[2]) for t in trace) * 1.08

    def sx(x: float) -> float:
        return margin + (x - 1) / max(max_x - 1, 1) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - y / max(max_y, 1.0) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">Adaptive shot schedule</text>',
        f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="12">reverse step</text>',
        f'<text x="16" y="{height/2}" transform="rotate(-90 16,{height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">shots</text>',
    ]
    for idx, color, label in [(1, "#0072B2", "actual"), (2, "#D55E00", "PSD minimum")]:
        pts = " ".join(f"{sx(t[0]):.2f},{sy(t[idx]):.2f}" for t in trace)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx = width - margin - 150
        ly = margin + 18 * (idx - 1)
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+30}" y="{ly+4}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--a-bound", type=float, default=3.0)
    parser.add_argument("--sigma0", type=float, default=0.12)
    parser.add_argument("--beta-max", type=float, default=0.95)
    parser.add_argument("--beta-min", type=float, default=0.10)
    parser.add_argument("--m-min", type=int, default=32)
    parser.add_argument("--m-low", type=int, default=48)
    parser.add_argument("--m-high", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="DDIM stochasticity: 1.0=DDPM, 0.0=deterministic, 0<eta<1=intermediate")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results_noncomm_generation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    adaptive_trace: list[tuple[int, float, float]] = []
    for seed_idx in range(args.num_seeds):
        seed_rows, trace = run_once(args, args.seed + 1000 * seed_idx)
        rows.extend(seed_rows)
        if seed_idx == 0:
            adaptive_trace = trace
    summary = summarize_rows(rows)

    detail_path = args.out_dir / "generation_quality_by_seed.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    csv_path = args.out_dir / "generation_quality.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    trace_path = args.out_dir / "adaptive_shots.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_actual_shots", "mean_psd_minimum"])
        writer.writerows(adaptive_trace)

    write_quality_svg(summary, args.out_dir / "quality_vs_shots.svg")
    write_trace_svg(adaptive_trace, args.out_dir / "adaptive_shots.svg")

    print(f"wrote {detail_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {trace_path}")
    print(f"wrote {args.out_dir / 'quality_vs_shots.svg'}")
    print(f"wrote {args.out_dir / 'adaptive_shots.svg'}")
    for row in summary:
        print(
            f"{row['mode']:>19s}: shots/sample={row['shots_per_sample_mean']:.1f}±{row['shots_per_sample_std']:.1f}, "
            f"MMD^2={row['mmd2_mean']:.4f}±{row['mmd2_std']:.4f}, "
            f"cov_err={row['cov_rel_error_mean']:.3f}±{row['cov_rel_error_std']:.3f}, "
            f"PSD clips={row['psd_clips_mean']:.0f}±{row['psd_clips_std']:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
