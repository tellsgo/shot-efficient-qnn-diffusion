#!/usr/bin/env python3
"""Sanity checks for under-sampled quantum DDPM shot-noise accounting.

The script does not train a diffusion model. It isolates the paper's key
mathematical contract:

    shot_noise_cov + residual_classical_cov = target_DDPM_noise_cov.

It is intentionally numpy-only so it can run on a clean standard machine.

Run from the repository root with:

    python scripts/validation/shot_noise_sanity.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "validation" / "shot_noise"


def projection(n: int) -> np.ndarray:
    one = np.ones((n, 1))
    return np.eye(n) - one @ one.T / n


def multinomial_cov(p: np.ndarray) -> np.ndarray:
    return np.diag(p) - np.outer(p, p)


def psd_sqrt(cov: np.ndarray, *, clip: bool) -> tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
    min_eval = float(vals.min())
    if not clip and min_eval < -1e-10:
        raise ValueError(f"covariance is not PSD; min eigenvalue={min_eval:.3e}")
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(np.sqrt(vals)), min_eval


def relative_fro_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro") / max(np.linalg.norm(b, "fro"), 1e-12))


def make_probability_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return p in the bounded affine image of the zero-sum hyperplane.

    The paper's chi -> p map with chi_i in [-A, A] implies p_i in [0, 2/N].
    We sample a zero-sum perturbation around uniform and rescale it to obey
    that bound without needing rejection loops.
    """

    z = rng.normal(size=n)
    z -= z.mean()
    z /= max(np.max(np.abs(z)), 1e-12)
    radius = rng.uniform(0.15, 0.90) / n
    p = np.ones(n) / n + radius * z
    p = np.clip(p, 1e-9, 2.0 / n - 1e-9)
    p /= p.sum()
    return p


def alpha_beta_schedule(steps: int) -> list[tuple[float, float, float, float]]:
    """Return reverse pairs (alpha_t, beta_t, alpha_next, beta_next)."""

    # Noise variance decreases during reverse sampling; avoid beta_next=0 where
    # the residual covariance formula becomes singular.
    noise_levels = np.linspace(0.92, 0.08, steps + 1)
    pairs = []
    for i in range(steps):
        beta_t = math.sqrt(float(noise_levels[i]))
        alpha_t = math.sqrt(1.0 - beta_t**2)
        beta_next = math.sqrt(float(noise_levels[i + 1]))
        alpha_next = math.sqrt(1.0 - beta_next**2)
        pairs.append((alpha_t, beta_t, alpha_next, beta_next))
    return pairs


def analytic_diagnostics(
    *,
    p: np.ndarray,
    xi: np.ndarray,
    n: int,
    a_bound: float,
    alpha_t: float,
    beta_t: float,
    alpha_next: float,
    beta_next: float,
    shots: int,
) -> dict[str, float | np.ndarray]:
    sigma = multinomial_cov(p)
    denoiser_gain = alpha_next * beta_t / alpha_t
    gamma = (a_bound * n * denoiser_gain) ** 2 / (shots * beta_next**2)

    residual = xi - gamma * sigma
    residual_bug = xi - gamma * (sigma @ sigma)

    target_total = beta_next**2 * xi
    total_good = (denoiser_gain * a_bound * n) ** 2 / shots * sigma + beta_next**2 * residual
    total_bug = (denoiser_gain * a_bound * n) ** 2 / shots * sigma + beta_next**2 * residual_bug

    sigma_eigs = np.linalg.eigvalsh(sigma)
    min_shots = math.ceil(
        ((a_bound * n * denoiser_gain) ** 2)
        * float(sigma_eigs.max())
        / max(beta_next**2, 1e-12)
    )

    good_identity_error = relative_fro_error(total_good, target_total)
    bug_identity_error = relative_fro_error(total_bug, target_total)

    # This assertion is the regression test for the draft's Sigma^2 slip.
    if good_identity_error > 1e-10:
        raise AssertionError(f"correct residual covariance identity failed: {good_identity_error:.3e}")
    if bug_identity_error < 1e-5:
        raise AssertionError("Sigma^2 bug was not detected; choose a less degenerate p")

    return {
        "sigma": sigma,
        "gamma": gamma,
        "min_shots_psd": float(min_shots),
        "lambda_max_sigma": float(sigma_eigs.max()),
        "residual_min_eval": float(np.linalg.eigvalsh((residual + residual.T) / 2.0).min()),
        "residual_bug_min_eval": float(np.linalg.eigvalsh((residual_bug + residual_bug.T) / 2.0).min()),
        "good_identity_error": good_identity_error,
        "bug_identity_error": bug_identity_error,
    }


def simulate_total_noise(
    *,
    rng: np.random.Generator,
    p: np.ndarray,
    xi: np.ndarray,
    n: int,
    a_bound: float,
    alpha_t: float,
    beta_t: float,
    alpha_next: float,
    beta_next: float,
    shots: int,
    trials: int,
    clip_residual: bool,
) -> tuple[float, float]:
    sigma = multinomial_cov(p)
    denoiser_gain = alpha_next * beta_t / alpha_t
    gamma = (a_bound * n * denoiser_gain) ** 2 / (shots * beta_next**2)
    residual = xi - gamma * sigma
    sqrt_residual, min_eval = psd_sqrt(residual, clip=clip_residual)

    samples = np.empty((trials, n))
    for k in range(trials):
        counts = rng.multinomial(shots, p)
        p_hat = counts / shots
        chi_error = a_bound * n * (p_hat - p)
        residual_noise = sqrt_residual @ rng.normal(size=n)
        samples[k] = -denoiser_gain * chi_error + beta_next * residual_noise

    empirical_cov = np.cov(samples, rowvar=False, bias=True)
    target_cov = beta_next**2 * xi
    return relative_fro_error(empirical_cov, target_cov), min_eval


def write_svg(rows: list[dict[str, float | str]], out: Path) -> None:
    width, height = 920, 360
    margin = 54
    methods = ["adaptive", "fixed_low", "fixed_high", "sigma_squared_bug"]
    colors = {
        "adaptive": "#0072B2",
        "fixed_low": "#D55E00",
        "fixed_high": "#009E73",
        "sigma_squared_bug": "#7A3E9D",
    }
    grouped = {m: [] for m in methods}
    for row in rows:
        grouped[str(row["method"])].append((float(row["step"]), float(row["rel_cov_error"])))
    max_x = max(float(row["step"]) for row in rows)
    max_y = max(float(row["rel_cov_error"]) for row in rows)
    max_y = max(max_y * 1.05, 0.02)

    def sx(x: float) -> float:
        return margin + x / max(max_x, 1.0) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - y / max_y * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">Shot-noise covariance matching sanity check</text>',
        f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="12">reverse step</text>',
        f'<text x="16" y="{height/2}" transform="rotate(-90 16,{height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">relative covariance error</text>',
    ]
    for i in range(5):
        y = max_y * i / 4
        yy = sy(y)
        parts.append(f'<line x1="{margin}" y1="{yy:.2f}" x2="{width-margin}" y2="{yy:.2f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin-8}" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="10">{y:.2f}</text>')
    for m in methods:
        pts = grouped[m]
        if not pts:
            continue
        poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{colors[m]}" stroke-width="2"/>')
        lx = width - margin - 180
        ly = margin + 18 * methods.index(m)
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{colors[m]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+30}" y="{ly+4}" font-family="sans-serif" font-size="12">{m}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a-bound", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--m-min", type=int, default=32)
    parser.add_argument("--m-max", type=int, default=4096)
    parser.add_argument("--fixed-low", type=int, default=8)
    parser.add_argument("--fixed-high", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    xi = projection(args.n)

    rows: list[dict[str, float | str]] = []
    for step, (alpha_t, beta_t, alpha_next, beta_next) in enumerate(alpha_beta_schedule(args.steps), start=1):
        p = make_probability_vector(args.n, rng)
        probe = analytic_diagnostics(
            p=p,
            xi=xi,
            n=args.n,
            a_bound=args.a_bound,
            alpha_t=alpha_t,
            beta_t=beta_t,
            alpha_next=alpha_next,
            beta_next=beta_next,
            shots=args.fixed_high,
        )
        min_shots = int(probe["min_shots_psd"])
        adaptive_shots = int(np.clip(math.ceil(1.10 * min_shots), args.m_min, args.m_max))
        configs = [
            ("adaptive", adaptive_shots, True, False),
            ("fixed_low", args.fixed_low, True, False),
            ("fixed_high", args.fixed_high, True, False),
            ("sigma_squared_bug", adaptive_shots, True, True),
        ]
        for method, shots, clip_residual, use_bug in configs:
            if use_bug:
                sigma = multinomial_cov(p)
                denoiser_gain = alpha_next * beta_t / alpha_t
                gamma = (args.a_bound * args.n * denoiser_gain) ** 2 / (shots * beta_next**2)
                residual_bug = xi - gamma * (sigma @ sigma)
                sqrt_bug, min_eval = psd_sqrt(residual_bug, clip=True)
                samples = np.empty((args.trials, args.n))
                for k in range(args.trials):
                    counts = rng.multinomial(shots, p)
                    p_hat = counts / shots
                    chi_error = args.a_bound * args.n * (p_hat - p)
                    samples[k] = -denoiser_gain * chi_error + beta_next * (sqrt_bug @ rng.normal(size=args.n))
                rel_err = relative_fro_error(np.cov(samples, rowvar=False, bias=True), beta_next**2 * xi)
            else:
                rel_err, min_eval = simulate_total_noise(
                    rng=rng,
                    p=p,
                    xi=xi,
                    n=args.n,
                    a_bound=args.a_bound,
                    alpha_t=alpha_t,
                    beta_t=beta_t,
                    alpha_next=alpha_next,
                    beta_next=beta_next,
                    shots=shots,
                    trials=args.trials,
                    clip_residual=clip_residual,
                )
            rows.append(
                {
                    "step": float(step),
                    "method": method,
                    "shots": float(shots),
                    "min_shots_psd": float(min_shots),
                    "beta_t": beta_t,
                    "beta_next": beta_next,
                    "lambda_max_sigma": float(probe["lambda_max_sigma"]),
                    "residual_min_eval": float(min_eval),
                    "rel_cov_error": float(rel_err),
                    "sigma_squared_identity_error": float(probe["bug_identity_error"]),
                }
            )

    csv_path = args.out_dir / "shot_noise_sanity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = args.out_dir / "shot_noise_sanity.svg"
    write_svg(rows, svg_path)

    adaptive = [r for r in rows if r["method"] == "adaptive"]
    fixed_low = [r for r in rows if r["method"] == "fixed_low"]
    bug = [r for r in rows if r["method"] == "sigma_squared_bug"]
    print(f"wrote {csv_path}")
    print(f"wrote {svg_path}")
    print(f"adaptive mean relative covariance error: {np.mean([r['rel_cov_error'] for r in adaptive]):.4f}")
    print(f"fixed_low mean relative covariance error: {np.mean([r['rel_cov_error'] for r in fixed_low]):.4f}")
    print(f"Sigma^2-bug mean relative covariance error: {np.mean([r['rel_cov_error'] for r in bug]):.4f}")
    print(f"max PSD minimum-shot requirement: {max(r['min_shots_psd'] for r in adaptive):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
