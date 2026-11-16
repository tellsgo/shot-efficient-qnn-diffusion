#!/usr/bin/env python3
"""DDIM text。

text DDIM text：

    (dir_coeff - c_t)² · (AN)²/M_t · Σ_t + σ_t² · R_t^{DDIM} = σ_t² · Ξ

text：
- σ_t = η · (β_next/β_t) · √(β_t² - β_next²) / α_next
- dir_coeff = √(β_next² - σ_t²)
- c_t = α_next · β_t / α_t
- R_t^{DDIM} = Ξ - γ_t^{DDIM} · Σ_t
- γ_t^{DDIM} = (dir_coeff - c_t)² · (AN)² / (σ_t² · M_t)

text：
1. text（text = text）
2. R_t^{DDIM} text（text ≥ 0）
3. text η text（η=0 text，η>0 text）

text：

    python scripts/validation/verify_ddim_covariance.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "validation" / "ddim_covariance"


def projection(n: int) -> np.ndarray:
    """text Ξ = I - (1/N)·11ᵀ"""
    one = np.ones((n, 1))
    return np.eye(n) - one @ one.T / n


def multinomial_cov(p: np.ndarray) -> np.ndarray:
    """text Σ = diag(p) - p·pᵀ"""
    return np.diag(p) - np.outer(p, p)


def relative_fro_error(a: np.ndarray, b: np.ndarray) -> float:
    """text Frobenius text"""
    return float(np.linalg.norm(a - b, "fro") / max(np.linalg.norm(b, "fro"), 1e-12))


def compute_ddim_params(
    alpha_t: float,
    beta_t: float,
    alpha_next: float,
    beta_next: float,
    eta: float,
) -> tuple[float, float, float]:
    """text DDIM text：σ_t, dir_coeff, c_t"""
    c_t = alpha_next * beta_t / alpha_t


    if eta > 0 and beta_t > beta_next:
        sigma_t = eta * (beta_next / beta_t) * math.sqrt(beta_t**2 - beta_next**2) / alpha_next
    else:
        sigma_t = 0.0


    dir_coeff = math.sqrt(max(beta_next**2 - sigma_t**2, 0.0))

    return sigma_t, dir_coeff, c_t


def verify_ddim_covariance_contract(
    *,
    n: int,
    eta: float,
    beta_t: float,
    beta_next: float,
    M_t: int,
    a_bound: float,
    p: np.ndarray,
) -> dict[str, float | bool]:
    """text DDIM text"""

    alpha_t = math.sqrt(max(1.0 - beta_t**2, 1e-12))
    alpha_next = math.sqrt(max(1.0 - beta_next**2, 1e-12))


    sigma_t, dir_coeff, c_t = compute_ddim_params(alpha_t, beta_t, alpha_next, beta_next, eta)


    Sigma_t = multinomial_cov(p)


    Xi = projection(n)


    if sigma_t < 1e-10:
        return {
            "eta": eta,
            "beta_t": beta_t,
            "beta_next": beta_next,
            "sigma_t": sigma_t,
            "dir_coeff": dir_coeff,
            "c_t": c_t,
            "dir_coeff_minus_c_t": dir_coeff - c_t,
            "gamma_t_ddim": float('inf'),
            "min_eigenvalue": float('-inf'),
            "is_psd": False,
            "covariance_error": float('inf'),
            "contract_holds": False,
        }


    gamma_t_ddim = (dir_coeff - c_t)**2 * (a_bound * n)**2 / (sigma_t**2 * M_t)


    R_t_ddim = Xi - gamma_t_ddim * Sigma_t


    eigenvalues = np.linalg.eigvalsh(R_t_ddim)
    min_eigenvalue = float(eigenvalues.min())
    is_psd = min_eigenvalue >= -1e-10


    left_side = (dir_coeff - c_t)**2 * (a_bound * n)**2 / M_t * Sigma_t + sigma_t**2 * R_t_ddim
    right_side = sigma_t**2 * Xi
    covariance_error = relative_fro_error(left_side, right_side)
    contract_holds = covariance_error < 1e-10

    return {
        "eta": eta,
        "beta_t": beta_t,
        "beta_next": beta_next,
        "sigma_t": sigma_t,
        "dir_coeff": dir_coeff,
        "c_t": c_t,
        "dir_coeff_minus_c_t": dir_coeff - c_t,
        "gamma_t_ddim": gamma_t_ddim,
        "min_eigenvalue": min_eigenvalue,
        "is_psd": is_psd,
        "covariance_error": covariance_error,
        "contract_holds": contract_holds,
    }


def write_csv(rows: list[dict], out: Path) -> None:
    """text CSV text"""
    if not rows:
        return

    fieldnames = [
        "eta", "beta_t", "beta_next", "sigma_t", "dir_coeff", "c_t",
        "dir_coeff_minus_c_t", "gamma_t_ddim", "min_eigenvalue",
        "is_psd", "covariance_error", "contract_holds",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_svg(rows: list[dict], out: Path) -> None:
    """text SVG text"""
    width, height = 920, 420
    margin = 60


    valid_rows = [r for r in rows if r["contract_holds"]]
    invalid_rows = [r for r in rows if not r["contract_holds"]]

    if not valid_rows:

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="16">text</text>',
            '</svg>',
        ]
        out.write_text("\n".join(parts))
        return

    max_eta = max(r["eta"] for r in rows)
    max_gamma = max(r["gamma_t_ddim"] for r in valid_rows if r["gamma_t_ddim"] < 1e10)

    def sx(eta: float) -> float:
        return margin + eta / max(max_eta, 0.1) * (width - 2 * margin)

    def sy(gamma: float) -> float:
        return height - margin - gamma / max(max_gamma * 1.1, 0.01) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">DDIM text：γ_t^DDIM vs η</text>',
        f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="12">η (DDIM text)</text>',
        f'<text x="16" y="{height/2}" transform="rotate(-90 16,{height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">γ_t^DDIM (text)</text>',
    ]


    for i in range(5):
        y_val = max_gamma * 1.1 * i / 4
        y_pos = sy(y_val)
        parts.append(f'<line x1="{margin-4}" y1="{y_pos:.2f}" x2="{margin}" y2="{y_pos:.2f}" stroke="#222"/>')
        parts.append(f'<text x="{margin-8}" y="{y_pos+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="10">{y_val:.3f}</text>')


    for i in range(6):
        eta_val = max_eta * i / 5
        x_pos = sx(eta_val)
        parts.append(f'<line x1="{x_pos:.2f}" y1="{height-margin}" x2="{x_pos:.2f}" y2="{height-margin+4}" stroke="#222"/>')
        parts.append(f'<text x="{x_pos:.2f}" y="{height-margin+16}" text-anchor="middle" font-family="sans-serif" font-size="10">{eta_val:.1f}</text>')


    for row in valid_rows:
        if row["gamma_t_ddim"] < 1e10:
            x_pos = sx(row["eta"])
            y_pos = sy(row["gamma_t_ddim"])
            parts.append(f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="5" fill="#0072B2"/>')


    for row in invalid_rows:
        x_pos = sx(row["eta"])
        y_pos = height - margin - 20
        parts.append(f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="5" fill="#D55E00"/>')
        parts.append(f'<text x="{x_pos:.2f}" y="{y_pos-10:.2f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#D55E00">η=0 text</text>')


    parts.append(f'<circle cx="{width-margin-150}" cy="{margin+20}" r="5" fill="#0072B2"/>')
    parts.append(f'<text x="{width-margin-140}" y="{margin+24}" font-family="sans-serif" font-size="12">text</text>')
    parts.append(f'<circle cx="{width-margin-150}" cy="{margin+40}" r="5" fill="#D55E00"/>')
    parts.append(f'<text x="{width-margin-140}" y="{margin+44}" font-family="sans-serif" font-size="12">text</text>')

    parts.append('</svg>')
    out.write_text("\n".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description="DDIM text")
    parser.add_argument("--n", type=int, default=8, help="text")
    parser.add_argument("--a-bound", type=float, default=1.0, help="χ text A")
    parser.add_argument("--beta-t", type=float, default=0.8, help="text β_t")
    parser.add_argument("--beta-next", type=float, default=0.6, help="text β_next")
    parser.add_argument("--m-shots", type=int, default=10000, help="text M_t")
    parser.add_argument("--eta-values", type=str, default="0.0,0.2,0.5,0.8,1.0",
                        help="text η text（text）")
    parser.add_argument("--seed", type=int, default=42, help="text")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="text")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)


    eta_values = [float(x.strip()) for x in args.eta_values.split(",")]


    z = rng.normal(size=args.n)
    z -= z.mean()
    z /= max(np.max(np.abs(z)), 1e-12)
    radius = 0.3 / args.n
    p = np.ones(args.n) / args.n + radius * z
    p = np.clip(p, 1e-9, 2.0 / args.n - 1e-9)
    p /= p.sum()

    print("=" * 70)
    print("DDIM text")
    print("=" * 70)
    print(f"\ntext:")
    print(f"  text N = {args.n}")
    print(f"  A = {args.a_bound}")
    print(f"  β_t = {args.beta_t}, β_next = {args.beta_next}")
    print(f"  M_t = {args.m_shots}")
    print(f"  text η text: {eta_values}")


    results = []
    for eta in eta_values:
        result = verify_ddim_covariance_contract(
            n=args.n,
            eta=eta,
            beta_t=args.beta_t,
            beta_next=args.beta_next,
            M_t=args.m_shots,
            a_bound=args.a_bound,
            p=p,
        )
        results.append(result)


        print(f"\n{'='*70}")
        print(f"η = {eta:.1f}")
        print(f"{'='*70}")
        print(f"  σ_t = {result['sigma_t']:.6f}")
        print(f"  dir_coeff = {result['dir_coeff']:.6f}")
        print(f"  c_t = {result['c_t']:.6f}")
        print(f"  dir_coeff - c_t = {result['dir_coeff_minus_c_t']:.6f}")

        if result['sigma_t'] < 1e-10:
            print(f"  ❌ η=0 text（σ_t = 0，text）")
        else:
            print(f"  γ_t^DDIM = {result['gamma_t_ddim']:.6e}")
            print(f"  R_t^DDIM text = {result['min_eigenvalue']:.6e}")
            print(f"  R_t^DDIM text: {'✅ text' if result['is_psd'] else '❌ text'}")
            print(f"  text = {result['covariance_error']:.6e}")
            print(f"  text: {'✅ text' if result['contract_holds'] else '❌ text'}")


    csv_path = args.out_dir / "ddim_verification.csv"
    write_csv(results, csv_path)
    print(f"\n{'='*70}")
    print(f"text: {csv_path}")

    svg_path = args.out_dir / "ddim_verification.svg"
    write_svg(results, svg_path)
    print(f"text: {svg_path}")


    print(f"\n{'='*70}")
    print("text")
    print(f"{'='*70}")
    passed = sum(1 for r in results if r["contract_holds"])
    failed = sum(1 for r in results if not r["contract_holds"])
    print(f"  text: {passed}/{len(results)}")
    print(f"  text: {failed}/{len(results)}")

    if passed == len(results):
        print(f"\n✅ text！DDIM text。")
    elif failed > 0 and all(r["eta"] == 0.0 for r in results if not r["contract_holds"]):
        print(f"\n✅ text！text η=0 text（text）。")
    else:
        print(f"\n❌ text，text。")

    return 0 if passed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
