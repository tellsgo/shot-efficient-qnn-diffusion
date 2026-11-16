"""Generate the multi-step shot-budget figure for the GMM experiment.

text T_infer ∈ {10, 20, 50} text"text shot"text，
text T=20 text fig_gmm_shots_per_step.png，
text"text、text"。

【text】——text shot text run_T1000_mainline.py text part_infer
text（text M.SEED+202，text）。text CSV text。

text（text：3-qubit text diagW + text phi_max=pi）：
    scripts/results/final_gmm_diagW_pi/diagW/infer/T{10,20,50}/metrics.csv
text metrics.csv text（part_infer text 316-320 text）：
    text 1 text:  T_infer,rf,mmd2,total_shots
    text 2 text:  20,0.9383,0.00413,303
    text
    text:        reverse_step,mean_shots
    text:     text,text shot
text：
    python3 scripts/run_T1000_mainline.py --part infer --modes diagW ...
    （--infers text 10/20/50；text）

text = text step/total ∈ (0,1]，text 1.0。
      （text，text step/total text t/T）
text = text shot，text（2 text ~200，text，text）。
text shot；T=20（text）text、text。

text（text，quddpm text；text numpy+matplotlib，text 1 text）：
    python3 scripts/plotting/make_fig_shots_multistep.py
text results/figures/fig_gmm_shots_multistep.png，
【text、text】text fig_gmm_shots_per_step.png。
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "axes.titleweight": "normal", "figure.autolayout": False,
})
SAVE = dict(dpi=150, bbox_inches="tight")
HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent


STEP_STYLE = {
    10: dict(color="#c85a11", lw=2.0, ms=6, zorder=3),
    20: dict(color="#2970b0", lw=3.0, ms=8, zorder=5),
    50: dict(color="#2a9d4a", lw=2.0, ms=4, zorder=3),
}


def read_metrics(csv_path: Path):
    """text part_infer text metrics.csv。
    text (total_shots:int, rf:float, steps:np.ndarray, shots:np.ndarray)。"""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    rf = float(rows[1][1])
    total = int(round(float(rows[1][3])))
    h = next(i for i, r in enumerate(rows) if r and r[0] == "reverse_step")
    per = [(int(r[0]), float(r[1])) for r in rows[h + 1:] if len(r) >= 2 and r[0].strip()]
    steps = np.array([s for s, _ in per], dtype=float)
    shots = np.array([v for _, v in per], dtype=float)
    return total, rf, steps, shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=SCRIPTS_DIR / "results" / "final_gmm_diagW_pi" / "diagW" / "infer",
                    help="infer text（text T{k}/metrics.csv）")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50],
                    help="text（text src text T{k}/metrics.csv）")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "results" / "figures" / "fig_gmm_shots_multistep.png")
    args = ap.parse_args()

    src = args.src
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for T in args.steps:
        total, rf, steps, shots = read_metrics(src / f"T{T}" / "metrics.csv")
        x = steps / float(T)
        style = STEP_STYLE.get(T, dict(lw=2.0, ms=6))
        ax.plot(x, shots, "-o", label=rf"$T_{{\mathrm{{infer}}}}={T}$  ({total} shots)", **style)
        floor = int(np.sum(shots <= 2.0))
        print(f"  T={T:>3d}: total={total:>4d}  rf={rf:.3f}  last-step={shots[-1]:.0f}  floor@2 for {floor}/{T} steps")

    ax.axhline(2, color="#999999", ls=":", lw=1.2, alpha=0.8)
    ax.text(0.02, 3.0, r"$M_{\min}=2$", va="bottom", ha="left", color="#555555", fontsize=16)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.set_ylim(1.7, 270)
    ax.set_ylabel("Mean shots at this step")
    ax.set_xlabel("Fraction of reverse-diffusion steps completed")
    ax.set_xlim(0, 1.03)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", framealpha=0.92)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
