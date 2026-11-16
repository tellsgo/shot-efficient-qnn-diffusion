"""make_fig11_bigfont.py — text Fig 11 (fig_gmm_readout) text，text +2。

text: Target | frozen 24I (fixed24I) | diagonal W (diagW) | per-dim, 8 qubits (8q)。
text "text + text seed(SEED+202)" text，text；
text fontsize 15 -> 17（rcParams text +1~2）。text。

text（text）:
  python scripts/plotting/make_fig11_bigfont.py --part dump3q
  python scripts/plotting/make_fig11_bigfont.py --part dump8q
  python scripts/plotting/make_fig11_bigfont.py --part plot

dump3q / dump8q text（text~text，text）。plot text、text。
text results/figures/fig11_rebuild/，text
results/figures/fig_gmm_readout.png。

text：text FONT（text）text；text 16，text 18。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
RES8 = SCRIPTS_DIR / "results" / "final_8q"
OUT = PROJECT_ROOT / "results" / "figures" / "fig11_rebuild"
FINAL_OUT = PROJECT_ROOT / "results" / "figures" / "fig_gmm_readout.png"
OUT.mkdir(parents=True, exist_ok=True)

TINFER = 20
NSAMP = 3000

FONT = 17


def _theta(train_dir):
    """text：weights.npz(text, diagW) text checkpoint.npz(text theta, fixed24I)。"""
    wp = train_dir / "weights.npz"
    if wp.exists():
        d = np.load(wp)
        return {k: d[k] for k in d.files if k != "mode"}
    d = np.load(train_dir / "checkpoint.npz", allow_pickle=True)
    return {k: np.asarray(v) for k, v in d["theta"].item().items()}


def part_dump3q():
    import math
    import run_T1000_mainline as T
    import run_quantum_3q_gpu as M
    import noise_budgeted_generation as G
    M.configure_gpu(8)
    T.DEALIAS_SCALE = math.pi / (0.6 * M.N + 0.15 * M.N * M.N)
    xis = G.psd_sqrt(G.projection(M.N))[0]
    C = G.mixture_centers(M.N)
    target = G.sample_target(np.random.default_rng(M.SEED + 99), NSAMP, C, M.S0, xis)
    out = {"target": target, "centers": C}
    root = SCRIPTS_DIR / "results" / "final_gmm_diagW_pi"
    for key, mode in [("frozen", "fixed24I"), ("diag", "diagW")]:
        theta = T._load_theta(root, mode)
        res = T.generate(theta, mode, 1000, TINFER, M.SEED + 202, NSAMP)
        x, sh = res[0], res[1]
        out[f"cloud_{key}"] = x
        print(f"  [{mode}] cloud={x.shape} mean_shots={sh:.0f}", flush=True)
    np.savez(OUT / "clouds_3q.npz", **out)
    print("saved", OUT / "clouds_3q.npz", flush=True)


def part_dump8q():
    import run_8qubit_perdim_gpu as E
    wr = np.load(RES8 / "train" / "weights.npz")["wr"]
    x, sh, per, snaps = E.generate(wr, 1000, TINFER, E.SEED + 202, NSAMP, "adaptive")
    np.savez(OUT / "clouds_8q.npz", cloud_perdim=x)
    print(f"  [8q per-dim] cloud={x.shape} mean_shots={sh:.0f}", flush=True)
    print("saved", OUT / "clouds_8q.npz", flush=True)


def part_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 20, "axes.labelsize": FONT,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
    })
    d3 = np.load(OUT / "clouds_3q.npz")
    d8 = np.load(OUT / "clouds_8q.npz")
    target, C = d3["target"], d3["centers"]
    panels = [
        ("Target",                                          target,             "#555555"),
        ("frozen 24$\\mathbf{I}$\n(0 readout params)",      d3["cloud_frozen"], "#2970b0"),
        ("diagonal $\\mathbf{W}$\n(8 readout params)",      d3["cloud_diag"],   "#2970b0"),
        ("per-dim, 8 qubits\n(0 readout params)",           d8["cloud_perdim"], "#2970b0"),
    ]
    cm = C.mean(0)
    _, _, vh = np.linalg.svd(C - cm, full_matrices=False)
    B2 = vh[:2].T
    pr = lambda X: (X - cm) @ B2
    cp = pr(C)
    PT = pr(target)
    mg = 0.4 * max(np.ptp(PT[:, 0]), np.ptp(PT[:, 1]))
    xl = (PT[:, 0].min() - mg, PT[:, 0].max() + mg)
    yl = (PT[:, 1].min() - mg, PT[:, 1].max() + mg)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), layout="constrained")
    for ax, (lbl, X, col) in zip(axes, panels):
        P2 = pr(X)
        ax.scatter(P2[:, 0], P2[:, 1], s=10, c=col, alpha=0.45, edgecolors="none")
        ax.scatter(cp[:, 0], cp[:, 1], s=230, facecolors="none",
                   edgecolors="black", linewidths=2.2, zorder=5)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(lbl, fontsize=FONT)
    outp = FINAL_OUT
    fig.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", outp, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dump3q", "dump8q", "plot"])
    a = ap.parse_args()
    {"dump3q": part_dump3q, "dump8q": part_dump8q, "plot": part_plot}[a.part]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
