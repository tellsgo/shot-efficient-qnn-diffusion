"""run_8qubit_mainline.py — 8-qubit text（GMM），text。

text（import，text）8-qubit text run_8qubit_perdim_gpu(E)：text ⟨Z_i⟩=x_i/A、
text 3I text x̂0=A·⟨Z⟩（**text**）、text DDIM text shot text。
text（text、text、bbox=tight）。

text：text、0 text，text GMM text → text。

--part：
  train  text（text，text）
  gen    text rep：text(Target|High-shot|Adaptive，text rf/MMD/shots) + text
  perdim text（8 text text vs target text；text：text qubit text）
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_8qubit_perdim_gpu as E
import noise_budgeted_generation as G
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.titleweight": "normal",
})
BLUE = "#2970b0"; GREY = "#555555"
SAVE = dict(dpi=150, bbox_inches="tight")


def pca_basis(C, target):
    cm = C.mean(0); _, _, vh = np.linalg.svd(C - cm, full_matrices=False); B2 = vh[:2].T
    PT = (target - cm) @ B2
    mg = 0.45 * max(np.ptp(PT[:, 0]), np.ptp(PT[:, 1]))
    xl = (PT[:, 0].min() - mg, PT[:, 0].max() + mg); yl = (PT[:, 1].min() - mg, PT[:, 1].max() + mg)
    return cm, B2, xl, yl


def _panel(ax, X, C, cm, B2, xl, yl, color, label, fontsize=15):
    """text；label text【text】（xlabel），text。"""
    P2 = (X - cm) @ B2; cp = (C - cm) @ B2
    ax.scatter(P2[:, 0], P2[:, 1], s=10, c=color, alpha=0.45, edgecolors="none")
    ax.scatter(cp[:, 0], cp[:, 1], s=230, facecolors="none", edgecolors="black", linewidths=2.2, zorder=5)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(label, fontsize=fontsize)


def fig_triptych(od, rep, target, hi, ad, C, basis):
    """Target | High-shot | Adaptive（text rf/MMD/shots）。hi/ad = (X, rf, mmd, sh)。"""
    cm, B2, xl, yl = basis
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.0), layout="constrained")
    _panel(ax[0], target, C, cm, B2, xl, yl, GREY, "Target")
    _panel(ax[1], hi[0], C, cm, B2, xl, yl, BLUE,
           f"High-shot\nrf={hi[1]:.3f}   MMD$^2$={hi[2]:.4f}   {hi[3]:.0f} shots")
    _panel(ax[2], ad[0], C, cm, B2, xl, yl, BLUE,
           f"Adaptive (ours)\nrf={ad[1]:.3f}   MMD$^2$={ad[2]:.4f}   {ad[3]:.0f} shots")
    fig.savefig(od / f"triptych_8q_T{rep}.png", **SAVE); plt.close(fig)


def fig_denoise4(od, rep, snaps, C, basis):
    """text：text→text text 4 text（text）。"""
    cm, B2, xl, yl = basis; n = len(snaps) - 1
    picks = sorted(set([0, round(n / 3), round(2 * n / 3), n]))
    while len(picks) < 4: picks.append(picks[-1])
    labels = ["Noise (start)", "Intermediate", "Intermediate", "Denoised (final)"]
    fig, ax = plt.subplots(1, 4, figsize=(20.5, 5.4), layout="constrained")
    for a, i, lb in zip(ax, picks, labels):
        _panel(a, snaps[i], C, cm, B2, xl, yl, BLUE, f"{lb}\n(step {i}/{n})")
    fig.savefig(od / f"denoise_8q_T{rep}.png", **SAVE); plt.close(fig)


def fig_perdim(od, rep, x_gen, target):
    """text：8 text text vs target text（text qubit text）。"""
    N = E.N
    fig, ax = plt.subplots(2, 4, figsize=(18, 8), layout="constrained")
    ax = ax.ravel()
    for i in range(N):
        lo = min(target[:, i].min(), x_gen[:, i].min()); hi = max(target[:, i].max(), x_gen[:, i].max())
        bins = np.linspace(lo, hi, 31)
        ax[i].hist(target[:, i], bins=bins, density=True, color=GREY, alpha=0.5, label="target")
        ax[i].hist(x_gen[:, i], bins=bins, density=True, histtype="step", color=BLUE, lw=2.2, label="generated")
        ax[i].set_yticks([]); ax[i].set_xlabel(f"qubit {i+1}  (dim {i+1})", fontsize=14)
    ax[0].legend(fontsize=12, loc="upper right")
    fig.savefig(od / f"perdim_8q_T{rep}.png", **SAVE); plt.close(fig)


def part_train(root, args, C):
    E.train(root / "train", C, args.Ttrain, args.iters, args.samples, args.batch,
            args.lr, args.layers, args.clip, args.resume, max(args.iters // 10, 1))


def _load(root):
    wp = root / "train" / "weights.npz"
    if not wp.exists(): raise SystemExit(f"text {wp}，text --part train")
    return np.load(wp)["wr"]


def part_gen(root, args, C, target, gamma):
    wr = _load(root); basis = pca_basis(C, target)
    x_ad, sh_ad, per, snaps = E.generate(wr, args.Ttrain, args.rep, E.SEED + 202,
                                         args.gen_samples, "adaptive", keep_snaps=True)
    rf_ad = E.resolved_frac(x_ad, C); mmd_ad = float(G.mmd2_rbf(x_ad, target, gamma))
    x_hi, sh_hi, _, _ = E.generate(wr, args.Ttrain, args.rep, E.SEED + 201,
                                   args.gen_samples, "high", m_fixed=args.high_M)
    rf_hi = E.resolved_frac(x_hi, C); mmd_hi = float(G.mmd2_rbf(x_hi, target, gamma))
    fig_triptych(root, args.rep, target, (x_hi, rf_hi, mmd_hi, sh_hi), (x_ad, rf_ad, mmd_ad, sh_ad), C, basis)
    fig_denoise4(root, args.rep, snaps, C, basis)
    (root / f"metrics_T{args.rep}.txt").write_text(
        f"8-qubit fixed-3I (0 trainable readout params), T_infer={args.rep}\n"
        f"adaptive : rf={rf_ad:.3f}  MMD2={mmd_ad:.4f}  total_shots={sh_ad:.0f}\n"
        f"high_shot: rf={rf_hi:.3f}  MMD2={mmd_hi:.4f}  total_shots={sh_hi:.0f}\n")
    print(f"    [gen T{args.rep}] adaptive rf={rf_ad:.3f} MMD2={mmd_ad:.4f} shots={sh_ad:.0f} | "
          f"high rf={rf_hi:.3f} shots={sh_hi:.0f}", flush=True)


def part_perdim(root, args, C, target, gamma):
    wr = _load(root)
    x_ad, _, _, _ = E.generate(wr, args.Ttrain, args.rep, E.SEED + 202, args.gen_samples, "adaptive")
    fig_perdim(root, args.rep, x_ad, target)
    print(f"    [perdim T{args.rep}] 8 text -> perdim_8q_T{args.rep}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "gen", "perdim"])
    ap.add_argument("--outdir", default="results/final_8q")
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=80000)
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--rep", type=int, default=20, help="text（gen/perdim text）")
    ap.add_argument("--high-M", type=int, default=80000)
    ap.add_argument("--gen-samples", type=int, default=1500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    C = G.mixture_centers(E.N)
    target = E.sample_gmm(np.random.default_rng(E.SEED + 99), args.gen_samples, C)
    gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))
    root = Path(args.outdir); root.mkdir(parents=True, exist_ok=True)
    print(f"[8q part={args.part}] T={args.Ttrain} rep={args.rep} iters={args.iters} -> {root}", flush=True)
    t0 = time.time()
    if args.part == "train":    part_train(root, args, C)
    elif args.part == "gen":    part_gen(root, args, C, target, gamma)
    elif args.part == "perdim": part_perdim(root, args, C, target, gamma)
    print(f"[8q part={args.part}] text，text {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
