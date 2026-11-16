r"""run_gmm_bounds.py — §VI text：M_min / M_max text。

text \eqref{M} text：
  M_min —— text CLT text（text CLT text M·p_i ≳ 5，p_i≈1/N=1/8 ⟹ M≈40）
  M_max —— text（text，text 145）
text，text rf / MMD² / text shots / text text。

【text】。import run_T1000_mainline（text），text generate / _load_theta /
text override / text，text。text，text。

text（text BASE/PY，text IMAGE/GMM text FINAL_commands）：
  --part mmin   text M_min（M_max text）
  --part mmax   text M_max（M_min text）
  --part both   text
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_T1000_mainline as MAIN
import run_quantum_3q_gpu as M
import noise_budgeted_generation as G
import run_quantum_x0 as Q
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
                     "axes.titleweight": "normal"})
BLUE, GREY, RED = "#2970b0", "#555555", "#c0392b"
SAVE = dict(dpi=150, bbox_inches="tight")
NO_CAP = 200000


def sweep(theta, mode, args, C, target, gamma, m_mins, m_maxs, label):
    """text (M_min, M_max) text，text seed text±text。"""
    rows = []
    for mn, mx in zip(m_mins, m_maxs):
        rf, mmd, sh, cap = [], [], [], []
        for s in range(args.seeds):
            x, tot, per, _, sat = MAIN.generate(
                theta, mode, args.Ttrain, args.rep, M.SEED + 700 + 13 * s,
                args.gen_samples, eta=1.0, shot_mode="adaptive",
                m_min=int(mn), m_max=int(mx))
            rf.append(Q.resolved_frac(x, C))
            mmd.append(float(G.mmd2_rbf(x, target, gamma)))
            sh.append(tot); cap.append(sat)
        r = dict(m_min=int(mn), m_max=int(mx),
                 rf=np.mean(rf), rf_sd=np.std(rf),
                 mmd2=np.mean(mmd), mmd2_sd=np.std(mmd),
                 shots=np.mean(sh), shots_sd=np.std(sh),
                 cap=np.mean(cap))
        rows.append(r)
        print(f"    [{label}] M_min={r['m_min']:>6} M_max={r['m_max']:>6} | "
              f"rf={r['rf']:.3f}±{r['rf_sd']:.3f}  MMD2={r['mmd2']:.5f}±{r['mmd2_sd']:.5f}  "
              f"shots={r['shots']:.0f}  cap={r['cap']:.2f}", flush=True)
    return rows


def write_csv(od, name, rows):
    with open(od / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m_min", "m_max", "rf", "rf_sd", "mmd2", "mmd2_sd", "total_shots", "shots_sd", "cap_frac"])
        for r in rows:
            w.writerow([r["m_min"], r["m_max"], f"{r['rf']:.4f}", f"{r['rf_sd']:.4f}",
                        f"{r['mmd2']:.5f}", f"{r['mmd2_sd']:.5f}",
                        f"{r['shots']:.0f}", f"{r['shots_sd']:.0f}", f"{r['cap']:.3f}"])
    print(f"      -> {od/name}")


def fig_sweep(od, rows, xkey, xlabel, fname):
    """text：rf / MMD² / text shots。text（M_min、M_max）text。"""
    x = [r[xkey] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6), layout="constrained")
    for a, (k, sd, yl) in zip(ax, [("rf", "rf_sd", "resolved fraction"),
                                   ("mmd2", "mmd2_sd", r"MMD$^2$"),
                                   ("shots", "shots_sd", "shots per sample")]):
        y = np.array([r[k] for r in rows]); e = np.array([r[sd] for r in rows])
        a.errorbar(x, y, yerr=e, marker="o", ms=6, lw=2, color=BLUE, capsize=3)
        a.set_xscale("log", base=2)
        a.set_xticks(x); a.set_xticklabels([str(v) for v in x])
        a.set_xlabel(xlabel); a.set_ylabel(yl); a.grid(alpha=0.3)
        if k == "shots":
            a.set_yscale("log")
    fig.savefig(od / fname, **SAVE); plt.close(fig)
    print(f"      -> {od/fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["mmin", "mmax", "both"])
    ap.add_argument("--outdir", default="results/final_gmm_diagW_pi")
    ap.add_argument("--mode", default="diagW", choices=["diagW", "fullW", "fixed24I"])
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--rep", type=int, default=20)
    ap.add_argument("--gen-samples", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dealias-max", type=float, default=math.pi, help="diagW text π")
    ap.add_argument("--mmin-list", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--mmax-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--fix-mmax", type=int, default=NO_CAP,
                    help="text M_min text M_max（text，text）")
    ap.add_argument("--fix-mmin", type=int, default=2, help="text M_max text M_min")
    args = ap.parse_args()

    M.configure_gpu(8)
    MAIN.DEALIAS_SCALE = args.dealias_max / (0.6 * M.N + 0.15 * M.N * M.N)
    print(f"[DEALIAS ON] scale={MAIN.DEALIAS_SCALE:.4f} (max≈{args.dealias_max:.2f} rad)", flush=True)

    xis = G.psd_sqrt(G.projection(M.N))[0]
    C = G.mixture_centers(M.N)
    target = G.sample_target(np.random.default_rng(M.SEED + 99), args.gen_samples, C, M.S0, xis)
    gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))

    root = Path(args.outdir)
    theta = MAIN._load_theta(root, args.mode)
    od = root / args.mode / "bounds"; od.mkdir(parents=True, exist_ok=True)
    print(f"[bounds] mode={args.mode} rep={args.rep} seeds={args.seeds} -> {od}", flush=True)

    t0 = time.time()
    if args.part in ("mmin", "both"):
        print(f"\n=== text M_min（M_max text = {args.fix_mmax}）===", flush=True)
        rows = sweep(theta, args.mode, args, C, target, gamma,
                     args.mmin_list, [args.fix_mmax] * len(args.mmin_list), "M_min")
        write_csv(od, f"bounds_mmin_{args.mode}.csv", rows)
        fig_sweep(od, rows, "m_min", r"lower bound $M_{\min}$", f"bounds_mmin_{args.mode}.png")

    if args.part in ("mmax", "both"):
        print(f"\n=== text M_max（M_min text = {args.fix_mmin}）===", flush=True)
        rows = sweep(theta, args.mode, args, C, target, gamma,
                     [args.fix_mmin] * len(args.mmax_list), args.mmax_list, "M_max")
        write_csv(od, f"bounds_mmax_{args.mode}.csv", rows)
        fig_sweep(od, rows, "m_max", r"upper bound $M_{\max}$", f"bounds_mmax_{args.mode}.png")

    print(f"\n[bounds] text，text {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
