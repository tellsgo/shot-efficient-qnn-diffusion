"""run_T1000_mainline.py — text GMM text（T=1000, 3-qubit text, GPU/jax）。

【text】。text（import，text）text：
  run_quantum_3q_gpu  (M)  text jax text / text / text
  noise_budgeted_generation (G)  DDIM / shot text / MMD text numpy text
  run_quantum_x0 (Q)  text / text / min_shots_W / residual_W / resolved_frac

text --part，text（text，text compare text infer）：
  train     text（fullW / diagW text）
  infer     text：<mode>/infer/ text T{Ti}/ text，text
              scatter.png（text，text rf/MMD/shots）
              denoise_full.png（text）+ denoise_4.png（text 4 text）
              shots_per_step.png（text shots，text shots）+ metrics.csv
            text：summary_rf/mmd/shots.png + summary_scatters.png + summary.csv
  triptych  ch1  Target | High-shot | Adaptive text
  uniform   ch3b adaptive vs uniform text（rf | MMD² text）
  eta       ch4  η∈{0,0.2,0.5,0.8,1} text rf / MMD / text shots
  compare   text infer/summary.csv → fullW vs diagW text

text results/final_gmm_mainline/<mode>/...。text、text、bbox='tight'。
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quantum_3q_gpu as M
import noise_budgeted_generation as G
import run_quantum_x0 as Q
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.titleweight": "normal", "figure.autolayout": False,
})
BLUE = "#2970b0"; GREY = "#555555"
SAVE = dict(dpi=150, bbox_inches="tight")





DEALIAS_SCALE = None
_ORIG_MAKE_STATE = Q.make_state
def _make_state_dealias(q, tau):
    if DEALIAS_SCALE is not None and Q.EMBEDDING == "hybrid" and Q.TIME_ON:
        st = np.sqrt(np.maximum(q, 1e-12)).astype(complex)
        idx = np.arange(1, Q.N_DATA + 1)
        ph = DEALIAS_SCALE * (0.6 * idx * tau[:, None] + 0.15 * idx * idx * tau[:, None])
        st = st * np.exp(1j * ph)
        return st / np.linalg.norm(st, axis=1, keepdims=True)
    return _ORIG_MAKE_STATE(q, tau)
Q.make_state = _make_state_dealias



def generate(theta, mode, T_train, T_infer, seed, nsamp, eta=1.0,
             shot_mode="adaptive", uniform_M=None, high_M=80000,
             m_min=2, m_max=200000, record_traj=False):
    """text。shot_mode: adaptive / uniform(text uniform_M) / high(high_M/text)。
    text (x_gen, mean_total_shots, per_step_mean, traj_or_None, saturated_frac)。"""
    N = M.N; A = M.A
    Wr, br = M.readout_numpy(theta, mode)
    wr = np.asarray(theta["wr"])
    betas = np.linspace(M.BMAX, M.BMIN, T_train + 1)
    idx = np.unique(np.linspace(0, T_train, T_infer + 1).round().astype(int))
    rng = np.random.default_rng(seed)
    P = G.projection(N); xis = G.psd_sqrt(P)[0]
    x = rng.normal(size=(nsamp, N)) @ xis.T
    total = np.zeros(nsamp); per_step = []
    traj = [x.copy()] if record_traj else None
    sat_hits = 0; n_steps = len(idx) - 1
    for k in range(n_steps):
        i0, i1 = idx[k], idx[k + 1]
        be, ben = float(betas[i0]), float(betas[i1])
        al = math.sqrt(max(1 - be**2, 1e-12)); aln = math.sqrt(max(1 - ben**2, 1e-12))
        sig, dirc, ct = G.compute_ddim_params(al, be, aln, ben, eta)
        qb, _ = G.chi_to_p(x, A)
        tau = i0 / max(T_train - 1, 1) * np.ones(nsamp)
        p = np.array(M.jcirc(jnp.asarray(wr), jnp.asarray(Q.make_state(qb, tau)),
                             jnp.asarray(Q.make_feat(qb, tau))))
        p = np.clip(p.astype(np.float64), 0.0, None); p = p / p.sum(axis=1, keepdims=True)
        kc = aln - dirc * al / max(be, 1e-9)
        if shot_mode == "adaptive":
            if sig < 1e-6:
                sh = np.full(nsamp, m_max, dtype=np.int64); sat_hits += nsamp
            else:
                req = Q.min_shots_W(p, Wr, kc, sig, P)
                sh = np.clip(np.ceil(1.10 * req), m_min, m_max).astype(np.int64)
                sat_hits += int(np.sum(sh >= m_max))
        elif shot_mode == "uniform":
            sh = np.full(nsamp, max(int(uniform_M), 2), dtype=np.int64)
        elif shot_mode == "high":
            sh = np.full(nsamp, int(high_M), dtype=np.int64)
        else:
            raise ValueError(shot_mode)
        per_step.append(float(np.mean(sh)))
        ph = G.empirical_born_readout(rng, p, sh)
        x0h = G.project_zero_sum(ph @ Wr + br)
        ev = (x - al * x0h) / max(be, 1e-9)
        res = Q.residual_W(rng, p, sh, Wr, kc, sig, P)
        x = G.project_zero_sum(np.clip(aln * x0h + dirc * ev + sig * res, -A, A))
        total += sh
        if record_traj: traj.append(x.copy())
    sat_frac = sat_hits / max(nsamp * n_steps, 1)
    return x, float(np.mean(total)), per_step, traj, sat_frac



def pca_basis(C, target):
    cm = C.mean(0); _, _, vh = np.linalg.svd(C - cm, full_matrices=False); B2 = vh[:2].T
    PT = (target - cm) @ B2
    mg = 0.45 * max(np.ptp(PT[:, 0]), np.ptp(PT[:, 1]))
    xl = (PT[:, 0].min() - mg, PT[:, 0].max() + mg); yl = (PT[:, 1].min() - mg, PT[:, 1].max() + mg)
    return cm, B2, xl, yl


def _panel(ax, X, C, cm, B2, xl, yl, color, label, fontsize=17):
    """text；label text【text】（xlabel），text。"""
    P2 = (X - cm) @ B2; cp = (C - cm) @ B2
    ax.scatter(P2[:, 0], P2[:, 1], s=10, c=color, alpha=0.45, edgecolors="none")
    ax.scatter(cp[:, 0], cp[:, 1], s=230, facecolors="none", edgecolors="black", linewidths=2.2, zorder=5)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(label, fontsize=fontsize)


def fig_triptych(od, mode, target, x_high, sh_high, x_ad, rf_ad, sh_ad, C, basis):
    cm, B2, xl, yl = basis
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.0), layout="constrained")
    _panel(ax[0], target, C, cm, B2, xl, yl, GREY, "Target", fontsize=30)
    _panel(ax[1], x_high, C, cm, B2, xl, yl, BLUE, f"High-shot{chr(10)}({sh_high:.0f} shots)", fontsize=30)
    _panel(ax[2], x_ad, C, cm, B2, xl, yl, BLUE, f"Adaptive{chr(10)}(ours, 303 shots)", fontsize=30)
    fig.savefig(od / f"ch1_triptych_{mode}.png", **SAVE); plt.close(fig)


def _draw_traj(traj, C, basis, picks, labels, figsize, fpath, fontsize=17):
    cm, B2, xl, yl = basis
    fig, ax = plt.subplots(1, len(picks), figsize=figsize, layout="constrained")
    if len(picks) == 1: ax = [ax]
    for a, i, lb in zip(ax, picks, labels):
        _panel(a, traj[i], C, cm, B2, xl, yl, BLUE, lb, fontsize=fontsize)
    fig.savefig(fpath, **SAVE); plt.close(fig)


def fig_denoise4(tdir, traj, C, basis):
    """text 4 text（text 4 text）。"""
    n = len(traj) - 1
    picks = sorted(set([0, round(n / 3), round(2 * n / 3), n]))
    while len(picks) < 4: picks.append(picks[-1])
    lab = ["Noise (start)", "Intermediate", "Intermediate", "Denoised (final)"]
    labels = [f"{l}\n(step {i}/{n})" for l, i in zip(lab, picks)]
    _draw_traj(traj, C, basis, picks, labels, (20.5, 5.4), tdir / "denoise_4.png", fontsize=30)


def fig_denoise_full(tdir, traj, C, basis, maxcols=12):
    """text（text ≤maxcols text，text maxcols text）。"""
    n = len(traj) - 1
    picks = list(range(n + 1)) if n + 1 <= maxcols else \
        sorted(set(np.linspace(0, n, maxcols).round().astype(int)))
    labels = [f"step {i}/{n}" for i in picks]
    _draw_traj(traj, C, basis, picks, labels, (max(2.6 * len(picks), 6), 5.0),
               tdir / "denoise_full.png")


def fig_result(tdir, x, target, C, basis, rf, mmd, sh, Ti, fname="scatter.png"):
    """text：Target | Generated，rf/MMD/shots text（text）。"""
    cm, B2, xl, yl = basis
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 6.6))
    _panel(ax[0], target, C, cm, B2, xl, yl, GREY, "Target")
    _panel(ax[1], x, C, cm, B2, xl, yl, BLUE, f"Generated ($T_{{\\mathrm{{infer}}}}={Ti}$)")
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.5, 0.04, f"rf = {rf:.3f}      MMD$^2$ = {mmd:.4f}      total shots = {sh:.0f}",
             ha="center", fontsize=16)
    fig.savefig(tdir / fname, **SAVE); plt.close(fig)


def fig_shots_per_step(tdir, per, total, Ti):
    """text shots text（T text shots text【text】）。"""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    steps = np.arange(1, len(per) + 1)
    ax.plot(steps, per, "-o", color=BLUE, lw=2.2, ms=7)
    ax.set_ylabel("Mean shots at this step")
    ax.set_xlabel(f"Reverse diffusion step\n$T_{{\\mathrm{{infer}}}}={Ti}$      total shots = {total:.0f}")
    ax.grid(alpha=0.3)
    fig.savefig(tdir / "shots_per_step.png", **SAVE); plt.close(fig)


def fig_summary_scatters(od, scatters, C, basis):
    """text（text T / rf / MMD）。"""
    cm, B2, xl, yl = basis; k = len(scatters)
    fig, ax = plt.subplots(1, k, figsize=(5 * k, 5.8))
    if k == 1: ax = [ax]
    for a, (Ti, x, rf, mmd, sh) in zip(ax, scatters):
        _panel(a, x, C, cm, B2, xl, yl, BLUE,
               f"$T_{{\\mathrm{{infer}}}}={Ti}$\nrf={rf:.3f}   MMD$^2$={mmd:.4f}", fontsize=15)
    fig.savefig(od / "summary_scatters.png", **SAVE); plt.close(fig)


def fig_row_scatters(outpath, target, entries, C, basis):
    """text：Target + text（text rf/MMD/shots），text。
    entries: [(title, X, rf, mmd, sh), ...]。text W text / text。"""
    cm, B2, xl, yl = basis
    panels = [("Target", target, None, None, None)] + list(entries)
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6.0), layout="constrained")
    if len(panels) == 1: ax = [ax]
    for a, (title, X, rf, mmd, sh) in zip(ax, panels):
        color = GREY if title == "Target" else BLUE
        lbl = title if rf is None else f"{title}\nrf={rf:.3f}   MMD$^2$={mmd:.4f}"
        _panel(a, X, C, cm, B2, xl, yl, color, lbl, fontsize=15)
    fig.savefig(outpath, **SAVE); plt.close(fig)


def fig_metric(od, xs, ys, ylab, fname, ylog=False):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, ys, "-o", color=BLUE, lw=2.4, ms=9)
    ax.set_xlabel("Inference steps $T_{\\mathrm{infer}}$"); ax.set_ylabel(ylab)
    if ylog: ax.set_yscale("log")
    ax.grid(alpha=0.3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=12)
    fig.savefig(od / fname, **SAVE); plt.close(fig)


def fig_pareto(od, mode, uni_shots, uni_rf, uni_mmd, ad_shots, ad_rf, ad_mmd):
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6), layout="constrained")
    ax[0].plot(uni_shots, uni_rf, "-o", color=GREY, lw=2.2, ms=8, label="Uniform shots")
    ax[0].plot([ad_shots], [ad_rf], "*", color=BLUE, ms=26, label="Adaptive (ours)", zorder=6)
    ax[0].axhline(ad_rf, color=BLUE, ls="--", lw=1.3, alpha=0.6)
    ax[0].set_xscale("log"); ax[0].set_ylabel("Resolved fraction  rf"); ax[0].legend(loc="lower right")
    ax[1].plot(uni_shots, uni_mmd, "-o", color=GREY, lw=2.2, ms=8, label="Uniform shots")
    ax[1].plot([ad_shots], [ad_mmd], "*", color=BLUE, ms=26, label="Adaptive (ours)", zorder=6)
    ax[1].axhline(ad_mmd, color=BLUE, ls="--", lw=1.3, alpha=0.6)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylabel("MMD$^2$"); ax[1].legend(loc="best")
    for a in ax: a.set_xlabel("Total shots"); a.grid(alpha=0.3)
    fig.savefig(od / f"ch3b_adaptive_vs_uniform_{mode}.png", **SAVE); plt.close(fig)


def fig_eta(od, mode, etas, rf, mmd, shots, fname):
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 5.4), layout="constrained")
    xt = [str(e) for e in etas]
    ax[0].plot(xt, rf, "-o", color=BLUE, lw=2.4, ms=9); ax[0].set_ylabel("Resolved fraction  rf")
    ax[1].plot(xt, mmd, "-o", color=BLUE, lw=2.4, ms=9); ax[1].set_ylabel("MMD$^2$")
    ax[2].plot(xt, shots, "-o", color=BLUE, lw=2.4, ms=9); ax[2].set_ylabel("Total shots"); ax[2].set_yscale("log")
    for a in ax: a.set_xlabel("DDIM stochasticity  $\\eta$"); a.grid(alpha=0.3)
    fig.savefig(od / fname, **SAVE); plt.close(fig)


CMP_COL = {"fullW": "#2970b0", "diagW": "#c85a11", "fixed24I": "#555555",
           "probability": "#2a9d4a", "amplitude": "#9b3fc4"}
CMP_LABEL = {"fullW": "full W (trainable)", "diagW": "diag W (trainable)", "fixed24I": "fixed 24I (no train)"}

def fig_compare(root, infers, data, fname, order):
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 5.4), layout="constrained")
    for mode in order:
        if mode not in data: continue
        d = data[mode]; c = CMP_COL.get(mode, "#777"); lb = CMP_LABEL.get(mode, mode)
        ax[0].plot(infers, d["rf"], "-o", color=c, lw=2.4, ms=8, label=lb)
        ax[1].plot(infers, d["mmd"], "-o", color=c, lw=2.4, ms=8, label=lb)
        ax[2].plot(infers, d["shots"], "-o", color=c, lw=2.4, ms=8, label=lb)
    ax[0].set_ylabel("Resolved fraction  rf"); ax[1].set_ylabel("MMD$^2$"); ax[1].set_yscale("log")
    ax[2].set_ylabel("Total shots"); ax[2].set_yscale("log")
    for a in ax:
        a.set_xlabel("Inference steps $T_{\\mathrm{infer}}$"); a.grid(alpha=0.3); a.legend()
    fig.savefig(root / fname, **SAVE); plt.close(fig)


def fig_enc_bars(od, results, nseed, fname="encoding_compare.png"):
    """text：text rf / MMD² text（text±text；x text）。
    results text = (name, emb, t, rf_mean, rf_std, mmd_mean, mmd_std, sh)。"""
    names = [r[0] for r in results]
    rf = [r[3] for r in results]; rfe = [r[4] for r in results]
    mmd = [r[5] for r in results]; mmde = [r[6] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(max(2.2 * len(names), 12), 6.2), layout="constrained")
    ax[0].bar(x, rf, yerr=rfe, color=BLUE, capsize=5, error_kw=dict(ecolor="#333", lw=1.5)); ax[0].set_ylabel("Resolved fraction  rf")
    ax[1].bar(x, mmd, yerr=mmde, color=BLUE, capsize=5, error_kw=dict(ecolor="#333", lw=1.5)); ax[1].set_ylabel("MMD$^2$")
    for i, (v, e) in enumerate(zip(rf, rfe)): ax[0].text(i, v + e, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    for i, (v, e) in enumerate(zip(mmd, mmde)): ax[1].text(i, v + e, f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=25, ha="right"); a.grid(alpha=0.3, axis="y")
        a.set_xlabel(f"encoding cell (mean±std over {nseed} inference seeds)", fontsize=13)
    fig.savefig(od / fname, **SAVE); plt.close(fig)



def _load_theta(root, mode):
    d = np.load(root / mode / "train" / "weights.npz")
    return {k: d[k] for k in d.files if k != "mode"}


def part_train(root, mode, args, C, xis):
    td = root / mode / "train"
    M.train(td, C, xis, args.Ttrain, mode, args.layers, args.iters, args.samples,
            args.batch, args.lr, args.clip, args.resume, max(args.iters // 10, 1), dd=None, wreg=args.wreg)


def part_infer(root, mode, args, C, target, gamma):
    """text：text（text+text+text4text+textshots+csv），
    text（rf/MMD/shots text + text + summary.csv）。"""
    od = root / mode / "infer"; od.mkdir(parents=True, exist_ok=True)
    theta = _load_theta(root, mode); basis = pca_basis(C, target)
    rf, mmd, shots, scatters = [], [], [], []
    for Ti in args.infers:
        x, sh, per, traj, _ = generate(theta, mode, args.Ttrain, Ti, M.SEED + 202,
                                       args.gen_samples, eta=1.0, shot_mode="adaptive", record_traj=True)
        r = Q.resolved_frac(x, C); m = float(G.mmd2_rbf(x, target, gamma))
        rf.append(r); mmd.append(m); shots.append(sh); scatters.append((Ti, x, r, m, sh))
        tdir = od / f"T{Ti}"; tdir.mkdir(exist_ok=True)
        fig_result(tdir, x, target, C, basis, r, m, sh, Ti)
        fig_denoise_full(tdir, traj, C, basis)
        fig_denoise4(tdir, traj, C, basis)
        fig_shots_per_step(tdir, per, sh, Ti)
        with open(tdir / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["T_infer", "rf", "mmd2", "total_shots"]); w.writerow([Ti, f"{r:.4f}", f"{m:.5f}", f"{sh:.0f}"])
            w.writerow([]); w.writerow(["reverse_step", "mean_shots"])
            for i, v in enumerate(per, 1): w.writerow([i, f"{v:.1f}"])
        print(f"    [{mode}] T_infer={Ti:>3d}  rf={r:.3f}  MMD2={m:.4f}  shots={sh:.0f}  -> {tdir.name}/", flush=True)
    with open(od / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["T_infer", "rf", "mmd2", "total_shots"])
        for Ti, r, m, s in zip(args.infers, rf, mmd, shots): w.writerow([Ti, f"{r:.4f}", f"{m:.5f}", f"{s:.0f}"])
    fig_metric(od, args.infers, rf, "Resolved fraction  rf", "summary_rf.png")
    fig_metric(od, args.infers, mmd, "MMD$^2$", "summary_mmd.png", ylog=True)
    fig_metric(od, args.infers, shots, "Total shots", "summary_shots.png", ylog=True)
    fig_summary_scatters(od, scatters, C, basis)


def part_triptych(root, mode, args, C, target):
    od = root / mode; od.mkdir(parents=True, exist_ok=True); theta = _load_theta(root, mode)
    basis = pca_basis(C, target)
    x_ad, sh_ad, _, _, _ = generate(theta, mode, args.Ttrain, args.rep, M.SEED + 202,
                                    args.gen_samples, eta=1.0, shot_mode="adaptive")
    rf_ad = Q.resolved_frac(x_ad, C)
    x_hi, sh_hi, _, _, _ = generate(theta, mode, args.Ttrain, args.rep, M.SEED + 303,
                                    args.gen_samples, eta=1.0, shot_mode="high", high_M=args.high_M)
    fig_triptych(od, mode, target, x_hi, sh_hi, x_ad, rf_ad, sh_ad, C, basis)
    print(f"    [{mode}] triptych: adaptive rf={rf_ad:.3f} shots={sh_ad:.0f} | high shots={sh_hi:.0f}", flush=True)


def part_uniform(root, mode, args, C, target, gamma):
    """uniform text shots text"text adaptive text(1×)"text，text adaptive text shots text 1..10 text，
    text adaptive text。text"text shot text adaptive text / uniform text"。"""
    od = root / mode; od.mkdir(parents=True, exist_ok=True); theta = _load_theta(root, mode)
    x_ad, sh_ad, per_ad, _, _ = generate(theta, mode, args.Ttrain, args.rep, M.SEED + 202,
                                         args.gen_samples, eta=1.0, shot_mode="adaptive")
    rf_ad = Q.resolved_frac(x_ad, C); mmd_ad = float(G.mmd2_rbf(x_ad, target, gamma))
    n_steps = len(per_ad)
    print(f"    [{mode}] adaptive: total_shots={sh_ad:.0f} ({n_steps} steps) rf={rf_ad:.3f} MMD2={mmd_ad:.4f}", flush=True)
    us, ur, um, Ms = [], [], [], []
    for k in args.uniform_mult:
        Mu = max(2, int(round(k * sh_ad / max(n_steps, 1))))
        xu, su, _, _, _ = generate(theta, mode, args.Ttrain, args.rep, M.SEED + 404,
                                   args.gen_samples, eta=1.0, shot_mode="uniform", uniform_M=Mu)
        us.append(su); ur.append(Q.resolved_frac(xu, C)); um.append(float(G.mmd2_rbf(xu, target, gamma))); Ms.append(Mu)
        print(f"    [{mode}] uniform {k:>2d}× (M={Mu}, shots={su:.0f})  rf={ur[-1]:.3f}  MMD2={um[-1]:.4f}", flush=True)
    with open(od / f"uniform_{mode}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["mult_of_adaptive", "uniform_M", "total_shots", "rf", "mmd2"])
        for k, Mu, s, r, m in zip(args.uniform_mult, Ms, us, ur, um):
            w.writerow([k, Mu, f"{s:.0f}", f"{r:.4f}", f"{m:.5f}"])
    fig_pareto(od, mode, us, ur, um, sh_ad, rf_ad, mmd_ad)


def part_eta(root, mode, args, C, target, gamma):
    """text：ch4_eta_full（text η）+ ch4_eta_repr（text，text；text）。"""
    od = root / mode; od.mkdir(parents=True, exist_ok=True); theta = _load_theta(root, mode)
    er, em, es, esat = [], [], [], []
    for eta in args.etas:
        xe, se, _, _, sat = generate(theta, mode, args.Ttrain, args.rep, M.SEED + 505,
                                     args.gen_samples, eta=eta, shot_mode="adaptive")
        er.append(Q.resolved_frac(xe, C)); em.append(float(G.mmd2_rbf(xe, target, gamma))); es.append(se); esat.append(sat)
        print(f"    [{mode}] eta={eta}  rf={er[-1]:.3f}  MMD2={em[-1]:.4f}  shots={se:.0f}  cap={sat:.2f}", flush=True)
    with open(od / f"eta_{mode}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["eta", "rf", "mmd2", "total_shots", "cap_frac"])
        for eta, r, m, s, sa in zip(args.etas, er, em, es, esat): w.writerow([eta, f"{r:.4f}", f"{m:.5f}", f"{s:.0f}", f"{sa:.3f}"])
    fig_eta(od, mode, args.etas, er, em, es, f"ch4_eta_full_{mode}.png")
    idx = [i for i, e in enumerate(args.etas) if any(abs(e - s) < 1e-9 for s in args.etas_subset)]
    if len(idx) >= 2:
        se_ = [args.etas[i] for i in idx]
        fig_eta(od, mode, se_, [er[i] for i in idx], [em[i] for i in idx], [es[i] for i in idx],
                f"ch4_eta_repr_{mode}.png")
        print(f"    [{mode}] text η={se_} -> ch4_eta_repr_{mode}.png", flush=True)
    else:
        print(f"    [{mode}] text：--etas-subset {args.etas_subset} text，text repr text", flush=True)


def part_compare(root, args, C, target, gamma):
    """W text：text = rep（text 20），text**text**
    Target | text mode text（text rf/MMD），text mode text。
    text rf/MMD/shots text（text mode text infer/summary.csv text）。
    text：--compare-modes {fixed24I diagW fullW}/{fixed24I fullW}/{fixed24I diagW}。"""
    modes = args.compare_modes; basis = pca_basis(C, target); tag = "_".join(modes)

    entries = []
    for mode in modes:
        wp = root / mode / "train" / "weights.npz"
        if not wp.exists():
            print(f"  [compare] text {wp}，text {mode}", flush=True); continue
        d = np.load(wp); theta = {k: d[k] for k in d.files if k != "mode"}
        x, sh, _, _, _ = generate(theta, mode, args.Ttrain, args.rep, M.SEED + 202,
                                  args.gen_samples, eta=1.0, shot_mode="adaptive")
        rf = Q.resolved_frac(x, C); mmd = float(G.mmd2_rbf(x, target, gamma))
        entries.append((CMP_LABEL.get(mode, mode), x, rf, mmd, sh))
        print(f"    [compare T{args.rep}] {mode}: rf={rf:.3f}  MMD2={mmd:.4f}  shots={sh:.0f}", flush=True)
    if len(entries) >= 2:
        fig_row_scatters(root / f"compare_{tag}_T{args.rep}.png", target, entries, C, basis)
        print(f"  [compare] text -> compare_{tag}_T{args.rep}.png", flush=True)

    data = {}; infers = None
    for mode in modes:
        p = root / mode / "infer" / "summary.csv"
        if not p.exists(): continue
        rf, mmd, shots, xs = [], [], [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                xs.append(int(row["T_infer"])); rf.append(float(row["rf"]))
                mmd.append(float(row["mmd2"])); shots.append(float(row["total_shots"]))
        infers = xs; data[mode] = {"rf": rf, "mmd": mmd, "shots": shots}
    if len(data) >= 2:
        fig_compare(root, infers, data, f"compare_{tag}_metrics.png", order=modes)
        print(f"  [compare] text（text）-> compare_{tag}_metrics.png", flush=True)


def part_encoding(root, args, C, xis, target, gamma):
    """text：text = --enc-readout（text fullW；text）。
    text 6 text（3 text × text on/off），text train + text T=rep text，text：
      ablation_encoding_<readout>/encoding_compare.png   6 text rf/MMD² text
      ablation_encoding_<readout>/encoding_scatters.png  6 text（text rf/MMD）
      ablation_encoding_<readout>/<cell>/scatter.png     text
      ablation_encoding_<readout>/encoding.csv
    text hybrid_off ≡ probability_off（text），text。text = hybrid_on。"""
    global DEALIAS_SCALE
    ro = args.enc_readout; rep = args.rep; ev = args.enc_eval_only
    dscale = args.dealias_max / (0.6 * M.N + 0.15 * M.N * M.N)
    od = root / f"ablation_encoding_{ro}"; od.mkdir(parents=True, exist_ok=True)
    basis = pca_basis(C, target); results = []; scat = []
    cells = [(e, t) for e in args.enc_embeddings for t in args.enc_times]
    print(f"[encoding readout={ro} rep={rep} {'EVAL-ONLY(text)' if ev else 'TRAIN+EVAL'}]", flush=True)
    for emb, t in cells:
        ton = (t == "on"); name = f"{emb}_{t}"; cd = od / name; cd.mkdir(exist_ok=True)

        if emb == "hybridD" or (emb == "hybrid" and args.enc_hybrid_dealias):
            Q.set_encoding("hybrid", ton); DEALIAS_SCALE = dscale
            tag = "hybridD" if emb == "hybridD" else "hybrid=text"
            print(f"    ({tag}: text scale={dscale:.3f}, text≈{args.dealias_max:.2f} rad)", flush=True)
        else:
            Q.set_encoding(emb, ton); DEALIAS_SCALE = None
        print(f"\n----- encoding {name} (readout={ro}) -----", flush=True)
        if not ev:
            M.train(cd / "train", C, xis, args.Ttrain, ro, args.layers, args.iters, args.samples,
                    args.batch, args.lr, args.clip, args.resume, max(args.iters // 10, 1), dd=None, wreg=args.wreg)
        wp = cd / "train" / "weights.npz"
        if not wp.exists():
            print(f"    [skip] text {wp}（eval-only text）", flush=True); continue
        d = np.load(wp); theta = {k: d[k] for k in d.files if k != "mode"}

        rfs, mmds, shs = [], [], []; x0 = None
        for s in range(args.enc_seeds):
            x, sh, _, _, _ = generate(theta, ro, args.Ttrain, rep, M.SEED + 202 + s,
                                      args.gen_samples, eta=1.0, shot_mode="adaptive")
            rfs.append(Q.resolved_frac(x, C)); mmds.append(float(G.mmd2_rbf(x, target, gamma))); shs.append(sh)
            if x0 is None: x0 = x
        rf, rfe = float(np.mean(rfs)), float(np.std(rfs))
        mmd, mmde = float(np.mean(mmds)), float(np.std(mmds)); sh = float(np.mean(shs))
        fig_result(cd, x0, target, C, basis, rf, mmd, sh, rep, fname=f"scatter_T{rep}.png")
        results.append((name, emb, t, rf, rfe, mmd, mmde, sh)); scat.append((name, x0, rf, mmd, sh))
        print(f"    [enc {name}] rf={rf:.3f}±{rfe:.3f}  MMD2={mmd:.4f}±{mmde:.4f}  shots={sh:.0f}  (n={args.enc_seeds}: rf={[f'{r:.3f}' for r in rfs]})", flush=True)
    Q.set_encoding("hybrid", True); DEALIAS_SCALE = None
    with open(od / f"encoding_T{rep}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cell", "embedding", "time", "rf_mean", "rf_std", "mmd2_mean", "mmd2_std", "total_shots", "n_seeds"])
        for name, emb, t, rf, rfe, mmd, mmde, sh in results:
            w.writerow([name, emb, t, f"{rf:.4f}", f"{rfe:.4f}", f"{mmd:.5f}", f"{mmde:.5f}", f"{sh:.0f}", args.enc_seeds])
    fig_enc_bars(od, results, args.enc_seeds, fname=f"encoding_compare_T{rep}.png")
    fig_row_scatters(od / f"encoding_scatters_T{rep}.png", target, scat, C, basis)
    print(f"\n  [encoding readout={ro} rep={rep}] {len(results)} text（text {args.enc_seeds} text）-> {od}/ (*_T{rep}.*)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["train", "infer", "triptych", "uniform", "eta", "compare", "encoding"])
    ap.add_argument("--modes", nargs="+", default=["fullW"],
                    help="fullW / diagW / fixed24I；per-mode part text（W text fixed24I）")
    ap.add_argument("--compare-modes", nargs="+", default=["fixed24I", "diagW", "fullW"],
                    help="compare text：text mode text，text compare_<text>.png。"
                         "W text {fixed24I diagW fullW}/{fixed24I fullW}/{fixed24I diagW}")
    ap.add_argument("--enc-embeddings", nargs="+", default=["amplitude", "probability", "hybrid"],
                    help="encoding text")
    ap.add_argument("--enc-times", nargs="+", default=["on", "off"], help="encoding text on/off")
    ap.add_argument("--enc-readout", default="fullW", choices=["fullW", "diagW", "fixed24I"],
                    help="encoding text；text（fullW / diagW）")
    ap.add_argument("--enc-seeds", type=int, default=3,
                    help="encoding text（text=text），text±text")
    ap.add_argument("--enc-eval-only", action="store_true",
                    help="text --rep text（text，text）；text *_T{rep}.* text rep text")
    ap.add_argument("--enc-hybrid-dealias", action="store_true",
                    help="text 'hybrid' text（text hybrid，text --dealias-max）；"
                         "text amplitude/probability/hybrid text，text hybridD")
    ap.add_argument("--dealias-max", type=float, default=2 * math.pi,
                    help="hybridD(text hybrid)text(rad)，text τ=1、idx=N text=text；text 2π text")
    ap.add_argument("--outdir", default="results/final_gmm_mainline")
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--wreg", type=float, default=0.0)
    ap.add_argument("--infers", type=int, nargs="+", default=[5, 10, 15, 20, 50, 100])
    ap.add_argument("--rep", type=int, default=20, help="text（triptych/uniform/eta text；text0.9）")
    ap.add_argument("--etas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    help="η text（0→1 text 0.1）。η=0 text、text∝1/σ²text，shots text；"
                         "shots text ∝1/η² text（η=0 text），text η text")
    ap.add_argument("--etas-subset", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0],
                    help="text η text（text --etas text），text ch4_eta_repr text；text+text")
    ap.add_argument("--uniform-mult", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    help="uniform text shots text adaptive text shots text（1×=text adaptive text，text 10×）")
    ap.add_argument("--high-M", type=int, default=80000)
    ap.add_argument("--gen-samples", type=int, default=3000,
                    help="text；text MMD text（1500→3000 text）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dealias", action="store_true",
                    help="text hybrid（text train/infer/triptych/uniform/eta text）。"
                         "text；text --outdir text")
    args = ap.parse_args()

    global DEALIAS_SCALE
    M.configure_gpu(8)
    if args.dealias:
        DEALIAS_SCALE = args.dealias_max / (0.6 * M.N + 0.15 * M.N * M.N)
        print(f"[DEALIAS ON] text hybrid：text scale={DEALIAS_SCALE:.3f}（max≈{args.dealias_max:.2f} rad）", flush=True)
    xis = G.psd_sqrt(G.projection(M.N))[0]; C = G.mixture_centers(M.N)
    target = G.sample_target(np.random.default_rng(M.SEED + 99), args.gen_samples, C, M.S0, xis)
    gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))
    root = Path(args.outdir); root.mkdir(parents=True, exist_ok=True)
    print(f"[part={args.part}] modes={args.modes} T={args.Ttrain} rep={args.rep} -> {root}", flush=True)
    t0 = time.time()
    if args.part == "compare":
        part_compare(root, args, C, target, gamma)
    elif args.part == "encoding":
        part_encoding(root, args, C, xis, target, gamma)
    else:
        for mode in args.modes:
            print(f"\n===== {args.part} / {mode} =====", flush=True)
            if args.part == "train":       part_train(root, mode, args, C, xis)
            elif args.part == "infer":     part_infer(root, mode, args, C, target, gamma)
            elif args.part == "triptych":  part_triptych(root, mode, args, C, target)
            elif args.part == "uniform":   part_uniform(root, mode, args, C, target, gamma)
            elif args.part == "eta":       part_eta(root, mode, args, C, target, gamma)
    print(f"\n[part={args.part}] text，text {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
