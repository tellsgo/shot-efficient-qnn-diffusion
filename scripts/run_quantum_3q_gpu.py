"""run_quantum_3q_gpu.py — 3-qubit text diffusion text GPU text（jax text），text。

text：import text run_quantum_x0(Q) + noise_budgeted_generation(G)，**text numpy text**
（build_dataset / make_state / make_feat / chi_to_p / min_shots_W / residual_W / empirical_born_readout /
project_zero_sum / resolved_frac …）。**text + text jax + jit**（GPU ~100× text）。
text，text 3-qubit text（jcirc text Q._circ text）。

text --readout：
  fixed24I : text 24I（text）      x̂0 = zs(24p − 3)
  fixedI   : text I                      x̂0 = zs(p)            （text，text）
  fullW    : text 8×8 W（init 24I）  x̂0 = zs(pW + b)      （text，≈0.99）
  diagW    : text 8 text（init 24） x̂0 = zs(p⊙wd + b)   （text，text 3-qubit）

text quddpm_gpu，text，text。
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from datetime import date
from pathlib import Path
import numpy as _np
import jax, jax.numpy as jnp
from jax import jit, value_and_grad
import jax.tree_util as jtu
import pennylane as qml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_budgeted_generation as G
import run_quantum_x0 as Q
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 400
A = 3.0
BMAX, BMIN = 0.95, 0.10
DATE = date.today().isoformat()

N = 8; NQ = 3; FDIM = 10; S0 = 0.12; GAIN = A * N
dev = None; jcirc = None

def zsj(x):
    return x - jnp.mean(x, axis=-1, keepdims=True)

def configure_gpu(dim):
    """text dim text：NQ=log2(dim)，text jax text，text Q text/text（Q.configure(dim)）。"""
    global N, NQ, FDIM, S0, dev, jcirc, GAIN
    N = dim; NQ = int(round(math.log2(dim)))
    assert 2 ** NQ == dim, "dim text 2 text"
    Q.configure(dim); Q.set_encoding("hybrid", True)
    S0 = Q.S0; FDIM = Q.FDIM; GAIN = A * N
    dev = qml.device("default.qubit", wires=NQ)
    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def _jc(wr, state, feat):
        qml.AmplitudeEmbedding(state, wires=range(NQ), normalize=True)
        L = wr.shape[0]
        for l in range(L):
            for q in range(NQ):
                qml.RY(jnp.sum(wr[l, q, 0] * feat, axis=-1), wires=q)
                qml.RZ(jnp.sum(wr[l, q, 1] * feat, axis=-1), wires=q)
            for q in range(NQ):
                qml.CNOT(wires=[q, (q + 1) % NQ])
        return qml.probs(wires=range(NQ))
    jcirc = _jc


def xhat_jax(theta, p, mode):
    if mode == "fixed24I": return zsj(A * N * p - A)
    if mode == "fixedI":   return zsj(p)
    if mode == "fixedC":   return zsj(GAIN * p)
    if mode == "fullW":    return zsj(p @ theta["W"] + theta["b"])
    if mode == "diagW":    return zsj(p * theta["wd"] + theta["b"])
    raise ValueError(mode)

def init_theta(mode, L, scale):
    wr = jnp.asarray(0.1 * scale * _np.random.default_rng(SEED + 1).standard_normal((L, NQ, 2, FDIM)))
    if mode in ("fixed24I", "fixedI", "fixedC"):
        return {"wr": wr}
    if mode == "fullW":
        return {"wr": wr, "W": jnp.asarray(A * N * _np.eye(N)), "b": jnp.asarray(-A * _np.ones(N))}
    if mode == "diagW":
        return {"wr": wr, "wd": jnp.asarray(A * N * _np.ones(N)), "b": jnp.asarray(-A * _np.ones(N))}

def readout_numpy(theta, mode):
    """text：text numpy text (Wr, br)，text Q.min_shots_W / residual_W / text。"""
    if mode == "fixed24I": return A * N * _np.eye(N), -A * _np.ones(N)
    if mode == "fixedI":   return _np.eye(N), _np.zeros(N)
    if mode == "fixedC":   return GAIN * _np.eye(N), _np.zeros(N)
    if mode == "fullW":    return _np.array(theta["W"]), _np.array(theta["b"])
    if mode == "diagW":    return _np.diag(_np.array(theta["wd"])), _np.array(theta["b"])



def train(od, C, xis, T, mode, L, iters, samples, batch, lr, clip, resume, ckpt_every, dd=None, wreg=0.0):
    od.mkdir(parents=True, exist_ok=True)
    scale = (12.0 / L) ** 0.5
    lr0 = lr * scale
    Q.STEPS = T; Q.BMAX = BMAX; Q.BMIN = BMIN
    rng = _np.random.default_rng(SEED)
    q, ta, x0 = Q.build_dataset(rng, samples, C, xis, digits=dd)
    states = jnp.asarray(Q.make_state(q, ta)); feats = jnp.asarray(Q.make_feat(q, ta)); x0j = jnp.asarray(x0)
    var = float(_np.mean(x0 ** 2))

    def loss(theta, st, ft, y):
        return jnp.mean((xhat_jax(theta, jcirc(theta["wr"], st, ft), mode) - y) ** 2) / var
    def loss_train(theta, st, ft, y):
        r = loss(theta, st, ft, y)
        if wreg > 0 and mode == "diagW": r = r + wreg * jnp.mean(theta["wd"] ** 2)
        if wreg > 0 and mode == "fullW": r = r + wreg * jnp.mean(theta["W"] ** 2)
        return r
    eval_loss = jit(loss)

    @jit
    def step(theta, mw, vw, st, ft, y, it, lr_t):
        l, g = value_and_grad(loss_train)(theta, st, ft, y)
        gn = jnp.sqrt(sum(jnp.sum(v ** 2) for v in jtu.tree_leaves(g)))
        sc = jnp.where((clip > 0) & (gn > clip), clip / (gn + 1e-12), 1.0)
        g = jtu.tree_map(lambda x: x * sc, g)
        mw = jtu.tree_map(lambda m, gg: 0.9 * m + 0.1 * gg, mw, g)
        vw = jtu.tree_map(lambda v, gg: 0.999 * v + 0.001 * gg ** 2, vw, g)
        theta = jtu.tree_map(lambda p, m, v: p - lr_t * (m / (1 - 0.9 ** it)) / (jnp.sqrt(v / (1 - 0.999 ** it)) + 1e-8),
                             theta, mw, vw)
        return theta, mw, vw, l * var, gn

    ckpt = od / "checkpoint.npz"
    if resume and ckpt.exists():
        d = _np.load(ckpt, allow_pickle=True)
        theta = jtu.tree_map(jnp.asarray, d["theta"].item()); mw = jtu.tree_map(jnp.asarray, d["mw"].item())
        vw = jtu.tree_map(jnp.asarray, d["vw"].item()); it0 = int(d["it"]); hist = list(map(tuple, d["hist"]))
        print(f"  [resume] text iter {it0} text", flush=True)
    else:
        theta = init_theta(mode, L, scale); mw = jtu.tree_map(jnp.zeros_like, theta)
        vw = jtu.tree_map(jnp.zeros_like, theta); it0 = 0; hist = []
    lr_decay_at = int(iters * 0.6)
    rec_every = max(iters // 150, 1); log_step = max(iters // 10, 1); next_log = it0 + log_step
    mon = min(2048, samples); clip_hits = 0; t0 = time.time()
    print(f"  [train-3qGPU] mode={mode} L={L} T={T} iters={iters} samples={samples} lr0={lr0:.4f} "
          f"dev={jax.devices()[0].platform} text {time.strftime('%H:%M:%S')}", flush=True)

    for it in range(it0 + 1, iters + 1):
        lr_t = lr0 if it < lr_decay_at else lr0 * 0.25
        idx = _np.random.default_rng(it).integers(0, samples, size=batch)
        theta, mw, vw, l_b, gn = step(theta, mw, vw, states[idx], feats[idx], x0j[idx],
                                      jnp.float32(it), jnp.float32(lr_t))
        if float(gn) > clip and clip > 0: clip_hits += 1
        if it % rec_every == 0 or it == it0 + 1:
            hist.append((it, float(eval_loss(theta, states[:mon], feats[:mon], x0j[:mon])) * var))
        if it == it0 + 1 or it >= next_log:
            fl = float(eval_loss(theta, states[:mon], feats[:mon], x0j[:mon])) * var
            ch = f" clip%={100*clip_hits/max(it-it0,1):.0f}" if clip else ""
            print(f"    [{100*it/iters:3.0f}%] iter {it:>6d}/{iters}  chi-MSE={fl:.5f}{ch}  "
                  f"({time.time()-t0:.0f}s) {time.strftime('%H:%M:%S')}", flush=True)
            if it != it0 + 1: next_log += log_step
        if ckpt_every and it % ckpt_every == 0:
            _np.savez(ckpt, theta={k: _np.array(v) for k, v in theta.items()},
                      mw={k: _np.array(v) for k, v in mw.items()}, vw={k: _np.array(v) for k, v in vw.items()},
                      it=it, hist=_np.array(hist))

    final = hist[-1][1] if hist else float("nan")
    theta_np = {k: _np.array(v) for k, v in theta.items()}
    _np.savez(od / "weights.npz", **theta_np, mode=mode)
    _np.savez(ckpt, theta=theta_np, mw={k: _np.array(v) for k, v in mw.items()},
              vw={k: _np.array(v) for k, v in vw.items()}, it=iters, hist=_np.array(hist))
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4)); h = _np.array(hist)
        ax.plot(h[:, 0], h[:, 1]); ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("chi-MSE")
        ax.grid(alpha=0.3); ax.set_title(f"3q GPU {mode} L={L} T={T}")
        fig.tight_layout(); fig.savefig(od / "training_loss.png", dpi=130); plt.close()
    print(f"  [train-3qGPU] done chi-MSE={final:.4f} clip_hits={clip_hits} text {(time.time()-t0)/60:.1f}min", flush=True)
    return theta_np, final



def generate(theta, mode, T_train, T_infer, seed, nsamp):
    Wr, br = readout_numpy(theta, mode)
    wr = _np.asarray(theta["wr"])
    betas = _np.linspace(BMAX, BMIN, T_train + 1)
    idx = _np.unique(_np.linspace(0, T_train, T_infer + 1).round().astype(int))
    rng = _np.random.default_rng(seed)
    P = G.projection(N); xis = G.psd_sqrt(P)[0]
    x = rng.normal(size=(nsamp, N)) @ xis.T
    total = _np.zeros(nsamp); per_step = []
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        be, ben = float(betas[i0]), float(betas[i1])
        al = math.sqrt(max(1 - be**2, 1e-12)); aln = math.sqrt(max(1 - ben**2, 1e-12))
        sig, dirc, ct = G.compute_ddim_params(al, be, aln, ben, 1.0)
        qb, _ = G.chi_to_p(x, A)
        tau = i0 / max(T_train - 1, 1) * _np.ones(nsamp)
        p = _np.array(jcirc(jnp.asarray(wr), jnp.asarray(Q.make_state(qb, tau)), jnp.asarray(Q.make_feat(qb, tau))))
        p = _np.clip(p.astype(_np.float64), 0.0, None); p = p / p.sum(axis=1, keepdims=True)
        kc = aln - dirc * al / max(be, 1e-9)
        req = Q.min_shots_W(p, Wr, kc, sig, P)
        sh = _np.clip(_np.ceil(1.10 * req), 2, 10**9).astype(_np.int64)
        per_step.append(float(_np.mean(sh)))
        ph = G.empirical_born_readout(rng, p, sh)
        x0h = G.project_zero_sum(ph @ Wr + br)
        ev = (x - al * x0h) / max(be, 1e-9)
        res = Q.residual_W(rng, p, sh, Wr, kc, sig, P)
        x = G.project_zero_sum(_np.clip(aln * x0h + dirc * ev + sig * res, -A, A))
        total += sh
    return x, float(_np.mean(total)), per_step


def save_scatter(od, target, clouds, C):
    od.mkdir(parents=True, exist_ok=True)
    cm = C.mean(0); _, _, vh = _np.linalg.svd(C - cm, full_matrices=False); B2 = vh[:2].T
    def pr(X): return (X - cm) @ B2
    cp = pr(C); PT = pr(target); mg = 0.4 * max(_np.ptp(PT[:, 0]), _np.ptp(PT[:, 1]))
    xl = (PT[:, 0].min() - mg, PT[:, 0].max() + mg); yl = (PT[:, 1].min() - mg, PT[:, 1].max() + mg)
    panels = [("Target", target, 1.0, None)] + clouds
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (lbl, X, rf, sh) in zip(axes, panels):
        P2 = pr(X)
        ax.scatter(P2[:, 0], P2[:, 1], s=9, c="#555" if lbl == "Target" else "#2970b0", alpha=0.45, edgecolors="none")
        ax.scatter(cp[:, 0], cp[:, 1], s=200, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel(f"{lbl}\nrf={rf:.2f}" + ("" if sh is None else f"  {sh:.0f} shots"), fontsize=12)
    fig.suptitle("3-qubit (GPU/jax) diffusion", fontweight="bold")
    fig.tight_layout(); fig.savefig(od / "scatter.png", dpi=135); plt.close()


def save_digit_grid(od, dd, x_gen, target, n=8):
    od.mkdir(parents=True, exist_ok=True)
    ig = dd.render(x_gen[:n]).reshape(-1, 8, 8); ir = dd.render(target[:n]).reshape(-1, 8, 8)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.8))
    for i in range(n):
        axes[0, i].imshow(ir[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(ig[i], cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_title("real", fontsize=9, loc="left"); axes[1, 0].set_title("generated", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(od / "digits.png", dpi=120); plt.close()


def run_one(root, mode, L, args, C, xis, target, gamma, dd=None):
    td = root / "train"
    if args.phase in ("train", "all"):
        train(td, C, xis, args.Ttrain, mode, L, args.iters, args.samples, args.batch,
              args.lr, args.clip, args.resume, max(args.iters // 10, 1), dd=dd, wreg=args.wreg)
    results = []
    if args.phase in ("infer", "all"):
        d = _np.load(td / "weights.npz"); theta = {k: d[k] for k in d.files if k != "mode"}
        for Ti in args.infers:
            if Ti > args.Ttrain: continue
            print(f"  [infer] T_infer={Ti} ... {time.strftime('%H:%M:%S')}", flush=True)
            x_ad, sh_ad, per = generate(theta, mode, args.Ttrain, Ti, SEED + 202, args.gen_samples)
            mmd = float(G.mmd2_rbf(x_ad, target, gamma))
            rf = float("nan") if dd is not None else Q.resolved_frac(x_ad, C)
            od = root / f"infer_T{Ti}"
            if dd is not None: save_digit_grid(od, dd, x_ad, target)
            else: save_scatter(od, target, [("ADAPTIVE", x_ad, rf, sh_ad)], C)
            lines = [f"3q-GPU {mode} T_train={args.Ttrain} T_infer={Ti}",
                     f"adaptive: rf={rf:.3f} MMD2={mmd:.4f} total_shots={sh_ad:.0f}",
                     f"per_step: {[round(v) for v in per]}"]
            (od / "metrics.txt").write_text("\n".join(lines)); print("    " + "\n    ".join(lines), flush=True)
            results.append({"T_infer": Ti, "rf": rf, "mmd": mmd, "total": sh_ad})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "infer", "all"], default="all")
    ap.add_argument("--readout", choices=["fixed24I", "fixedI", "fixedC", "fullW", "diagW"], default="fullW")
    ap.add_argument("--dim", type=int, default=8, help="text(2 text)；NQ=log2(dim)。8=3qubit,64=6qubit,256=8qubit")
    ap.add_argument("--gain", type=float, default=-1.0, help="fixedC text c（<0 text A·N=A·dim）")
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, nargs="+", default=[12])
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--infers", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--gen-samples", type=int, default=1500)
    ap.add_argument("--dataset", choices=["gmm", "digits"], default="gmm")
    ap.add_argument("--wreg", type=float, default=0.0, help="text Frobenius text（text）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--date", default=DATE)
    ap.add_argument("--outname", default="3q_gpu_seed100")
    args = ap.parse_args()
    configure_gpu(args.dim)
    global GAIN; GAIN = args.gain if args.gain > 0 else A * N

    xis = G.psd_sqrt(G.projection(N))[0]
    if args.dataset == "digits":
        dd = Q.DigitsData(N); C = None
        target = dd.sample_x0(_np.random.default_rng(SEED + 99), args.gen_samples)
    else:
        dd = None; C = G.mixture_centers(N)
        target = G.sample_target(_np.random.default_rng(SEED + 99), args.gen_samples, C, S0, xis)
    gamma = 1.0 / G.median_bandwidth(target, target, _np.random.default_rng(1))
    root0 = Path(f"results/experiments/{args.date}/{args.outname}"); root0.mkdir(parents=True, exist_ok=True)
    print(f"3q-GPU seed{SEED}: dataset={args.dataset} dim={N}(NQ={NQ}) readout={args.readout} GAIN={GAIN:.1f} "
          f"wreg={args.wreg} layers={args.layers} T={args.Ttrain} iters={args.iters} samples={args.samples} "
          f"jax={jax.devices()} -> {root0}", flush=True)
    t_all = time.time(); cross = []
    for L in args.layers:
        rootL = (root0 / f"L{L}") if len(args.layers) > 1 else root0
        rootL.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {args.readout} L={L} -> {rootL} =====", flush=True)
        res = run_one(rootL, args.readout, L, args, C, xis, target, gamma, dd=dd)
        if res:
            with open(rootL / "summary.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["T_infer", "rf", "mmd2", "total_shots"])
                for r in sorted(res, key=lambda r: r["T_infer"]):
                    w.writerow([r["T_infer"], f"{r['rf']:.4f}", f"{r['mmd']:.5f}", f"{r['total']:.0f}"])
            rmax = max(res, key=lambda r: r["T_infer"]); cross.append((L, rmax["rf"], rmax["total"]))
    if len(args.layers) > 1 and cross:
        print("=== text（text）===", flush=True)
        for L, rf, tot in cross: print(f"  L={L}: rf={rf:.3f} shots={tot:.0f}", flush=True)
    print(f"\ntext，text {(time.time()-t_all)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
