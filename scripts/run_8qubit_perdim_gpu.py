"""JAX implementation of the 8-qubit per-dimension QNN experiment.

Each input dimension is encoded with ``RY(arccos(x_i/A))``, followed by
data-reuploading RY/RZ layers and a CNOT ring. The model uses a fixed readout,
an x0-prediction objective, per-dimension DDIM sampling, and adaptive
per-dimension measurement-shot budgets. JAX uses a GPU when a compatible
backend is available and otherwise runs on CPU.
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from datetime import date
from pathlib import Path
import numpy as _np
import jax, jax.numpy as jnp
from jax import jit, value_and_grad
from functools import partial
import pennylane as qml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_budgeted_generation as G
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 100
A = 3.0; N = 8; NQ = 8; S0 = 0.12
BMAX, BMIN = 0.95, 0.10
FDIM = N + 2
GAIN = A
DATE = date.today().isoformat()

dev = qml.device("default.qubit", wires=NQ)

@qml.qnode(dev, interface="jax", diff_method="backprop")
def _circ(wr, xn, tau):
    for i in range(NQ):
        qml.RY(jnp.arccos(jnp.clip(xn[:, i], -1.0, 1.0)), wires=i)
    feat = jnp.concatenate([jnp.ones((xn.shape[0], 1)), xn, tau[:, None]], axis=-1)   # (batch, N+2)
    L = wr.shape[0]
    for l in range(L):
        for i in range(NQ):
            qml.RY(jnp.sum(wr[l, i, 0] * feat, axis=-1), wires=i)
            qml.RZ(jnp.sum(wr[l, i, 1] * feat, axis=-1), wires=i)
        for i in range(NQ):
            qml.CNOT(wires=[i, (i + 1) % NQ])
    return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]

def zexp(wr, x, tau):
    """x=(batch,N) text；text x̂0=(batch,N)=A·<Z>（jax text）。"""
    z = _circ(wr, jnp.asarray(x) / A, jnp.asarray(tau))
    return GAIN * jnp.stack(z, axis=-1)



def resolved_frac(X, C):
    d = _np.linalg.norm(C[:, None] - C[None], axis=-1)
    dmin = d[d > 0].min()
    nn = _np.linalg.norm(X[:, None] - C[None], axis=-1).min(1)
    return float((nn < 0.5 * dmin).mean())

def sample_gmm(rng, n, C):
    k = rng.integers(0, len(C), size=n)
    return _np.clip(C[k] + S0 * rng.standard_normal((n, N)), -A, A)

def build_dataset(rng, count, C, T):
    betas = _np.linspace(BMAX, BMIN, T + 1)
    x0 = sample_gmm(rng, count, C)
    ti = rng.integers(0, T, size=count)
    be = betas[ti]; al = _np.sqrt(_np.maximum(1 - be**2, 1e-12))
    tau = ti / max(T - 1, 1)
    eps = rng.standard_normal((count, N))
    xt = _np.clip(al[:, None] * x0 + be[:, None] * eps, -A, A)
    return xt.astype("float32"), tau.astype("float32"), x0.astype("float32")



def train(od, C, T, iters, samples, batch, lr, layers, clip, resume, ckpt_every):
    od.mkdir(parents=True, exist_ok=True)
    scale = (12.0 / layers) ** 0.5
    lr0 = lr * scale; init_std = 0.1 * scale
    rng = _np.random.default_rng(SEED)
    xt, tau, x0 = build_dataset(rng, samples, C, T)
    xt = jnp.asarray(xt); tau = jnp.asarray(tau); x0 = jnp.asarray(x0)
    var = float(jnp.mean(x0 ** 2))

    def loss_fn(wr, xb, tb, yb):
        return jnp.mean((zexp(wr, xb, tb) - yb) ** 2) / var
    eval_loss = jit(loss_fn)

    @partial(jit, static_argnums=())
    def step(wr, mw, vw, xb, tb, yb, it, lr_t):
        l, g = value_and_grad(loss_fn)(wr, xb, tb, yb)
        gn = jnp.sqrt(jnp.sum(g ** 2))
        g = jnp.where((clip > 0) & (gn > clip), g * clip / (gn + 1e-12), g)
        mw = 0.9 * mw + 0.1 * g
        vw = 0.999 * vw + 0.001 * g ** 2
        wr = wr - lr_t * (mw / (1 - 0.9 ** it)) / (jnp.sqrt(vw / (1 - 0.999 ** it)) + 1e-8)
        return wr, mw, vw, l * var, gn

    ckpt = od / "checkpoint.npz"
    if resume and ckpt.exists():
        d = _np.load(ckpt); wr = jnp.asarray(d["wr"]); mw = jnp.asarray(d["mw"]); vw = jnp.asarray(d["vw"])
        it0 = int(d["it"]); hist = list(map(tuple, d["hist"])) if "hist" in d else []
        print(f"  [resume] text iter {it0} text（{ckpt}）", flush=True)
    else:
        wr = jnp.asarray(init_std * _np.random.default_rng(SEED + 1).standard_normal((layers, NQ, 2, FDIM)))
        mw = jnp.zeros_like(wr); vw = jnp.zeros_like(wr); it0 = 0; hist = []
    lr_decay_at = int(iters * 0.6)
    rec_every = max(iters // 150, 1); log_step = max(iters // 10, 1); next_log = it0 + log_step
    mon = min(2048, samples); clip_hits = 0; t0 = time.time()
    print(f"  [train-GPU] 8q L={layers} T={T} iters={iters} samples={samples} lr0={lr0:.4f} clip={clip} "
          f"backend=jax dev={jax.devices()[0].platform} text {time.strftime('%H:%M:%S')}", flush=True)

    for it in range(it0 + 1, iters + 1):
        lr_t = lr0 if it < lr_decay_at else lr0 * 0.25
        idx = _np.random.default_rng(it).integers(0, samples, size=batch)
        wr, mw, vw, l_batch, gn = step(wr, mw, vw, xt[idx], tau[idx], x0[idx],
                                       jnp.float32(it), jnp.float32(lr_t))
        if float(gn) > clip and clip > 0: clip_hits += 1
        if it % rec_every == 0 or it == it0 + 1:
            fl = float(eval_loss(wr, xt[:mon], tau[:mon], x0[:mon])) * var
            hist.append((it, fl))
        if it == it0 + 1 or it >= next_log:
            fl_now = float(eval_loss(wr, xt[:mon], tau[:mon], x0[:mon])) * var
            zr = float(jnp.abs(zexp(wr, xt[:512], tau[:512])).max())
            ch = f" clip%={100*clip_hits/max(it-it0,1):.0f}" if clip else ""
            print(f"    [{100*it/iters:3.0f}%] iter {it:>6d}/{iters}  chi-MSE={fl_now:.5f}{ch}  "
                  f"|x̂0|max={zr:.2f}  ({time.time()-t0:.0f}s) {time.strftime('%H:%M:%S')}", flush=True)
            if it != it0 + 1: next_log += log_step
        if ckpt_every and it % ckpt_every == 0:
            _np.savez(ckpt, wr=_np.array(wr), mw=_np.array(mw), vw=_np.array(vw), it=it, hist=_np.array(hist))

    final = hist[-1][1] if hist else float("nan")
    _np.savez(od / "weights.npz", wr=_np.array(wr))
    _np.savez(ckpt, wr=_np.array(wr), mw=_np.array(mw), vw=_np.array(vw), it=iters, hist=_np.array(hist))
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4)); h = _np.array(hist)
        ax.plot(h[:, 0], h[:, 1]); ax.set_yscale("log"); ax.set_xlabel("iteration")
        ax.set_ylabel("chi-MSE (log)"); ax.grid(alpha=0.3); ax.set_title(f"8-qubit per-dim GPU, L={layers}, T={T}")
        fig.tight_layout(); fig.savefig(od / "training_loss.png", dpi=130); plt.close()
    print(f"  [train-GPU] done chi-MSE={final:.4f} clip_hits={clip_hits} text {time.strftime('%H:%M:%S')} "
          f"text {(time.time()-t0)/60:.1f}min", flush=True)
    return _np.array(wr), final



def generate(wr, T_train, T_infer, seed, nsamp, mode="adaptive", m_fixed=80000, keep_snaps=False):
    wr = jnp.asarray(wr)
    betas = _np.linspace(BMAX, BMIN, T_train + 1)
    idx = _np.unique(_np.linspace(0, T_train, T_infer + 1).round().astype(int))
    rng = _np.random.default_rng(seed)
    x = _np.clip(rng.standard_normal((nsamp, N)) * BMAX, -A, A)
    total = _np.zeros(nsamp); per_step = []; snaps = [x.copy()] if keep_snaps else None
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        be, ben = float(betas[i0]), float(betas[i1])
        al = math.sqrt(max(1 - be**2, 1e-12)); aln = math.sqrt(max(1 - ben**2, 1e-12))
        sig, dirc, ct = G.compute_ddim_params(al, be, aln, ben, 1.0)
        tau = i0 / max(T_train - 1, 1) * _np.ones(nsamp)
        Z = _np.asarray(zexp(wr, x, tau)) / GAIN
        kc = aln - dirc * al / max(be, 1e-9)
        unit = (GAIN**2) * _np.maximum(1 - Z**2, 0.0)
        if mode == "adaptive":
            req = (kc**2 / max(sig, 1e-12)**2) * unit
            M = _np.clip(_np.ceil(1.10 * _np.ceil(req)), 2, 10**9).astype(_np.int64)
        else:
            M = _np.full((nsamp, N), m_fixed, dtype=_np.int64)
        per_step.append(float(_np.mean(M))); total += _np.mean(M, axis=1)
        meas_var = unit / M
        x0h = GAIN * Z + _np.sqrt(meas_var) * rng.standard_normal((nsamp, N))
        res_var = _np.maximum(sig**2 - kc**2 * meas_var, 0.0)
        ev = (x - al * x0h) / max(be, 1e-9)
        x = _np.clip(aln * x0h + dirc * ev + _np.sqrt(res_var) * rng.standard_normal((nsamp, N)), -A, A)
        if keep_snaps: snaps.append(x.copy())
    return x, float(_np.mean(total)), per_step, snaps


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
    fig.suptitle("8-qubit per-dim (GPU/jax) — fixed 3I readout, circuit-only", fontweight="bold")
    fig.tight_layout(); fig.savefig(od / "scatter.png", dpi=135); plt.close()


def run_L(root, L, args, C, target, gamma):
    td = root / "train"
    if args.phase in ("train", "all"):
        train(td, C, args.Ttrain, args.iters, args.samples, args.batch, args.lr, L,
              args.clip, args.resume, max(args.iters // 10, 1))
    results = []
    if args.phase in ("infer", "all"):
        wp = td / "weights.npz"
        if not wp.exists():
            print(f"  [infer] text {wp}", flush=True); return results
        wr = _np.load(wp)["wr"]
        for Ti in args.infers:
            if Ti > args.Ttrain: continue
            print(f"  [infer] T_infer={Ti} ... {time.strftime('%H:%M:%S')}", flush=True)
            x_ad, sh_ad, per, _ = generate(wr, args.Ttrain, Ti, SEED + 202, args.gen_samples, "adaptive")
            x_hi, sh_hi, _, _ = generate(wr, args.Ttrain, Ti, SEED + 201, args.gen_samples, "high", m_fixed=80000)
            rf = resolved_frac(x_ad, C); rf_hi = resolved_frac(x_hi, C)
            mmd = float(G.mmd2_rbf(x_ad, target, gamma))
            od = root / f"infer_T{Ti}"
            save_scatter(od, target, [("high_shot", x_hi, rf_hi, sh_hi), ("ADAPTIVE", x_ad, rf, sh_ad)], C)
            lines = [f"8q-GPU T_train={args.Ttrain} T_infer={Ti}",
                     f"adaptive: rf={rf:.3f} MMD2={mmd:.4f} total_shots={sh_ad:.0f}",
                     f"high_shot: rf={rf_hi:.3f}", f"per_step: {[round(v) for v in per]}"]
            (od / "metrics.txt").write_text("\n".join(lines)); print("    " + "\n    ".join(lines), flush=True)
            results.append({"T_infer": Ti, "rf": rf, "mmd": mmd, "total": sh_ad})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "infer", "all"], default="all")
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=80000)
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--infers", type=int, nargs="+", default=[5, 10, 20, 50, 100, 1000])
    ap.add_argument("--gen-samples", type=int, default=1500)
    ap.add_argument("--gain", type=float, default=A, help="text x̂0=gain·<Z>（text A=3=text）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--date", default=DATE)
    ap.add_argument("--outname", default="8q_L12_gpu_seed100")
    args = ap.parse_args()
    global GAIN; GAIN = args.gain

    C = G.mixture_centers(8)
    target = sample_gmm(_np.random.default_rng(SEED + 99), args.gen_samples, C)
    gamma = 1.0 / G.median_bandwidth(target, target, _np.random.default_rng(1))
    root = Path(f"results/experiments/{args.date}/{args.outname}"); root.mkdir(parents=True, exist_ok=True)
    print(f"8-qubit per-dim GPU(jax) seed{SEED}: L={args.layers} T={args.Ttrain} phase={args.phase} "
          f"iters={args.iters} samples={args.samples} jax_dev={jax.devices()} text -> {root}", flush=True)
    t_all = time.time()
    res = run_L(root, args.layers, args, C, target, gamma)
    if res:
        with open(root / "summary.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["T_infer", "rf", "mmd2", "total_shots"])
            for r in sorted(res, key=lambda r: r["T_infer"]):
                w.writerow([r["T_infer"], f"{r['rf']:.4f}", f"{r['mmd']:.5f}", f"{r['total']:.0f}"])
    print(f"\ntext，text {(time.time()-t_all)/60:.1f}min {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
