"""run_image_direct.py — §VII.1 text·text（raw text、text、text=2ⁿ）。

text M(run_quantum_3q_gpu)：configure_gpu(dim) text 2ⁿ text（dim=64→6qubit, 256→8qubit），
text hybrid text、text、text、T=1000。**text = text + text**。

text（text data/）：
  digits : 8×8=64 text → 6 qubit
  mnist  : 28×28 → resize text side×side（text 16×16=256 → 8 qubit）

text（raw text）：text → text(text Ξ) → text GMM text → clip[-A,A]。
text：pixel = s·x0 + text。

--part：train / gen（text/text real vs generated + text + MMD）。
--dataset digits|mnist  --dim（2 text）  --readout fullW|diagW  --dealias --dealias-max
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import scipy.ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quantum_3q_gpu as M
import noise_budgeted_generation as G
import run_quantum_x0 as Q
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.titleweight": "normal"})
SAVE = dict(dpi=150, bbox_inches="tight")
A = 3.0; SIGMA_GMM = 0.34
DATA = Path(__file__).resolve().parent.parent / "data"


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


class ImageData:
    """text →(text PCA text dim text)→ text → text。
    --pca：PCA text dim text（text、text，text、text、text）。
    text --pca：raw text resize text side×side（dim text=side²）。"""
    def __init__(self, name, dim, pool, use_pca):
        if name == "digits":
            imgs = np.load(DATA / "digits8x8.npz")["images"].astype(np.float64)
        elif name == "mnist":
            imgs = np.load(DATA / "mnist28x28.npz")["images"].astype(np.float64)
        else:
            raise ValueError(name)
        rng = np.random.default_rng(0)
        if len(imgs) > pool:
            imgs = imgs[rng.choice(len(imgs), pool, replace=False)]
        self.use_pca = use_pca
        if use_pca:
            from sklearn.decomposition import PCA
            Xpix = imgs.reshape(len(imgs), -1)
            self.side = int(round(Xpix.shape[1] ** 0.5))
            self.pca = PCA(n_components=dim).fit(Xpix)
            base = self.pca.transform(Xpix)
            info = f"PCA {Xpix.shape[1]}px->{dim} evr={self.pca.explained_variance_ratio_.sum():.2f}"
        else:
            side = int(round(dim ** 0.5)); assert side * side == dim, "text PCA text dim text=side²"
            if imgs.shape[1] != side:
                imgs = np.stack([ndi.zoom(im, side / imgs.shape[1], order=1) for im in imgs])
            base = imgs.reshape(len(imgs), -1); self.side = side
            info = f"raw {side}x{side}"
        bc = base - base.mean(axis=1, keepdims=True)
        self.mean_ = float(base.mean())
        self.s = float(bc.std() / SIGMA_GMM)
        self.x0 = np.clip(bc / self.s, -A, A).astype(np.float64)
        self.dim = dim
        print(f"  [ImageData] {name} {info} pool={len(self.x0)} s={self.s:.3f} "
              f"x0text[{self.x0.min():.2f},{self.x0.max():.2f}]", flush=True)

    def sample_x0(self, rng, n):
        return self.x0[rng.integers(0, len(self.x0), size=n)]

    def render(self, x0):
        z = self.s * np.asarray(x0)
        pix = self.pca.inverse_transform(z) if self.use_pca else (z + self.mean_)
        return pix.reshape(-1, self.side, self.side)



def generate(theta, mode, T_train, T_infer, seed, nsamp, shot_mode="adaptive",
             high_M=80000, m_min=2, m_max=200000, record_traj=False, eta=1.0):
    N = M.N; Wr, br = M.readout_numpy(theta, mode); wr = np.asarray(theta["wr"])
    betas = np.linspace(M.BMAX, M.BMIN, T_train + 1)
    idx = np.unique(np.linspace(0, T_train, T_infer + 1).round().astype(int))
    rng = np.random.default_rng(seed)
    P = G.projection(N); xis = G.psd_sqrt(P)[0]
    x = rng.normal(size=(nsamp, N)) @ xis.T
    total = np.zeros(nsamp); per_step = []; traj = [x.copy()] if record_traj else None
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        be, ben = float(betas[i0]), float(betas[i1])
        al = math.sqrt(max(1 - be**2, 1e-12)); aln = math.sqrt(max(1 - ben**2, 1e-12))
        sig, dirc, ct = G.compute_ddim_params(al, be, aln, ben, eta)
        qb, _ = G.chi_to_p(x, A)
        tau = i0 / max(T_train - 1, 1) * np.ones(nsamp)
        p = np.array(M.jcirc(jnp.asarray(wr), jnp.asarray(Q.make_state(qb, tau)), jnp.asarray(Q.make_feat(qb, tau))))
        p = np.clip(p.astype(np.float64), 0.0, None); p = p / p.sum(axis=1, keepdims=True)
        kc = aln - dirc * al / max(be, 1e-9)
        if shot_mode == "adaptive":
            req = Q.min_shots_W(p, Wr, kc, sig, P)
            sh = np.clip(np.ceil(1.10 * req), m_min, m_max).astype(np.int64)
        else:
            sh = np.full(nsamp, int(high_M), dtype=np.int64)
        per_step.append(float(np.mean(sh)))
        ph = G.empirical_born_readout(rng, p, sh)
        x0h = G.project_zero_sum(ph @ Wr + br)
        ev = (x - al * x0h) / max(be, 1e-9)
        res = Q.residual_W(rng, p, sh, Wr, kc, sig, P)
        x = G.project_zero_sum(np.clip(aln * x0h + dirc * ev + sig * res, -A, A))
        total += sh
        x0h_last = x0h
        if record_traj: traj.append(x.copy())
    return x, float(np.mean(total)), per_step, traj, x0h_last


def _imshow(ax, img, vmin, vmax):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])


def fig_grid(od, fname, dd, real_x0, gen_x0, n, sublabel):
    real = dd.render(real_x0[:n]); gen = dd.render(gen_x0[:n])
    vmin, vmax = float(np.percentile(real, 1)), float(np.percentile(real, 99))
    fig, ax = plt.subplots(2, n, figsize=(n * 1.4, 3.4))
    for i in range(n):
        _imshow(ax[0, i], real[i], vmin, vmax); _imshow(ax[1, i], gen[i], vmin, vmax)
    ax[0, 0].set_ylabel("real", fontsize=14); ax[1, 0].set_ylabel("generated", fontsize=14)
    fig.text(0.5, 0.02, sublabel, ha="center", fontsize=15)
    fig.subplots_adjust(bottom=0.1, wspace=0.05, hspace=0.05)
    fig.savefig(od / fname, **SAVE); plt.close(fig)


def fig_denoise_grid(od, fname, dd, traj, ncol=8):
    n = len(traj) - 1
    picks = sorted(set([0, round(n / 3), round(2 * n / 3), n]))
    while len(picks) < 4: picks.append(picks[-1])
    labels = ["Noise", "Interm.", "Interm.", "Denoised"]
    fin = dd.render(traj[-1][:ncol]); vmin, vmax = float(np.percentile(fin, 1)), float(np.percentile(fin, 99))
    fig, ax = plt.subplots(4, ncol, figsize=(ncol * 1.3, 5.6))
    for r, (i, lb) in enumerate(zip(picks, labels)):
        imgs = dd.render(traj[i][:ncol])
        for c in range(ncol): _imshow(ax[r, c], imgs[c], vmin, vmax)
        ax[r, 0].set_ylabel(f"{lb}\n{i}/{n}", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(od / fname, **SAVE); plt.close(fig)


def part_train(root, args, dd, xis):
    M.train(root / "train", None, xis, args.Ttrain, args.readout, args.layers, args.iters,
            args.samples, args.batch, args.lr, args.clip, args.resume, max(args.iters // 10, 1),
            dd=dd, wreg=args.wreg)


def part_gen(root, args, dd, target, gamma):
    d = np.load(root / "train" / "weights.npz"); theta = {k: d[k] for k in d.files if k != "mode"}
    tag = f"T{args.rep}_eta{args.eta:g}"
    x_ad, sh_ad, per, traj, x0h_ad = generate(theta, args.readout, args.Ttrain, args.rep, M.SEED + 202,
                                              args.gen_samples, "adaptive", record_traj=True, eta=args.eta)
    mmd_ad = float(G.mmd2_rbf(x0h_ad, target, gamma))
    x_hi, sh_hi, _, _, x0h_hi = generate(theta, args.readout, args.Ttrain, args.rep, M.SEED + 201,
                                         args.gen_samples, "high", high_M=args.high_M, eta=args.eta)
    mmd_hi = float(G.mmd2_rbf(x0h_hi, target, gamma))
    fig_grid(root, f"grid_adaptive_{tag}.png", dd, target, x0h_ad, args.grid_n,
             f"Adaptive (η={args.eta:g}): MMD$^2$={mmd_ad:.4f}   {sh_ad:.0f} shots   ({args.dataset}, dim={args.dim}, {M.NQ} qubits)")
    fig_grid(root, f"grid_highshot_{tag}.png", dd, target, x0h_hi, args.grid_n,
             f"High-shot (η={args.eta:g}): MMD$^2$={mmd_hi:.4f}   {sh_hi:.0f} shots")
    fig_denoise_grid(root, f"denoise_grid_{tag}.png", dd, traj)
    (root / f"metrics_{tag}.txt").write_text(
        f"{args.dataset} dim={args.dim} qubits={M.NQ} readout={args.readout} dealias={args.dealias} T_infer={args.rep} eta={args.eta}\n"
        f"adaptive : MMD2={mmd_ad:.4f}  total_shots={sh_ad:.0f}\n"
        f"high_shot: MMD2={mmd_hi:.4f}  total_shots={sh_hi:.0f}\n")
    print(f"    [gen {tag}] adaptive MMD2={mmd_ad:.4f} shots={sh_ad:.0f} | high MMD2={mmd_hi:.4f} shots={sh_hi:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "gen"])
    ap.add_argument("--dataset", required=True, choices=["digits", "mnist"])
    ap.add_argument("--dim", type=int, default=64, help="text=qubit text 2 text。PCA:text(32→5q,64→6q)；raw:side²")
    ap.add_argument("--pca", action="store_true", help="text PCA text dim text（text：text、text；raw text）")
    ap.add_argument("--readout", default="diagW", choices=["fullW", "diagW", "fixed24I"])
    ap.add_argument("--dealias", action="store_true", help="text hybrid")
    ap.add_argument("--dealias-max", type=float, default=math.pi, help="text(text π)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--wreg", type=float, default=0.0)
    ap.add_argument("--rep", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0, help="DDIM text；text 0.5/0.3 text、text")
    ap.add_argument("--high-M", type=int, default=80000)
    ap.add_argument("--gen-samples", type=int, default=1500)
    ap.add_argument("--grid-n", type=int, default=8)
    ap.add_argument("--pool", type=int, default=20000, help="MNIST text")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    global DEALIAS_SCALE
    M.configure_gpu(args.dim)
    if args.dealias:
        DEALIAS_SCALE = args.dealias_max / (0.6 * M.N + 0.15 * M.N * M.N)
        print(f"[DEALIAS ON] scale={DEALIAS_SCALE:.4g} (max≈{args.dealias_max:.2f} rad)", flush=True)
    dd = ImageData(args.dataset, args.dim, args.pool, args.pca)
    xis = G.psd_sqrt(G.projection(M.N))[0]
    target = dd.sample_x0(np.random.default_rng(M.SEED + 99), args.gen_samples)
    gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))
    enc = ("pca" if args.pca else "raw") + (f"_dealias" if args.dealias else "")
    out = args.outdir or f"results/final_image/{args.dataset}_d{args.dim}_{args.readout}_{enc}"
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    print(f"[image part={args.part}] {args.dataset} dim={args.dim} qubits={M.NQ} readout={args.readout} -> {root}", flush=True)
    t0 = time.time()
    if args.part == "train":  part_train(root, args, dd, xis)
    elif args.part == "gen":  part_gen(root, args, dd, target, gamma)
    print(f"[image part={args.part}] text，text {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
