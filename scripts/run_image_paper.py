"""run_image_paper.py — text §VII text（text：text/text + text）。

text，text、text，text GMM text run_T1000_mainline.py：
  --source pca : §VII.1 Direct  MNIST -> PCA(N-1 text) -> 【text】text N text -> nq=log2(N) qubit
  --source vae : §VII.2 Latent  MNIST -> VAE(N-1 text) -> 【text】text N text -> nq=log2(N) qubit

text（text，text bug）：
  text R^N text N-1 text。text"text N text"text，
  text render text pca.inverse_transform()（text）-> text/text。
  text：text N-1 text，text U(N×(N-1)) text N text —— text、text。
  GMM text（mixture_centers text），text。

text（import，text）：run_quantum_3q_gpu(M) / noise_budgeted_generation(G) / run_quantum_x0(Q)
                     run_image_direct(RID) text generate / fig_grid / fig_denoise_grid
text：diagW text π，fullW text 2π。
"""
from __future__ import annotations
import argparse, math, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quantum_3q_gpu as M
import noise_budgeted_generation as G
import run_quantum_x0 as Q
import run_image_direct as RID

ROOT = Path(__file__).resolve().parent / "results/final_image_paper"
VAE = ROOT / "vae"
DATA = Path(__file__).resolve().parent.parent / "data"
A, SIGMA_GMM = 3.0, 0.34


def zero_sum_basis(n):
    """U:(n, n-1) text 0 —— text（text/text）。"""
    P = np.eye(n) - np.ones((n, n)) / n
    _, V = np.linalg.eigh(P)
    U = V[:, 1:]
    assert np.allclose(U.sum(0), 0, atol=1e-10) and np.allclose(U.T @ U, np.eye(n - 1), atol=1e-10)
    return U


class PCAData:
    """§VII.1：PCA(N-1) -> text N text。text。"""
    def __init__(self, N, whiten, pool=20000, dataset="mnist"):
        from sklearn.decomposition import PCA
        f = "mnist28x28.npz" if dataset == "mnist" else "digits8x8.npz"
        imgs = np.load(DATA / f)["images"].astype(np.float64)
        imgs = imgs / (255.0 if dataset == "mnist" else max(imgs.max(), 1.0))
        rng = np.random.default_rng(0)
        if len(imgs) > pool:
            imgs = imgs[rng.choice(len(imgs), pool, replace=False)]
        X = imgs.reshape(len(imgs), -1)
        self.side = int(round(X.shape[1] ** 0.5))
        self.dataset = dataset
        self.dim, self.whiten = N, whiten
        self.U = zero_sum_basis(N)
        self.pca = PCA(n_components=N - 1).fit(X)
        z = self.pca.transform(X)
        self.sd = np.sqrt(self.pca.explained_variance_)
        if whiten:
            z = z / self.sd
        zn = z @ self.U.T
        self.s = float(zn.std() / SIGMA_GMM)
        self.x0 = np.clip(zn / self.s, -A, A)
        print(f"  [PCAData] {dataset} {self.side}x{self.side} PCA-{N-1}->text{N} whiten={whiten} "
              f"evr={self.pca.explained_variance_ratio_.sum():.3f} "
              f"pool={len(self.x0)} s={self.s:.3f} clip={np.mean(np.abs(zn/self.s)>A)*100:.2f}% "
              f"text={np.abs(self.x0.mean(1)).max():.1e}", flush=True)

    def sample_x0(self, rng, n):
        return self.x0[rng.integers(0, len(self.x0), size=n)]

    def render(self, x0):
        z = (np.asarray(x0) * self.s) @ self.U
        if self.whiten:
            z = z * self.sd
        return self.pca.inverse_transform(z).reshape(-1, self.side, self.side)


class VAEData:
    """§VII.2：VAE(N-1 text) -> text N text。"""
    def __init__(self, N, beta):
        d = np.load(VAE / f"latent_N{N}_beta{beta:g}.npz")
        with open(VAE / f"vae_N{N}_beta{beta:g}.pkl", "rb") as f:
            p = pickle.load(f)
        self.x0 = d["x0"].astype(np.float64)
        self.s = float(d["scale"])
        self.U = d["U"]
        self.dec = p["dec"]
        self.side, self.dim = 28, N
        print(f"  [VAEData] VAE N={N}(K={N-1}) β={beta:g} pool={len(self.x0)} s={self.s:.3f} "
              f"text={np.abs(self.x0.mean(1)).max():.1e}", flush=True)

    def sample_x0(self, rng, n):
        return self.x0[rng.integers(0, len(self.x0), size=n)]

    def render(self, x0):
        h = (np.asarray(x0) * self.s) @ self.U
        for w, b in self.dec[:-1]:
            h = np.tanh(h @ w + b)
        w, b = self.dec[-1]
        return (1.0 / (1.0 + np.exp(-(h @ w + b)))).reshape(-1, self.side, self.side)


def real_pixels(n, seed=12345, dataset="mnist"):
    f = "mnist28x28.npz" if dataset == "mnist" else "digits8x8.npz"
    imgs = np.load(DATA / f)["images"].astype(np.float64)[:20000]
    imgs = imgs / (255.0 if dataset == "mnist" else max(imgs.max(), 1.0))
    r = np.random.default_rng(seed)
    return imgs[r.choice(len(imgs), n, replace=False)].reshape(n, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "gen"])
    ap.add_argument("--source", required=True, choices=["pca", "vae"])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "digits"], help="text --source pca text")
    ap.add_argument("--dim", type=int, default=8, help="text N=2^nq（8->3q, 16->4q, 32->5q）")
    ap.add_argument("--beta", type=float, default=0.01, help="VAE text β")
    ap.add_argument("--whiten", action="store_true", help="PCA text（text，text）")
    ap.add_argument("--readout", default="diagW", choices=["fullW", "diagW", "fixed24I"])
    ap.add_argument("--dealias-max", type=float, default=None, help="text diagW=π, fullW=2π（text）")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--wreg", type=float, default=0.0)
    ap.add_argument("--rep", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--high-M", type=int, default=80000)
    ap.add_argument("--gen-samples", type=int, default=1500)
    ap.add_argument("--grid-n", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dmax = args.dealias_max if args.dealias_max is not None else (
        2 * math.pi if args.readout == "fullW" else math.pi)

    M.configure_gpu(args.dim)
    RID.DEALIAS_SCALE = dmax / (0.6 * M.N + 0.15 * M.N * M.N)
    print(f"[DEALIAS] max={dmax:.4f} rad  scale={RID.DEALIAS_SCALE:.4g}", flush=True)

    dd = (PCAData(args.dim, args.whiten, dataset=args.dataset) if args.source == "pca"
          else VAEData(args.dim, args.beta))
    xis = G.psd_sqrt(G.projection(M.N))[0]
    target = dd.sample_x0(np.random.default_rng(M.SEED + 99), args.gen_samples)
    gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))

    if args.outdir:
        root = Path(args.outdir)
    else:
        sec = "sec7_1_direct_pca" if args.source == "pca" else "sec7_2_latent_3q"
        pre = f"{args.dataset}_" if (args.source == "pca" and args.dataset != "mnist") else ""
        tag = f"{pre}d{args.dim}_{args.readout}" + ("_whiten" if (args.source == "pca" and args.whiten) else "")
        root = ROOT / sec / tag
    root.mkdir(parents=True, exist_ok=True)
    print(f"[{args.source} part={args.part}] N={args.dim} qubits={M.NQ} readout={args.readout} -> {root}", flush=True)

    t0 = time.time()
    if args.part == "train":
        M.train(root / "train", None, xis, args.Ttrain, args.readout, args.layers, args.iters,
                args.samples, args.batch, args.lr, args.clip, args.resume,
                max(args.iters // 10, 1), dd=dd, wreg=args.wreg)
    else:
        d = np.load(root / "train" / "weights.npz")
        theta = {k: d[k] for k in d.files if k != "mode"}
        tag = f"T{args.rep}_eta{args.eta:g}"
        _, sh_ad, _, traj, x0h_ad = RID.generate(theta, args.readout, args.Ttrain, args.rep,
                                                 M.SEED + 202, args.gen_samples, "adaptive",
                                                 record_traj=True, eta=args.eta)
        _, sh_hi, _, _, x0h_hi = RID.generate(theta, args.readout, args.Ttrain, args.rep,
                                              M.SEED + 201, args.gen_samples, "high",
                                              high_M=args.high_M, eta=args.eta)
        mmd_ad = float(G.mmd2_rbf(x0h_ad, target, gamma))
        mmd_hi = float(G.mmd2_rbf(x0h_hi, target, gamma))

        RP = real_pixels(args.gen_samples, dataset=(args.dataset if args.source == "pca" else "mnist"))
        gpix = 1.0 / G.median_bandwidth(RP, RP, np.random.default_rng(1))
        pm_ad = float(G.mmd2_rbf(dd.render(x0h_ad).reshape(len(x0h_ad), -1), RP, gpix))
        pm_hi = float(G.mmd2_rbf(dd.render(x0h_hi).reshape(len(x0h_hi), -1), RP, gpix))

        lbl = f"{'PCA' if args.source=='pca' else 'VAE'}-{args.dim}, {M.NQ} qubits, {args.readout}"
        RID.fig_grid(root, f"grid_adaptive_{tag}.png", dd, target, x0h_ad, args.grid_n,
                     f"Adaptive (η={args.eta:g}): MMD$^2_{{pix}}$={pm_ad:.4f}   {sh_ad:.0f} shots   ({lbl})")
        RID.fig_grid(root, f"grid_highshot_{tag}.png", dd, target, x0h_hi, args.grid_n,
                     f"High-shot (η={args.eta:g}): MMD$^2_{{pix}}$={pm_hi:.4f}   {sh_hi:.0f} shots")
        RID.fig_denoise_grid(root, f"denoise_grid_{tag}.png", dd, traj, ncol=args.grid_n)
        (root / f"metrics_{tag}.txt").write_text(
            f"source={args.source} N={args.dim} qubits={M.NQ} readout={args.readout} "
            f"dealias_max={dmax:.4f} L={args.layers} T_infer={args.rep} eta={args.eta}\n"
            f"adaptive : MMD2_latent={mmd_ad:.4f}  MMD2_pixel={pm_ad:.4f}  shots={sh_ad:.0f}\n"
            f"high_shot: MMD2_latent={mmd_hi:.4f}  MMD2_pixel={pm_hi:.4f}  shots={sh_hi:.0f}\n"
            f"shot_saving = {sh_hi/max(sh_ad,1e-9):.0f}x\n")
        print(f"    [gen {tag}] adaptive MMD2_pix={pm_ad:.4f} (latent {mmd_ad:.4f}) shots={sh_ad:.0f}"
              f" | high MMD2_pix={pm_hi:.4f} shots={sh_hi:.0f} | text {sh_hi/max(sh_ad,1e-9):.0f}x", flush=True)
    print(f"[{args.source} part={args.part}] text {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
