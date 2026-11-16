"""Run the 8-qubit per-dimension QNN in an 8-dimensional VAE latent space.

The implementation reuses the circuit, training loop, and fixed readout from
``run_8qubit_perdim_gpu`` while replacing its data source with the VAE latent
representation. The 3-qubit and 8-qubit image experiments use the same
zero-sum latent representation and support the same DDIM stochasticity
parameter for a consistent comparison.
"""
from __future__ import annotations
import argparse, math, pickle, sys, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_8qubit_perdim_gpu as E
import noise_budgeted_generation as G
from run_image_paper import VAEData, real_pixels, ROOT
import run_image_direct as RID


def generate_8q(wr, T_train, T_infer, seed, nsamp, mode="adaptive", m_fixed=80000,
                eta=1.0, keep_snaps=False):
    """text E.generate，text (a) η text (b) text x0h。text。"""
    wr = jnp.asarray(wr)
    A, N, GAIN = E.A, E.N, E.GAIN
    betas = np.linspace(E.BMAX, E.BMIN, T_train + 1)
    idx = np.unique(np.linspace(0, T_train, T_infer + 1).round().astype(int))
    rng = np.random.default_rng(seed)
    x = np.clip(rng.standard_normal((nsamp, N)) * E.BMAX, -A, A)
    total = np.zeros(nsamp); per_step = []
    snaps = [x.copy()] if keep_snaps else None
    x0h_last = None
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        be, ben = float(betas[i0]), float(betas[i1])
        al = math.sqrt(max(1 - be**2, 1e-12)); aln = math.sqrt(max(1 - ben**2, 1e-12))
        sig, dirc, _ = G.compute_ddim_params(al, be, aln, ben, eta)
        tau = i0 / max(T_train - 1, 1) * np.ones(nsamp)
        Z = np.asarray(E.zexp(wr, x, tau)) / GAIN
        kc = aln - dirc * al / max(be, 1e-9)
        unit = (GAIN ** 2) * np.maximum(1 - Z ** 2, 0.0)
        if mode == "adaptive":
            req = (kc ** 2 / max(sig, 1e-12) ** 2) * unit
            Msh = np.clip(np.ceil(1.10 * np.ceil(req)), 2, 10 ** 9).astype(np.int64)
        else:
            Msh = np.full((nsamp, N), m_fixed, dtype=np.int64)
        per_step.append(float(np.mean(Msh))); total += np.mean(Msh, axis=1)
        meas_var = unit / Msh
        x0h = GAIN * Z + np.sqrt(meas_var) * rng.standard_normal((nsamp, N))
        res_var = np.maximum(sig ** 2 - kc ** 2 * meas_var, 0.0)
        ev = (x - al * x0h) / max(be, 1e-9)
        x = np.clip(aln * x0h + dirc * ev + np.sqrt(res_var) * rng.standard_normal((nsamp, N)), -A, A)
        x0h_last = x0h
        if keep_snaps: snaps.append(x.copy())
    return x, float(np.mean(total)), per_step, snaps, x0h_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["train", "gen"])
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--outdir", default=str(ROOT / "sec7_2_latent_8q"))
    ap.add_argument("--Ttrain", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=80000)
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--rep", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--high-M", type=int, default=80000)
    ap.add_argument("--gen-samples", type=int, default=1500)
    ap.add_argument("--grid-n", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dd = VAEData(8, args.beta)
    assert dd.x0.shape[1] == E.N == 8, "8-qubit text N=8"


    pool = dd.x0.astype(np.float64)
    E.sample_gmm = lambda rng, n, C=None: pool[rng.integers(0, len(pool), size=n)]

    root = Path(args.outdir); root.mkdir(parents=True, exist_ok=True)
    print(f"[8q part={args.part}] VAE-8 latent, {E.NQ} qubits, text 3I text(0 text) -> {root}", flush=True)

    t0 = time.time()
    if args.part == "train":
        E.train(root / "train", None, args.Ttrain, args.iters, args.samples, args.batch,
                args.lr, args.layers, args.clip, args.resume, max(args.iters // 10, 1))
    else:
        wr = np.load(root / "train" / "weights.npz")["wr"]
        tag = f"T{args.rep}_eta{args.eta:g}"
        target = dd.sample_x0(np.random.default_rng(E.SEED + 99), args.gen_samples)
        gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))

        _, sh_ad, _, snaps, x0h_ad = generate_8q(wr, args.Ttrain, args.rep, E.SEED + 202,
                                                 args.gen_samples, "adaptive", eta=args.eta, keep_snaps=True)
        _, sh_hi, _, _, x0h_hi = generate_8q(wr, args.Ttrain, args.rep, E.SEED + 201,
                                             args.gen_samples, "high", m_fixed=args.high_M, eta=args.eta)
        mmd_ad = float(G.mmd2_rbf(x0h_ad, target, gamma))
        mmd_hi = float(G.mmd2_rbf(x0h_hi, target, gamma))
        RP = real_pixels(args.gen_samples)
        gpix = 1.0 / G.median_bandwidth(RP, RP, np.random.default_rng(1))
        pm_ad = float(G.mmd2_rbf(dd.render(x0h_ad).reshape(len(x0h_ad), -1), RP, gpix))
        pm_hi = float(G.mmd2_rbf(dd.render(x0h_hi).reshape(len(x0h_hi), -1), RP, gpix))

        RID.fig_grid(root, f"grid_adaptive_{tag}.png", dd, target, x0h_ad, args.grid_n,
                     f"Adaptive (η={args.eta:g}): MMD$^2_{{pix}}$={pm_ad:.4f}   {sh_ad:.0f} shots   "
                     f"(VAE-8, 8 qubits, fixed readout)")
        RID.fig_grid(root, f"grid_highshot_{tag}.png", dd, target, x0h_hi, args.grid_n,
                     f"High-shot (η={args.eta:g}): MMD$^2_{{pix}}$={pm_hi:.4f}   {sh_hi:.0f} shots")
        RID.fig_denoise_grid(root, f"denoise_grid_{tag}.png", dd, snaps, ncol=args.grid_n)
        (root / f"metrics_{tag}.txt").write_text(
            f"8-qubit per-dim, fixed 3I readout (0 trainable readout params), VAE-8 latent\n"
            f"L={args.layers} T_infer={args.rep} eta={args.eta}\n"
            f"adaptive : MMD2_latent={mmd_ad:.4f}  MMD2_pixel={pm_ad:.4f}  shots={sh_ad:.0f}\n"
            f"high_shot: MMD2_latent={mmd_hi:.4f}  MMD2_pixel={pm_hi:.4f}  shots={sh_hi:.0f}\n"
            f"shot_saving = {sh_hi/max(sh_ad,1e-9):.0f}x\n")
        print(f"    [gen {tag}] adaptive MMD2_pix={pm_ad:.4f} (latent {mmd_ad:.4f}) shots={sh_ad:.0f}"
              f" | high MMD2_pix={pm_hi:.4f} shots={sh_hi:.0f} | text {sh_hi/max(sh_ad,1e-9):.0f}x", flush=True)
    print(f"[8q part={args.part}] text {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
