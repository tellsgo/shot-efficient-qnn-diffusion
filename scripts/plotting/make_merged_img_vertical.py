#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""text：5 text(real/VAE + 3q/4q/8q)text。
   text VAE text。
   text merged_img_arrays.npz，text。

   text(text, quddpm_gpu text, text GPU):
     source ~/miniconda3/etc/profile.d/conda.sh && conda activate quddpm_gpu
     CUDA_VISIBLE_DEVICES=1 python scripts/plotting/make_merged_img_vertical.py
   text: results/figures/fig_merged_img_8col.png
"""
import sys, math, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_quantum_3q_gpu as M
import run_image_direct as RID
import run_image_paper as RIP
from run_image_paper import VAEData
import run_8qubit_perdim_gpu as E
import run_image_paper_8q as R8

GEN_N = 10
DATA = PROJECT_ROOT / "data"
RIP.ROOT = SCRIPTS_DIR / "results" / "final_image_paper"
RIP.VAE = RIP.ROOT / "vae"
VAE = RIP.ROOT / "vae"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
K15_SEED = M.SEED + 909


def npmlp(ps, x, sig=False):
    h = x
    for w, b in ps[:-1]:
        h = np.tanh(h @ w + b)
    w, b = ps[-1]
    o = h @ w + b
    return (1.0 / (1.0 + np.exp(-o))) if sig else o


# ---------- (a) Real samples and VAE reconstructions ----------
imgs = np.load(DATA / "mnist28x28.npz")["images"].astype(np.float64)[:20000] / 255.0
sel = [1, 3, 5, 7, 2, 0, 13, 15, 17, 4][:GEN_N]
realflat = imgs[sel].reshape(len(sel), -1)
real = realflat.reshape(-1, 28, 28)
with open(VAE / "vae_N8_beta0.01.pkl", "rb") as f:
    vp = pickle.load(f)
vae_rt = npmlp(vp["dec"], npmlp(vp["enc_mu"], realflat), sig=True).reshape(-1, 28, 28)



def gen_log(dim, wdir, seed):
    M.configure_gpu(dim)
    RID.DEALIAS_SCALE = math.pi / (0.6 * M.N + 0.15 * M.N * M.N)
    dd = VAEData(dim, 0.01)
    d = np.load(RIP.ROOT / wdir / "train" / "weights.npz")
    theta = {k: d[k] for k in d.files if k != "mode"}
    out = RID.generate(theta, "diagW", 1000, 20, seed, GEN_N, "adaptive", record_traj=True, eta=1.0)
    return dd.render(out[4][:GEN_N])


gen_k7 = gen_log(8, "sec7_2_latent_3q/d8_diagW", M.SEED + 202)
gen_k15 = gen_log(16, "sec7_2_latent_3q/d16_diagW", K15_SEED)
dd8 = VAEData(8, 0.01)
wr = np.load(RIP.ROOT / "sec7_2_latent_8q" / "train" / "weights.npz")["wr"]
out8 = R8.generate_8q(wr, 1000, 20, E.SEED + 202, GEN_N, "adaptive", eta=1.0)
gen_8q = dd8.render(out8[4][:GEN_N])


np.savez(RIP.ROOT / "merged_img_arrays.npz",
         real=real, vae=vae_rt, gen_k7=gen_k7, gen_k15=gen_k15, gen_8q=gen_8q)



FS_ROW, FS_HDR = 18, 20


def render_vertical(ncol, outpath):

    imgrows = [("real", real), ("VAE", vae_rt),
               ("Generated, 3-qubit, trainable readout", gen_k7),
               ("Generated, 4-qubit, trainable readout", gen_k15),
               ("Generated, 8-qubit, fixed readout", gen_8q)]
    nr = len(imgrows)

    WSPACE = 0.03
    HSPACE = 0.36
    LEFT, RIGHT = 0.01, 0.99
    TOP_LAB, BOT_PAD = 0.30, 0.05
    cell = (RIGHT - LEFT) * (ncol * 1.08) / (ncol + (ncol - 1) * WSPACE)
    plot_h = cell * nr + HSPACE * cell * (nr - 1)
    fig_h = plot_h + TOP_LAB + BOT_PAD
    fig = plt.figure(figsize=(ncol * 1.08, fig_h))
    gs = fig.add_gridspec(nr, ncol, wspace=WSPACE, hspace=HSPACE,
                          left=LEFT, right=RIGHT,
                          bottom=BOT_PAD / fig_h, top=(BOT_PAD + plot_h) / fig_h)
    for r, (lab, im) in enumerate(imgrows):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(im[c], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_title(lab, loc="left", fontsize=FS_ROW, pad=3)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", outpath)


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
render_vertical(8, FIGURES_DIR / "fig_merged_img_8col.png")
