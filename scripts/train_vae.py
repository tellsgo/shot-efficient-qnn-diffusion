"""train_vae.py — §VII.2 text VAE（text）。

text（text GMM text）：
  MNIST 784 --encoder--> (mu, logvar) text K text【text】 --text--> z_K ~ N(0,I_K)
  --text U(N×K, text、text)--> N text【text】text  --text--> x0 (σ=0.34, clip[-3,3])
  text DDPM text x0 text（N=2^nq -> nq qubits）
  text x0 --*s--> N text --text U^T--> z_K --decoder--> 784

text K = N-1（text）text N：
  text R^N text N-1 text。text encoder text N text，
  KL text latent ~ N(0,I_N)，text 0，text -> KL text。
  text K=N-1 text KL、text N text，text，text、text。

β text：1.0 / 1e-2 / 1e-4 / 1e-6，text"text"text β。
"""
import numpy as np, pickle, time, json
from pathlib import Path
import jax, jax.numpy as jnp
from jax import grad, jit, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results/final_image_paper/vae"
OUT.mkdir(parents=True, exist_ok=True)

DIMS = [8, 16]
BETAS = [1.0, 1e-2, 1e-4, 1e-6]
EPOCHS, BS, LR = 120, 256, 2e-3
A, SIGMA_GMM = 3.0, 0.34

mnist = np.load(Path(__file__).resolve().parent.parent / "data/mnist28x28.npz")
images = mnist["images"].astype(np.float32)[:20000] / 255.0
X = images.reshape(len(images), -1)          # (20000, 784)
Xj = jnp.asarray(X)


def zero_sum_basis(n):
    """U: (n, n-1)，text，text=0 —— text。
    P = I - 11^T/n text 1(text n-1) text 0(text 1)；text 1 text。"""
    P = np.eye(n) - np.ones((n, n)) / n
    w, V = np.linalg.eigh(P)
    U = V[:, 1:]                             # (n, n-1)
    assert np.allclose(U.sum(axis=0), 0, atol=1e-10), "text"
    assert np.allclose(U.T @ U, np.eye(n - 1), atol=1e-10), "text"
    return U


def init_params(sizes, key):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = random.split(key)
        lim = jnp.sqrt(6.0 / (a + b))
        ps.append((random.uniform(k, (a, b), minval=-lim, maxval=lim), jnp.zeros(b)))
    return ps


def mlp(ps, x, act_last=None):
    h = x
    for w, b in ps[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = ps[-1]
    o = h @ w + b
    return jax.nn.sigmoid(o) if act_last == "sigmoid" else o


def vae_loss(params, x, key, beta):
    enc_mu, enc_lv, dec = params
    mu = mlp(enc_mu, x)
    lv = jnp.clip(mlp(enc_lv, x), -8.0, 8.0)
    eps = random.normal(key, mu.shape)
    z = mu + jnp.exp(0.5 * lv) * eps
    xh = mlp(dec, z, act_last="sigmoid")
    recon = jnp.sum((x - xh) ** 2, axis=1).mean()
    kl = (-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), axis=1)).mean()
    return recon + beta * kl, (recon, kl)


@jit
def step(params, opt, x, key, t, lr, beta):
    g = grad(lambda p: vae_loss(p, x, key, beta)[0])(params)
    m, v = opt
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(lambda a, b_: b1 * a + (1 - b1) * b_, m, g)
    v = jax.tree_util.tree_map(lambda a, b_: b2 * a + (1 - b2) * b_ ** 2, v, g)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(lambda p, a, b_: p - lr * a / (jnp.sqrt(b_) + eps), params, mh, vh)
    return params, (m, v)


def np_mlp(ps, x, sigmoid=False):
    h = x
    for w, b in ps[:-1]:
        h = np.tanh(h @ w + b)
    w, b = ps[-1]
    o = h @ w + b
    return 1.0 / (1.0 + np.exp(-o)) if sigmoid else o


results = []
rows = [(images[:10], "Real")]

for N in DIMS:
    K = N - 1
    U = zero_sum_basis(N)                       # (N, K)
    nq = int(np.log2(N))
    print(f"\n{'='*70}\nN={N} text ({nq} qubit) | K={K} text | text U: {U.shape}\n{'='*70}")

    for beta in BETAS:
        key = random.PRNGKey(7)
        k1, k2, k3, key = random.split(key, 4)
        params = (init_params([784, 512, 256, K], k1),      # mu
                  init_params([784, 512, 256, K], k2),      # logvar
                  init_params([K, 256, 512, 784], k3))      # decoder(K -> 784)
        opt = (jax.tree_util.tree_map(jnp.zeros_like, params),
               jax.tree_util.tree_map(jnp.zeros_like, params))
        t, t0 = 1, time.time()
        for ep in range(EPOCHS):
            perm = np.random.permutation(len(X))
            lr = LR * (0.5 ** (ep // 40))
            for i in range(0, len(X) - BS + 1, BS):
                key, sk = random.split(key)
                params, opt = step(params, opt, Xj[perm[i:i + BS]], sk, t, lr, beta)
                t += 1
        _, (rec_f, kl_f) = vae_loss(params, Xj, random.PRNGKey(0), beta)

        enc_mu, enc_lv, dec = [[(np.asarray(w), np.asarray(b)) for w, b in p] for p in params]


        Z = np_mlp(enc_mu, X)
        Zn = Z @ U.T
        s = float(Zn.std() / SIGMA_GMM)
        x0 = np.clip(Zn / s, -A, A)
        clipr = float(np.mean(np.abs(Zn / s) > A))
        rec = np_mlp(dec, (x0 * s) @ U, sigmoid=True)
        mse = float(np.mean((X - rec) ** 2))
        zs_res = float(np.abs(Zn.mean(1)).max())

        print(f"  β={beta:<7g} textMSE={mse:.5f}  recon={float(rec_f)/784:.5f}  "
              f"KL={float(kl_f):.3f}  s={s:.3f}  clip={clipr*100:.2f}%  text={zs_res:.1e}  "
              f"({time.time()-t0:.0f}s)")

        results.append(dict(N=N, K=K, nq=nq, beta=beta, ceiling_mse=mse,
                            kl=float(kl_f), scale=s, clip_rate=clipr))
        with open(OUT / f"vae_N{N}_beta{beta:g}.pkl", "wb") as f:
            pickle.dump({"enc_mu": enc_mu, "enc_lv": enc_lv, "dec": dec,
                         "U": U, "N": N, "K": K, "beta": beta, "scale": s}, f)
        np.savez(OUT / f"latent_N{N}_beta{beta:g}.npz", x0=x0, scale=s, U=U)
        if N == 8:
            rows.append((rec[:10].reshape(10, 28, 28), f"VAE N=8\nβ={beta:g}\nMSE={mse:.4f}"))


best16 = min([r for r in results if r["N"] == 16], key=lambda r: r["ceiling_mse"])
d16 = np.load(OUT / f"latent_N16_beta{best16['beta']:g}.npz")
with open(OUT / f"vae_N16_beta{best16['beta']:g}.pkl", "rb") as f:
    p16 = pickle.load(f)
rec16 = np_mlp(p16["dec"], (d16["x0"] * d16["scale"]) @ d16["U"], sigmoid=True)
rows.append((rec16[:10].reshape(10, 28, 28), f"VAE N=16\nβ={best16['beta']:g}\nMSE={best16['ceiling_mse']:.4f}"))

fig, ax = plt.subplots(len(rows), 10, figsize=(14, 1.45 * len(rows)))
for r, (imgs, name) in enumerate(rows):
    for c in range(10):
        ax[r, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
    ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    ax[r, 0].set_ylabel(name, fontsize=6)
plt.tight_layout()
plt.savefig(OUT / "vae_beta_sweep_ceiling.png", dpi=130)
(OUT / "vae_sweep.json").write_text(json.dumps(results, indent=2))

print("\n" + "=" * 74)
print(f"{'N':>3} {'qubit':>6} {'K(text)':>10} {'beta':>8} {'textMSE':>10} {'KL':>8}")
print("=" * 74)
for r in sorted(results, key=lambda z: (z["N"], -z["beta"])):
    print(f"{r['N']:>3} {r['nq']:>6} {r['K']:>10} {r['beta']:>8g} {r['ceiling_mse']:>10.5f} {r['kl']:>8.2f}")
print("=" * 74)
print(f"\ntext: {OUT/'vae_beta_sweep_ceiling.png'}")
