"""recognizability.py — text"§VII.1 text"。

text MNIST text（text），text"text"：
  1) text：text --text->text--> text（text）
     -> text"text"
  2) text：text softmax text + text
     -> text = text；text = text
"""
import sys, math, pickle, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax import grad, jit, random
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quantum_3q_gpu as M
import noise_budgeted_generation as G
import run_image_direct as RID
from run_image_paper import PCAData, VAEData, ROOT

DATA = Path(__file__).resolve().parent.parent / "data"
d = np.load(DATA / "mnist28x28.npz")
X = d["images"].astype(np.float32).reshape(-1, 784) / 255.0
Y = d["target"].astype(np.int32)

ntr = 50000
Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:ntr + 5000], Y[ntr:ntr + 5000]


def init(sizes, key):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = random.split(key)
        lim = jnp.sqrt(6.0 / (a + b))
        ps.append((random.uniform(k, (a, b), minval=-lim, maxval=lim), jnp.zeros(b)))
    return ps


def fwd(ps, x):
    h = x
    for w, b in ps[:-1]:
        h = jax.nn.relu(h @ w + b)
    w, b = ps[-1]
    return h @ w + b


def loss(ps, x, y):
    lg = fwd(ps, x)
    return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(len(y)), y])


@jit
def step(ps, x, y, lr):
    g = grad(loss)(ps, x, y)
    return jax.tree_util.tree_map(lambda p, gg: p - lr * gg, ps, g)


print("text MNIST text (784-256-128-10)...")
ps = init([784, 256, 128, 10], random.PRNGKey(0))
Xj, Yj = jnp.asarray(Xtr), jnp.asarray(Ytr)
for ep in range(30):
    perm = np.random.permutation(ntr)
    lr = 0.05 * (0.5 ** (ep // 10))
    for i in range(0, ntr - 256, 256):
        idx = perm[i:i + 256]
        ps = step(ps, Xj[idx], Yj[idx], lr)
acc = float(jnp.mean(jnp.argmax(fwd(ps, jnp.asarray(Xte)), 1) == jnp.asarray(Yte)))
print(f"  text MNIST text = {acc*100:.1f}%\n")


def judge(pix):
    lg = fwd(ps, jnp.asarray(pix.reshape(len(pix), -1).astype(np.float32)))
    p = np.asarray(jax.nn.softmax(lg))
    return np.asarray(jnp.argmax(lg, 1)), p.max(1)


NS = 3000
rng = np.random.default_rng(5)
sub = rng.choice(ntr, NS, replace=False)
Xs, Ys = X[sub], Y[sub]

print("=" * 78)
print("【1】text：text -> text -> text -> text？（text）")
print("=" * 78)
print(f"{'text':>22} {'text':>5} {'qubit':>6} {'text':>10} {'text':>10}")
print("-" * 78)
pred, conf = judge(Xs)
print(f"{'text MNIST（text）':>20} {'784':>5} {'-':>6} {np.mean(pred==Ys)*100:>9.1f}% {conf.mean()*100:>9.1f}%")

def np_mlp(ps_, x, sig=False):
    h = x
    for w, b in ps_[:-1]:
        h = np.tanh(h @ w + b)
    w, b = ps_[-1]
    o = h @ w + b
    return 1 / (1 + np.exp(-o)) if sig else o



dpca = PCAData(8, False)
z = dpca.pca.transform(Xs.astype(np.float64))
if dpca.whiten:
    z = z / dpca.sd
x0p = np.clip((z @ dpca.U.T) / dpca.s, -3, 3)
pca_rec = dpca.render(x0p)

with open(ROOT / "vae/vae_N8_beta0.01.pkl", "rb") as f:
    vp = pickle.load(f)
dvae = VAEData(8, 0.01)
zv = np_mlp(vp["enc_mu"], Xs.astype(np.float64))
x0v = np.clip((zv @ vp["U"].T) / vp["scale"], -3, 3)
vae_rec = dvae.render(x0v)

for name, Xp in [("PCA-7  (§VII.1)", pca_rec), ("VAE-7  (§VII.2)", vae_rec)]:
    pr, cf = judge(np.clip(Xp, 0, 1))
    print(f"{name:>22} {8:>5} {3:>6} {np.mean(pr==Ys)*100:>9.1f}% {cf.mean()*100:>9.1f}%")

print("\n" + "=" * 78)
print("【2】text：text？")
print("=" * 78)
print(f"{'text':>34} {'text':>10} {'text>90%text':>12} {'text':>8}")
print("-" * 78)

NG = 1500
CFG = [
    ("PCA-7->8, 3-qubit diagW", PCAData(8, False),
     ROOT / "sec7_1_direct_pca/d8_diagW", "diagW", 8, M.SEED + 202),
    ("VAE-7->8, 3-qubit diagW", VAEData(8, 0.01),
     ROOT / "sec7_2_latent_3q/d8_diagW", "diagW", 8, M.SEED + 202),
    ("VAE-15->16, 4-qubit diagW", VAEData(16, 0.01),
     ROOT / "sec7_2_latent_3q/d16_diagW", "diagW", 16, M.SEED + 909),
]
for tag, dd, dpath, ro, N, seed in CFG:
    M.configure_gpu(N)
    RID.DEALIAS_SCALE = math.pi / (0.6 * M.N + 0.15 * M.N * M.N)
    w = np.load(dpath / "train" / "weights.npz")
    theta = {k: w[k] for k in w.files if k != "mode"}
    _, sh, _, _, x0h = RID.generate(theta, ro, 1000, 20, seed, NG,
                                     "adaptive", eta=1.0)
    pr, cf = judge(np.clip(dd.render(x0h), 0, 1))
    cov = len(np.unique(pr))
    print(f"{tag:>34} {cf.mean()*100:>9.1f}% {np.mean(cf>0.9)*100:>11.1f}% {cov:>6}/10")


print(f"{'（text）text MNIST':>34} {conf.mean()*100:>9.1f}% {np.mean(conf>0.9)*100:>11.1f}% {'10':>6}/10")
print("\ntext：text MNIST，text；")
print("      text = text = text。")
