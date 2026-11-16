"""run_quantum_x0.py — text：text PQC x0-prediction (data re-uploading)
                      + W text PSD shot text  + text

text（text，text quddpm conda text）:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate quddpm
  python3 scripts/run_quantum_x0.py --seed 42

text（text，text high-shot rf=0.92）:
  3 qubits, AmplitudeEmbedding + L=12 data-re-uploading layers
  Features: [1, 1.4*(8*q−1), τ]  (amplified data-space features)
  Trainable readout: full 8×8 W (identity init) + bias
  Adam lr=0.02 → 0.005 (step 9000)

text，text:
  high_shot  : text shot（≈denoiser text，text rf≈0.92 text）
  adaptive   : W text PSD text shot（text：text shot text）
  uniform    : text adaptive text（text，text）

text（text results/experiments/YYYY-MM-DD/quantum_x0_seed<seed>/）:
  training_loss.png          — text
  scatter.png                — text: Target | high_shot | adaptive | uniform
  denoising_process.png      — adaptive text（step 0..12）
  per_step_shots.csv/.png    — adaptive text shot text
  training_loss.csv          — text
  weights.npz                — text
  gen_high.npy / gen_adaptive.npy / gen_uniform.npy / target.npy
  snapshots_adaptive/step*.npy — adaptive text
  metrics.txt                — text resolved_frac / shot / chi-MSE text

W text（text）:
  MLP text chi=A·n·(p−1/n): shot text = (A·n)²·Σ/M
  text x0=P(p·W+b):  shot text = P·Wᵀ·Σ·W·P/M  (text 8×8)
  → PSD text: R = Ξ − (k²/σ²)·(P·Wᵀ·Σ·W·P)/M,  M ≥ (k²/σ²)·λ_max(P·Wᵀ·Σ·W·P)
  W=A·n·I text MLP text; M text R≈Ξ text（high-shot text）。
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from datetime import date
from pathlib import Path

import numpy as _np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_budgeted_generation as G

import pennylane as qml
from pennylane import numpy as np  # autograd-capable numpy

# ── constants (dimension/qubits set at runtime by configure()) ─────────────────

N_DATA = 8       # data dimension (2^NQ); set by configure()
NQ = 3           # qubits (log2 N_DATA); set by configure()
A = 3.0          # affine bound
S0 = 0.12        # GMM cluster std
STEPS = 12       # diffusion steps
BMAX, BMIN = 0.95, 0.10
FDIM = N_DATA + 2   # re-uploading feature dim: [1, 1.4*(Nq-1), τ]; set by configure()
NGEN = 600       # samples generated per mode (override with --gen-samples)

dev = None       # pennylane device; built by configure()
circ = None      # qnode; built by configure()

def configure(n_data):
    """Set data dimension / qubit count and (re)build the circuit accordingly.
    Called once at startup so the same script serves GMM (8D) and digits (16D)."""
    global N_DATA, NQ, FDIM, dev, circ
    N_DATA = n_data
    NQ = int(round(math.log2(n_data)))
    assert 2 ** NQ == n_data, f"dim must be a power of 2 (got {n_data})"
    FDIM = N_DATA + 2
    dev = qml.device("default.qubit", wires=NQ)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def _circ(wrot, state, feat):
        """Data re-uploading PQC.
        wrot: (L, NQ, 2, FDIM)  state: (batch, 2^NQ)  feat: (batch, FDIM)"""
        qml.AmplitudeEmbedding(state, wires=range(NQ), normalize=True)
        L = wrot.shape[0]
        for l in range(L):
            for qb in range(NQ):
                ay = np.sum(wrot[l, qb, 0] * feat, axis=-1)
                az = np.sum(wrot[l, qb, 1] * feat, axis=-1)
                qml.RY(ay, wires=qb)
                qml.RZ(az, wires=qb)
            for qb in range(NQ):
                qml.CNOT(wires=[qb, (qb + 1) % NQ])
        return qml.probs(wires=range(NQ))

    circ = _circ

# ── input encoding (ablation: amp/prob/hybrid × time on/off) ───────────────────

EMBEDDING = "hybrid"   # amplitude | probability | hybrid; set by set_encoding()
TIME_ON = True         # whether the time step τ enters the network; set by set_encoding()

def set_encoding(embedding, time_on):
    """Select input-encoding mode and whether time embedding is used.
    Mirrors the teacher's basic_qmodule.py knobs qubit_embedding_method / use_time_embedding:
      amplitude   : state amplitudes = data q directly (no sqrt)
      probability : amplitudes = √q  → Born probs recover q
      hybrid      : amplitudes = √q · e^{iφ(τ)}  (probability + time phase)
    time_on=False zeroes the τ feature channel; hybrid then loses its phase and
    coincides with probability (teacher's documented degeneracy)."""
    global EMBEDDING, TIME_ON
    assert embedding in ("amplitude", "probability", "hybrid")
    EMBEDDING = embedding
    TIME_ON = bool(time_on)
    if EMBEDDING == "hybrid" and not TIME_ON:
        print("Note: hybrid without time embedding degrades to probability "
              "(phase e^{i·0}=1); this run is numerically identical to probability/off.")

def make_state(q, tau):
    """Prepare the amplitude-embedded input state per the selected encoding."""
    if EMBEDDING == "amplitude":
        st = _np.maximum(q, 1e-12).astype(complex)      # data as amplitudes directly
    else:
        st = _np.sqrt(_np.maximum(q, 1e-12)).astype(complex)  # √q  (probability / hybrid)
    if EMBEDDING == "hybrid" and TIME_ON:
        idx = _np.arange(1, N_DATA + 1)
        ph = 0.6 * idx * tau[:, None] + 0.15 * idx * idx * tau[:, None]
        st = st * _np.exp(1j * ph)                        # time enters as a quantum phase
    return st / _np.linalg.norm(st, axis=1, keepdims=True)

def make_feat(q, tau):
    """Amplified data-space features — breaks narrow-band degeneracy.
    The τ channel is zeroed when TIME_ON is False (time embedding removed);
    FDIM stays fixed so weight shapes are unchanged across ablation runs."""
    tcol = tau[:, None] if TIME_ON else _np.zeros((len(q), 1))
    return _np.concatenate([
        _np.ones((len(q), 1)),
        1.4 * (N_DATA * q - 1.0),
        tcol,
    ], axis=1)

def zs(x):
    return x - np.mean(x, axis=-1, keepdims=True)

# ── digits dataset (PCA -> zero-sum -> scaled to GMM signal level) ──────────────

class DigitsData:
    """Real sklearn digits mapped into the bounded zero-sum hyperplane.
    Pipeline: load_digits -> StandardScaler -> PCA(dim) -> zero-sum projection
    -> scale so the signal level matches GMM (x0 std ~ GMM), leaving noise headroom.
    Provides sample_x0() for training/target and render() for image visualization."""
    def __init__(self, dim, target_std=None):
        from sklearn.datasets import load_digits
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        d = load_digits()
        self.raw = d.data.astype(float); self.y = d.target
        self.scaler = StandardScaler().fit(self.raw)
        Xs = self.scaler.transform(self.raw)
        self.pca = PCA(dim).fit(Xs)
        Z = self.pca.transform(Xs)
        Xi = _np.eye(dim) - _np.ones((dim, dim)) / dim
        z = Z @ Xi.T                                  # zero-sum PCA coords
        # scale so x0 std matches GMM's (target_std), keeping room for diffusion noise
        gmm_std = target_std if target_std is not None else float(_np.std(G.mixture_centers(8)))
        self.scale = float(_np.std(z)) / gmm_std
        self.x0_all = z / self.scale                  # zero-sum, GMM-scale
        clip = _np.mean(_np.abs(self.x0_all) > A) * 100
        print(f"      [digits] dim={dim} x0 std={_np.std(self.x0_all):.3f} "
              f"(GMM {gmm_std:.3f}) range[{self.x0_all.min():.2f},{self.x0_all.max():.2f}] clip={clip:.2f}%")

    def sample_x0(self, rng, count):
        idx = rng.integers(0, len(self.x0_all), size=count)
        return _np.clip(self.x0_all[idx], -A, A)

    def render(self, x0_batch):
        """x0 (n,dim) zero-sum GMM-scale -> (n,64) image space for visualization."""
        Z = x0_batch * self.scale
        return self.scaler.inverse_transform(self.pca.inverse_transform(Z))


# ── dataset ────────────────────────────────────────────────────────────────────

def build_dataset(rng, count, C, xis, digits=None):
    """Build (q, tau, x0) training pairs. GMM: sample from mixture C.
    digits: sample real x0 from DigitsData instead of the GMM."""
    pairs = G.schedule(STEPS, BMAX, BMIN)
    qs, ts, x0s = [], [], []
    att = 0
    while len(qs) < count and att < count * 30:
        att += 1
        sid = int(rng.integers(0, STEPS))
        al, be, _, _ = pairs[sid]
        if digits is not None:
            x0 = digits.sample_x0(rng, 1)
        else:
            x0 = G.sample_target(rng, 1, C, S0, xis)
        w = rng.normal(size=(1, N_DATA)) @ xis.T
        xt = G.project_zero_sum(al * x0 + be * w)
        q, bad = G.chi_to_p(xt, A)
        _, bad2 = G.chi_to_p(x0, A)
        if bad or bad2:
            continue
        qs.append(q[0])
        ts.append(sid / max(STEPS - 1, 1))
        x0s.append(x0[0])
    return _np.array(qs), _np.array(ts), _np.array(x0s)

# ── training ───────────────────────────────────────────────────────────────────

def train(wr, Wr, br, states, feats, x0_tg, *, iters, batch, lr0, lr_decay_at, log_every):
    var = float(_np.mean(x0_tg ** 2))

    def loss(wr, Wr, br, st, ft, tg):
        return np.mean((zs(circ(wr, st, ft) @ Wr + br) - tg) ** 2) / var

    gf = qml.grad(loss, argnum=(0, 1, 2))

    mw = _np.zeros_like(wr); vw = _np.zeros_like(wr)
    mR = _np.zeros_like(Wr); vR = _np.zeros_like(Wr)
    mb = _np.zeros_like(br); vb = _np.zeros_like(br)
    b1, b2c, eps = 0.9, 0.999, 1e-8
    t0 = time.time()
    hist = []
    rec_every = max(iters // 150, 1)   # record ~150 points -> smooth loss curve

    for it in range(1, iters + 1):
        lr = lr0 if it < lr_decay_at else lr0 * 0.25
        idx = _np.random.default_rng(it).integers(0, len(states), size=batch)
        g = gf(wr, Wr, br, states[idx], feats[idx], x0_tg[idx])
        mw = b1 * mw + (1 - b1) * g[0]; vw = b2c * vw + (1 - b2c) * g[0] ** 2
        mR = b1 * mR + (1 - b1) * g[1]; vR = b2c * vR + (1 - b2c) * g[1] ** 2
        mb = b1 * mb + (1 - b1) * g[2]; vb = b2c * vb + (1 - b2c) * g[2] ** 2
        wr = wr - lr * (mw / (1 - b1 ** it)) / (_np.sqrt(vw / (1 - b2c ** it)) + eps)
        Wr = Wr - lr * (mR / (1 - b1 ** it)) / (_np.sqrt(vR / (1 - b2c ** it)) + eps)
        br = br - lr * (mb / (1 - b1 ** it)) / (_np.sqrt(vb / (1 - b2c ** it)) + eps)
        if it % rec_every == 0 or it == 1:                          # record for the curve
            full_loss = float(loss(wr, Wr, br, states, feats, x0_tg)) * var
            hist.append({"iteration": it, "chi_mse": full_loss})
            if it % log_every == 0 or it == 1:                      # print less often
                print(f"  [qpqc] {it:>6d}/{iters}  chi-MSE={full_loss:.5f}  ({time.time()-t0:.0f}s)")

    return wr, _np.array(Wr), _np.array(br), hist

# ── W-generalized PSD shot budget ─────────────────────────────────────────────

def _WSW(pi, Wr, P):
    """P·Wᵀ·Σ(pi)·W·P  — shot-noise covariance of the trainable readout (×M)."""
    S = G.multinomial_cov(pi)
    M = P @ Wr.T @ S @ Wr @ P
    return 0.5 * (M + M.T)  # symmetrize for numerical safety

def min_shots_W(p, Wr, kc, sig, P):
    """Smallest M s.t. R = Ξ − (kc²/σ²)·WSW/M ⪰ 0, per sample."""
    out = []
    coef = kc ** 2 / max(sig, 1e-12) ** 2
    for pi in p:
        lam = float(_np.linalg.eigvalsh(_WSW(pi, Wr, P)).max())
        out.append(math.ceil(coef * max(lam, 0.0)))
    return _np.asarray(out, dtype=_np.int64)

def residual_W(rng, p, shots, Wr, kc, sig, P):
    """Sample corrected residual: cov R = Ξ − (kc²/(σ²·M))·WSW."""
    out = _np.empty((len(p), N_DATA))
    coef = kc ** 2 / max(sig, 1e-12) ** 2
    for i, (pi, mi) in enumerate(zip(p, shots)):
        R = P - (coef / int(mi)) * _WSW(pi, Wr, P)
        sq, _, _ = G.psd_sqrt(R)
        out[i] = sq @ rng.normal(size=N_DATA)
    return out

# ── generation (all modes share one budgeted path) ────────────────────────────

def generate(wr_, Wr_, br_, *, mode, seed, m_unif=0, m_high=80000,
             m_min=32, m_max=200000, keep_snapshots=False):
    rng = _np.random.default_rng(seed)
    P = G.projection(N_DATA)              # Ξ = zero-sum projection
    xis = G.psd_sqrt(P)[0]
    x = rng.normal(size=(NGEN, N_DATA)) @ xis.T
    total = _np.zeros(NGEN)
    per_step = []                          # mean shots each step (adaptive/actual)
    snaps = [x.copy()] if keep_snapshots else None

    for sid, (al, be, aln, ben) in enumerate(G.schedule(STEPS, BMAX, BMIN), 1):
        sig, dirc, ct = G.compute_ddim_params(al, be, aln, ben, 1.0)
        qb, _ = G.chi_to_p(x, A)
        tau = (sid - 1) / max(STEPS - 1, 1) * _np.ones(NGEN)
        p = _np.array(circ(wr_, make_state(qb, tau), make_feat(qb, tau)))
        kc = aln - dirc * al / max(be, 1e-9)   # x0-prediction step coefficient

        req = min_shots_W(p, Wr_, kc, sig, P)
        if mode == "adaptive":
            sh = _np.clip(_np.ceil(1.10 * req), m_min, m_max).astype(_np.int64)
        elif mode == "high_shot":
            sh = _np.full(NGEN, m_high, dtype=_np.int64)
        elif mode == "uniform":
            sh = _np.full(NGEN, m_unif, dtype=_np.int64)
        else:  # fixed_low
            sh = _np.full(NGEN, 64, dtype=_np.int64)
        sh = _np.maximum(sh, 2)
        per_step.append(float(_np.mean(sh)))

        ph = G.empirical_born_readout(rng, p, sh)
        x0h = G.project_zero_sum(ph @ Wr_ + br_)
        ev = (x - al * x0h) / max(be, 1e-9)
        res = residual_W(rng, p, sh, Wr_, kc, sig, P)
        x = aln * x0h + dirc * ev + sig * res
        x = G.project_zero_sum(_np.clip(x, -A, A))
        total += sh
        if keep_snapshots:
            snaps.append(x.copy())

    return x, float(_np.mean(total)), per_step, snaps

# ── resolved_frac ─────────────────────────────────────────────────────────────

def resolved_frac(X, C):
    cc = _np.linalg.norm(C[:, None] - C[None], axis=2)
    inter = cc[cc > 0].min()
    d = _np.linalg.norm(X[:, None] - C[None], axis=2)
    return float(_np.mean(d.min(1) < inter / 2))

# ── visualization helpers ──────────────────────────────────────────────────────

def _pca_basis(C):
    cm = C.mean(0)
    _, _, vh = _np.linalg.svd(C - cm, full_matrices=False)
    return cm, vh[:2].T

def save_scatter(od, target, gens, C, mmd_fn=None):
    """gens: list of (label, X, rf, shots).  mmd_fn(X)->MMD² added to titles if given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm, B = _pca_basis(C)
    def proj(X): return (X - cm) @ B
    cp = proj(C)
    rf_tgt = resolved_frac(target, C)
    panels = [("Target", target, rf_tgt, None)] + gens
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    cols = ["#555555", "#27ae60", "#2970b0", "#c0392b"]
    for ax, (label, X, rf, sh), color in zip(axes, panels, cols):
        P2 = proj(X)
        ax.scatter(P2[:, 0], P2[:, 1], s=8, c=color, alpha=0.45, edgecolors="none")
        ax.scatter(cp[:, 0], cp[:, 1], s=200, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
        for i, (x, y) in enumerate(cp, 1):
            ax.annotate(f"c{i}", (x, y), textcoords="offset points", xytext=(8, 6), fontweight="bold", fontsize=10)
        t = f"{label}\nrf={rf:.2f}"
        if mmd_fn is not None:
            t += f" / MMD2={mmd_fn(X):.4f}"
        if sh:
            t += f"\n{sh:.0f} shots"
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("PC1"); ax.grid(alpha=0.25); ax.set_aspect("equal", "datalim")
    axes[0].set_ylabel("PC2")
    fig.suptitle("GMM 8D — Quantum PQC (data re-uploading) x0-prediction  |  W-generalized adaptive shot budget\n"
                 "resolved_frac >= 0.9 -> 3 clean clusters   (judge by SPACE not color)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = od / "scatter.png"; fig.savefig(out, dpi=130); plt.close()
    print(f"  scatter saved -> {out}")

def save_denoising_process(od, snaps, C, per_step, mode_label):
    """Grid of per-step point clouds showing the reverse-diffusion trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm, B = _pca_basis(C)
    def proj(X): return (X - cm) @ B
    cp = proj(C)
    # snaps has STEPS+1 entries (step 0..STEPS). show all if <=13 else subsample
    n_snap = len(snaps)
    idxs = list(range(n_snap))
    ncol = min(n_snap, 7)
    nrow = math.ceil(n_snap / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.7 * nrow))
    axes = _np.array(axes).reshape(-1)
    for k, si in enumerate(idxs):
        ax = axes[k]
        P2 = proj(snaps[si])
        ax.scatter(P2[:, 0], P2[:, 1], s=5, c="#27ae60", alpha=0.4, edgecolors="none")
        ax.scatter(cp[:, 0], cp[:, 1], s=90, facecolors="none", edgecolors="black", linewidths=1.4, zorder=5)
        rf = resolved_frac(snaps[si], C)
        shot_txt = "" if si == 0 else f"\n{per_step[si-1]:.0f} sh"
        ax.set_title(f"step {si}  rf={rf:.2f}{shot_txt}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", "datalim")
    for k in range(len(idxs), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Quantum PQC reverse-diffusion denoising process ({mode_label})\n"
                 "step 0 = pure noise  ->  step 12 = generated;  per-step avg shots shown",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = od / "denoising_process.png"; fig.savefig(out, dpi=125); plt.close()
    print(f"  denoising_process saved -> {out}")

def save_loss_curve(od, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["iteration"] for h in hist], [h["chi_mse"] for h in hist], color="#27ae60", linewidth=1.2)
    ax.set_yscale("log")  # loss spans 5.8 -> 0.07; log-y shows a smooth decay
    ax.set_xlabel("Iteration"); ax.set_ylabel("chi-MSE (x0 target, log scale)")
    ax.set_title("Quantum PQC (data re-uploading) Training Loss")
    ax.grid(alpha=0.3, which="both"); fig.tight_layout()
    out = od / "training_loss.png"; fig.savefig(out, dpi=130); plt.close()
    print(f"  training_loss saved -> {out}")

def save_per_step_shots(od, per_step, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = list(range(1, len(per_step) + 1))
    with open(od / "per_step_shots.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "avg_shots"])
        for s, sh in zip(steps, per_step): w.writerow([s, f"{sh:.1f}"])
        w.writerow(["total_per_sample", f"{total:.1f}"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(steps, per_step, color="#2970b0", alpha=0.8)
    ax.set_xlabel("Diffusion step"); ax.set_ylabel("Avg shots / sample")
    ax.set_title(f"Adaptive per-step shot budget (total ≈ {total:.0f}/sample)")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    out = od / "per_step_shots.png"; fig.savefig(out, dpi=130); plt.close()
    print(f"  per_step_shots saved -> {out} (+ .csv)")

# ── digits image visualization ─────────────────────────────────────────────────

def save_digit_grid(od, digits, target, gens, ncol=10):
    """Rows: real digits + each generation mode, each rendered as 8x8 images.
    gens: list of (label, X, shots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("real digits", target, None)] + gens
    imgs_rows = [digits.render(X[:ncol]) for _, X, _ in rows]
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.2 * ncol, 1.35 * len(rows)))
    for r, ((label, _, sh), imgs) in enumerate(zip(rows, imgs_rows)):
        for c in range(ncol):
            ax = axes[r, c]; ax.imshow(imgs[c].reshape(8, 8), cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
        ylab = label if sh is None else f"{label}\n{sh:.0f} sh"
        axes[r, 0].set_ylabel(ylab, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Digits generation — real vs generated (8x8 render)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = od / "digit_grid.png"; fig.savefig(out, dpi=130); plt.close()
    print(f"  digit_grid saved -> {out}")

def save_digit_denoise(od, digits, snaps, per_step, mode_label, n_examples=4):
    """Rows: a few example samples; cols: rendered image after each diffusion step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_snap = len(snaps)
    fig, axes = plt.subplots(n_examples, n_snap, figsize=(1.1 * n_snap, 1.25 * n_examples))
    for e in range(n_examples):
        for s in range(n_snap):
            img = digits.render(snaps[s][e:e+1])[0]
            ax = axes[e, s]; ax.imshow(img.reshape(8, 8), cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            if e == 0:
                sh = "" if s == 0 else f"\n{per_step[s-1]:.0f}sh"
                ax.set_title(f"step{s}{sh}", fontsize=7)
    fig.suptitle(f"Digits reverse-diffusion denoising ({mode_label})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = od / "digit_denoise.png"; fig.savefig(out, dpi=125); plt.close()
    print(f"  digit_denoise saved -> {out}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global NGEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gmm", "digits"], default="gmm",
                    help="gmm (8D, rf metric) or digits (real data, MMD2 + image render)")
    ap.add_argument("--dim", type=int, default=None,
                    help="data dimension = 2^qubits (default: gmm=8, digits=32)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=int, default=6000)
    ap.add_argument("--gen-samples", type=int, default=NGEN,
                    help="samples generated per mode (default 3000; higher = less MMD² noise)")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr-decay-at", type=int, default=9000)
    ap.add_argument("--m-highshot", type=int, default=80000)
    ap.add_argument("--m-min", type=int, default=32)
    ap.add_argument("--m-max", type=int, default=200000)
    ap.add_argument("--load-weights", type=Path, default=None,
                    help="skip training, load weights.npz (for re-running generation only)")
    ap.add_argument("--embedding", choices=["amplitude", "probability", "hybrid"],
                    default="hybrid", help="input encoding (ablation axis)")
    ap.add_argument("--time-embedding", choices=["on", "off"], default="on",
                    help="whether the diffusion time step enters the network (ablation axis)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    NGEN = args.gen_samples
    set_encoding(args.embedding, args.time_embedding == "on")
    print(f"Encoding = {EMBEDDING}, time_embedding = {'on' if TIME_ON else 'off'}")

    # dimension: default gmm=8, digits=32 (5 qubits; 16D too blurry to read); build circuit
    dim = args.dim if args.dim is not None else (8 if args.dataset == "gmm" else 32)
    configure(dim)
    is_digits = args.dataset == "digits"

    if args.out_dir is None:
        today = date.today().isoformat()
        tag = f"quantum_{args.dataset}{'' if args.dataset=='gmm' else dim}_seed{args.seed}"
        args.out_dir = Path(f"results/experiments/{today}/{tag}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = args.out_dir / "snapshots_adaptive"
    snap_dir.mkdir(exist_ok=True)
    print(f"Output -> {args.out_dir}  (dataset={args.dataset}, dim={dim}, qubits={NQ})")

    P = G.projection(N_DATA)
    xis = G.psd_sqrt(P)[0]
    C = G.mixture_centers(N_DATA) if not is_digits else None
    digits = DigitsData(dim) if is_digits else None

    # ── train (or load) ──
    if args.load_weights is not None:
        print(f"[1/5] Loading weights from {args.load_weights}...")
        d = _np.load(args.load_weights)
        wr = np.array(d["wr"], requires_grad=False)
        Wr_ = d["Wr"]; br_ = d["br"]
        final_chi_mse = float("nan"); hist = []
    else:
        print(f"[1/5] Building dataset ({args.data} samples)...")
        rng = _np.random.default_rng(args.seed)
        q, ta, x0 = build_dataset(rng, args.data, C, xis, digits=digits)
        print(f"      got {len(q)}/{args.data}  x0_var={float(_np.mean(x0**2)):.4f}")
        states = make_state(q, ta); feats = make_feat(q, ta)
        L = args.layers
        wr = np.array(0.1 * _np.random.default_rng(args.seed + 1).standard_normal((L, NQ, 2, FDIM)),
                      requires_grad=True)
        Wr = np.array(A * N_DATA * _np.eye(N_DATA), requires_grad=True)
        br = np.array(-A * _np.ones(N_DATA), requires_grad=True)
        print(f"[2/5] Training quantum PQC ({args.iters} iters, L={L}) ~15-30 min...")
        wr, Wr_, br_, hist = train(wr, Wr, br, states, feats, x0,
                                    iters=args.iters, batch=args.batch,
                                    lr0=args.lr, lr_decay_at=args.lr_decay_at,
                                    log_every=max(args.iters // 15, 1))
        final_chi_mse = hist[-1]["chi_mse"] if hist else float("nan")
        save_loss_curve(args.out_dir, hist)
        with open(args.out_dir / "training_loss.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["iteration", "chi_mse"]); w.writeheader(); w.writerows(hist)
        _np.savez(args.out_dir / "weights.npz", wr=_np.array(wr), Wr=Wr_, br=br_)
        print(f"  weights saved -> {args.out_dir}/weights.npz")

    wr_np = _np.array(wr)

    # ── generate: high_shot, adaptive (with snapshots), uniform ──
    if is_digits:
        target = digits.sample_x0(_np.random.default_rng(args.seed + 99), NGEN)
    else:
        target = G.sample_target(_np.random.default_rng(args.seed + 99), NGEN, C, S0, xis)
    _np.save(args.out_dir / "target.npy", target)
    _gamma = 1.0 / G.median_bandwidth(target, target, _np.random.default_rng(1))
    def mmd_fn(X): return G.mmd2_rbf(X, target, _gamma)
    def rf_of(X): return float("nan") if is_digits else resolved_frac(X, C)

    print("[3/5] Generating high_shot (ceiling reference)...")
    x_high, sh_high, _, _ = generate(wr_np, Wr_, br_, mode="high_shot",
                                     seed=args.seed + 201, m_high=args.m_highshot)
    rf_high = rf_of(x_high); _np.save(args.out_dir / "gen_high.npy", x_high)
    print(f"      high_shot: rf={rf_high:.3f}  MMD2={mmd_fn(x_high):.4f}  shots/sample={sh_high:.0f}")

    print("[4/5] Generating adaptive (W-budget, keeping per-step snapshots)...")
    x_adap, sh_adap, per_step, snaps = generate(wr_np, Wr_, br_, mode="adaptive",
                                                seed=args.seed + 202, m_min=args.m_min,
                                                m_max=args.m_max, keep_snapshots=True)
    rf_adap = rf_of(x_adap); _np.save(args.out_dir / "gen_adaptive.npy", x_adap)
    for si, sx in enumerate(snaps):
        _np.save(snap_dir / f"step{si:02d}.npy", sx)
    print(f"      adaptive: rf={rf_adap:.3f}  MMD2={mmd_fn(x_adap):.4f}  shots/sample={sh_adap:.0f}")
    print("      per-step avg shots:")
    for si, sh in enumerate(per_step, 1):
        print(f"        step {si:2d}: {sh:8.0f} shots")

    m_unif = max(int(round(sh_adap / STEPS)), 2)
    print(f"[5/5] Generating uniform same-budget ({m_unif}/step)...")
    x_unif, sh_unif, _, _ = generate(wr_np, Wr_, br_, mode="uniform",
                                     seed=args.seed + 203, m_unif=m_unif)
    rf_unif = rf_of(x_unif); _np.save(args.out_dir / "gen_uniform.npy", x_unif)
    print(f"      uniform: rf={rf_unif:.3f}  MMD2={mmd_fn(x_unif):.4f}  shots/sample={sh_unif:.0f}")

    # ── figures (dataset-dependent) ──
    if is_digits:
        save_digit_grid(args.out_dir, digits, target, [
            ("high_shot", x_high, sh_high),
            ("ADAPTIVE", x_adap, sh_adap),
            ("uniform", x_unif, sh_unif),
        ])
        save_digit_denoise(args.out_dir, digits, snaps, per_step, f"adaptive, {sh_adap:.0f} shots/sample")
    else:
        save_scatter(args.out_dir, target, [
            ("Quantum high_shot", x_high, rf_high, sh_high),
            ("Quantum ADAPTIVE (W-budget)", x_adap, rf_adap, sh_adap),
            ("Quantum uniform same-budget", x_unif, rf_unif, sh_unif),
        ], C, mmd_fn=mmd_fn)
        save_denoising_process(args.out_dir, snaps, C, per_step, f"adaptive, {sh_adap:.0f} shots/sample")
    save_per_step_shots(args.out_dir, per_step, sh_adap)

    mmd_high, mmd_adap, mmd_unif = mmd_fn(x_high), mmd_fn(x_adap), mmd_fn(x_unif)

    # ── metrics ──
    rf_note = ("rf=NaN (digits: no centers, judge by image + MMD2)"
               if is_digits else "rf>=0.9 -> 3 clusters")
    lines = [
        f"dataset={args.dataset}  dim={dim}  qubits={NQ}  seed={args.seed}  layers={args.layers}  iters={args.iters}",
        f"final_chi_mse={final_chi_mse:.5f}",
        "",
        f"high_shot : rf={rf_high:.3f}  MMD2={mmd_high:.4f}  shots/sample={sh_high:.0f}   (ceiling ref)",
        f"ADAPTIVE  : rf={rf_adap:.3f}  MMD2={mmd_adap:.4f}  shots/sample={sh_adap:.0f}   (W-generalized PSD budget)",
        f"uniform   : rf={rf_unif:.3f}  MMD2={mmd_unif:.4f}  shots/sample={sh_unif:.0f}   (same budget as adaptive)",
        "",
        f"speedup: adaptive uses {sh_adap:.0f} vs high_shot {sh_high:.0f} shots/sample "
        f"(~{sh_high/max(sh_adap,1):.0f}x cheaper)",
        f"NOTE: {rf_note}. adaptive should beat uniform at the same budget (paper's claim).",
    ]
    (args.out_dir / "metrics.txt").write_text("\n".join(lines))
    print("\n=== RESULTS ===")
    for l in lines: print("  " + l)
    print(f"\n  Done. All outputs in {args.out_dir}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
