"""gmm_stats.py — text §VI text：W text / Frobenius text、high-shot text。"""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_T1000_mainline as MAIN
import run_quantum_3q_gpu as M
import noise_budgeted_generation as G
import run_quantum_x0 as Q

ROOT = Path("results/final_gmm_diagW_pi")
M.configure_gpu(8)
MAIN.DEALIAS_SCALE = math.pi / (0.6 * M.N + 0.15 * M.N * M.N)

print("=" * 72)
print("text W text（AN = 3x8 = 24 text）")
print("=" * 72)
print(f"{'readout':>10} {'text':>14} {'||W||_2 text':>15} {'||W||_F':>10}")
print("-" * 72)
for mode in ["fixed24I", "diagW", "fullW"]:
    theta = MAIN._load_theta(ROOT, mode)
    Wr, br = M.readout_numpy(theta, mode)
    s2 = float(np.linalg.norm(Wr, 2))
    sf = float(np.linalg.norm(Wr, "fro"))
    npar = {"fixed24I": 0, "diagW": M.N, "fullW": M.N * M.N}[mode]
    print(f"{mode:>10} {npar:>14} {s2:>15.2f} {sf:>10.2f}")

print("\n" + "=" * 72)
print("rep=20, eta=1 : adaptive vs high-shot（text）")
print("=" * 72)
xis = G.psd_sqrt(G.projection(M.N))[0]
C = G.mixture_centers(M.N)
target = G.sample_target(np.random.default_rng(M.SEED + 99), 3000, C, M.S0, xis)
gamma = 1.0 / G.median_bandwidth(target, target, np.random.default_rng(1))
print(f"{'readout':>10} {'mode':>10} {'rf':>8} {'MMD2':>10} {'shots':>10}")
print("-" * 72)
for mode in ["diagW", "fullW", "fixed24I"]:
    theta = MAIN._load_theta(ROOT, mode)
    for sm, kw in [("adaptive", {}), ("high", dict(high_M=80000))]:
        x, sh, _, _, sat = MAIN.generate(theta, mode, 1000, 20, M.SEED + 202, 3000,
                                         eta=1.0, shot_mode=sm, **kw)
        rf = Q.resolved_frac(x, C); mmd = float(G.mmd2_rbf(x, target, gamma))
        print(f"{mode:>10} {sm:>10} {rf:>8.3f} {mmd:>10.5f} {sh:>10.0f}")
