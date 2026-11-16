# -*- coding: utf-8 -*-
# ============================================================================





#


#

#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate quddpm_gpu
#   python scripts/plotting/make_fig_bounds_singlecol.py
#



# ============================================================================
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter


FS_LABEL, FS_TICK = 20, 18
RF_YLIM = (0.65, 1.0)
XTICKS_A = [2, 8, 32, 64]
XTICKS_B = [8, 32, 128, 256]
LOG_SHOTS = False
SHOTS_YTICKS = [300, 400, 600, 800, 1000]
SHOTS_YTICKS_LIN = [300, 600, 900, 1200]
WSPACE = 0.02
FIGSIZE = (8.8, 3.9)
# -------------------------

BLUE, ORANGE = "#2970b0", "#c85a11"
SAVE = dict(dpi=150, bbox_inches="tight")
HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BASE = SCRIPTS_DIR / "results" / "final_gmm_diagW_pi" / "diagW" / "bounds"
OUT = PROJECT_ROOT / "results" / "figures" / "fig_gmm_bounds.png"


def load(path, xkey):
    x, rf, shots, cap = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            x.append(float(r[xkey])); rf.append(float(r["rf"]))
            shots.append(float(r["total_shots"])); cap.append(float(r["cap_frac"]))
    return x, rf, shots, cap


mmin_x, mmin_rf, mmin_sh, _ = load(f"{BASE}/bounds_mmin_diagW.csv", "m_min")
mmax_x, mmax_rf, _, mmax_cap = load(f"{BASE}/bounds_mmax_diagW.csv", "m_max")

fig, (axA, axB) = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained", sharey=True)


def sparse_x(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=FS_TICK)



axA.plot(mmin_x, mmin_rf, "-o", color=BLUE, lw=2.6, ms=8)
axA.set_ylim(*RF_YLIM)
axA.set_xlabel(r"Lower bound  $M_{\min}$", fontsize=FS_LABEL)
axA.set_ylabel("Resolved fraction", color=BLUE, fontsize=FS_LABEL)
axA.tick_params(axis="y", labelcolor=BLUE, labelsize=FS_TICK)
sparse_x(axA, XTICKS_A)
axA.grid(alpha=0.25)
axA2 = axA.twinx()
axA2.plot(mmin_x, mmin_sh, "-s", color=ORANGE, lw=2.6, ms=8)
if LOG_SHOTS:
    axA2.set_yscale("log")
    _t = SHOTS_YTICKS
    axA2.set_ylim(_t[0] * 0.97, max(mmin_sh) * 1.08)
else:
    _t = SHOTS_YTICKS_LIN
    axA2.set_ylim(min(min(mmin_sh), _t[0]) * 0.94, max(max(mmin_sh), _t[-1]) * 1.04)
axA2.yaxis.set_major_locator(FixedLocator(_t))
axA2.yaxis.set_minor_locator(NullLocator())
axA2.yaxis.set_major_formatter(FixedFormatter([str(v) for v in _t]))
axA2.set_ylabel("Shots per sample", color=ORANGE, fontsize=FS_LABEL)
axA2.tick_params(axis="y", which="both", labelcolor=ORANGE, labelsize=FS_TICK)


axB.plot(mmax_x, mmax_rf, "-o", color=BLUE, lw=2.6, ms=8)
axB.set_xlabel(r"Upper bound  $M_{\max}$", fontsize=FS_LABEL)
sparse_x(axB, XTICKS_B)
axB.grid(alpha=0.25)
axB2 = axB.twinx()
axB2.plot(mmax_x, mmax_cap, "-s", color=ORANGE, lw=2.6, ms=8)
axB2.set_ylabel("Saturated fraction", color=ORANGE, fontsize=FS_LABEL)
axB2.tick_params(axis="y", which="both", labelcolor=ORANGE, labelsize=FS_TICK)


try:
    fig.get_layout_engine().set(wspace=WSPACE)
except Exception:
    pass

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, **SAVE); plt.close(fig)
print("saved", OUT)
