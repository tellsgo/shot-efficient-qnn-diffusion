# -*- coding: utf-8 -*-
# ============================================================================



#



#

#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate quddpm_gpu
#   python scripts/plotting/make_fig_eta_merged.py
#






# ============================================================================
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


FS_LABEL, FS_TICK, FS_LEG = 18, 15, 14
LEG_LOC   = "upper right"
RF_YLIM   = (0.80, 1.0)
ANCHOR_ETA = 0.2
DROP_ETA0  = False
FIGSIZE    = (6.6, 4.7)
# -------------------------

BLUE, ORANGE, GREY = "#2970b0", "#c85a11", "#555555"
SAVE = dict(dpi=150, bbox_inches="tight")

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CSV = SCRIPTS_DIR / "results" / "final_gmm_diagW_pi" / "diagW" / "eta_diagW.csv"
OUT = PROJECT_ROOT / "results" / "figures" / "fig_gmm_eta.png"

eta, rf, shots = [], [], []
with open(CSV) as f:
    for row in csv.DictReader(f):
        e = float(row["eta"])
        if DROP_ETA0 and e == 0.0:
            continue
        eta.append(e); rf.append(float(row["rf"])); shots.append(float(row["total_shots"]))

fig, axL = plt.subplots(figsize=FIGSIZE, layout="constrained")


l1, = axL.plot(eta, rf, "-o", color=BLUE, lw=2.6, ms=8, label="Resolved fraction")
axL.set_xlabel(r"DDIM stochasticity  $\eta$", fontsize=FS_LABEL)
axL.set_ylabel("Resolved fraction  rf", color=BLUE, fontsize=FS_LABEL)
axL.set_ylim(*RF_YLIM)
axL.tick_params(axis="y", labelcolor=BLUE, labelsize=FS_TICK)
axL.tick_params(axis="x", labelsize=FS_TICK)
axL.grid(alpha=0.25)


axR = axL.twinx()
l2, = axR.plot(eta, shots, "-s", color=ORANGE, lw=2.6, ms=8, label="Shots per sample")
axR.set_yscale("log")
axR.set_ylabel("Shots per sample", color=ORANGE, fontsize=FS_LABEL)
axR.tick_params(axis="y", labelcolor=ORANGE, labelsize=FS_TICK)


shots_at_anchor = dict(zip(eta, shots))[ANCHOR_ETA]
C = shots_at_anchor * ANCHOR_ETA ** 2
xr = [e for e in eta if e > 0]
l3, = axR.plot(xr, [C / (e * e) for e in xr], "--", color=GREY, lw=2.2, label=r"$\propto 1/\eta^{2}$")

axL.legend(handles=[l1, l2, l3], loc=LEG_LOC, fontsize=FS_LEG, framealpha=0.9)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, **SAVE); plt.close(fig)
print("saved", OUT)
