#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot resolved fraction versus total shots for adaptive and uniform sampling.

The script reads existing CSV results and redraws the comparison without
training or inference.

text:  scripts/results/final_gmm_diagW_pi/diagW/uniform_diagW.csv     (text uniform text: total_shots text rf)
     scripts/results/final_gmm_diagW_pi/diagW/infer/T20/metrics.csv (text adaptive text: 303 shots / rf 0.938, seed+202)
text:  results/figures/fig_gmm_adaptive_vs_uniform_rf.png

The styling matches the corresponding panel produced by
``run_T1000_mainline.py``.
"""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter, LogLocator

sys.stdout.reconfigure(encoding="utf-8")

# Increase font sizes for consistency with the other figures in this section.
plt.rcParams.update({
    "font.size": 17, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
})

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
RES = SCRIPTS_DIR / "results" / "final_gmm_diagW_pi" / "diagW"
UNI = RES / "uniform_diagW.csv"
STAR = RES / "infer/T20/metrics.csv"
OUT = PROJECT_ROOT / "results" / "figures" / "fig_gmm_adaptive_vs_uniform_rf.png"

BLUE = "#2970b0"
GREY = "#555555"
SAVE = dict(dpi=150, bbox_inches="tight")


uni_shots, uni_rf = [], []
with open(UNI, encoding="utf-8") as f:
    rows = list(csv.reader(f))

for row in rows[1:]:
    if not row or not row[0].strip().isdigit():
        continue
    uni_shots.append(float(row[2]))
    uni_rf.append(float(row[3]))


with open(STAR, encoding="utf-8") as f:
    rows = list(csv.reader(f))

data = rows[1]
ad_rf = float(data[1])
ad_shots = float(data[3])

print(f"[uniform ] {len(uni_shots)} text: shots {uni_shots[0]:.0f}..{uni_shots[-1]:.0f}  rf {min(uni_rf):.3f}..{max(uni_rf):.3f}")
print(f"[adaptive] star shots={ad_shots:.0f}  rf={ad_rf:.4f}")


fig, ax = plt.subplots(figsize=(7.6, 5.2), layout="constrained")
ax.plot(uni_shots, uni_rf, "-o", color=GREY, lw=2.2, ms=8, label="Uniform shots")
ax.plot([ad_shots], [ad_rf], "*", color=BLUE, ms=26, label="Adaptive (ours)", zorder=6)
ax.axhline(ad_rf, color=BLUE, ls="--", lw=1.3, alpha=0.6)
ax.set_xscale("log")

ax.xaxis.set_major_locator(FixedLocator([300, 600, 1000, 2000, 3000]))
ax.xaxis.set_major_formatter(FixedFormatter(
    [r"$3\times10^{2}$", r"$6\times10^{2}$", r"$10^{3}$", r"$2\times10^{3}$", r"$3\times10^{3}$"]))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Total shots")
ax.set_ylabel("Resolved fraction  rf")
ax.grid(alpha=0.3)

ax.legend(loc="lower right", labelspacing=0.5, handleheight=1.6,
          handletextpad=0.7, borderpad=0.7, markerscale=0.75)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, **SAVE)
plt.close(fig)
print(f"[done] -> {OUT}")
