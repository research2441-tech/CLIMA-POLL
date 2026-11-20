import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 6.5})
fig, ax = plt.subplots(figsize=(7.16, 3.6))
ax.set_xlim(0, 104); ax.set_ylim(-1, 53); ax.axis("off")
BOX = {}
g = "#E6E6E6"
def box(key, x, y, w, h, title, sub=None, fill="white", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fill, ec="black", lw=lw))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=6.6, fontweight="bold", linespacing=1.1)
    if sub:
        nl = title.count("\n")
        ax.text(x + w / 2, y + h - 5.0 - 2.6 * nl, sub, ha="center", va="top", fontsize=6.0, linespacing=1.3)
    BOX[key] = (x, y, w, h)
def arrow(p0, p1, ls="-", lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, ls=ls, color="black", shrinkA=0, shrinkB=0))
def R(k): x, y, w, h = BOX[k]; return (x + w, y + h / 2)
def L(k): x, y, w, h = BOX[k]; return (x, y + h / 2)
def T(k, f=0.5): x, y, w, h = BOX[k]; return (x + w * f, y + h)
def B(k, f=0.5): x, y, w, h = BOX[k]; return (x + w * f, y)

# inputs column
box("ens", 0.5, 36, 17, 12, "Station ensemble\nforecast", "$T^{(m)}_h$, $RH^{(m)}_h$, $w^{(m)}_h$,\nrain, radiation; 48 h")
box("sens", 0.5, 20, 17, 12, "In-orchard sensors", "block history of\n$T$, $RH$, $w$ (72 h)")
box("phen", 0.5, 4, 17, 12, "Phenology state", "thermal time $\\Theta_b$,\nreceptive pool $R_{b,d}$")
# processing
box("down", 22.5, 28, 17, 12, "Microclimate\ndownscaling", "offsets $\\Delta\\hat{T}_b$, $\\Delta\\widehat{RH}_b$,\nexposure $\\hat{e}_b$")
box("pcf", 44.5, 28, 18, 12, "Physiologically\nconstrained forecaster", "$v=\\pi_0\\, g_T\\, g_V\\, g_W\\,(1-r)$", fill=g)
box("trial", 43.5, 44.0, 20, 7.0, "Viability trials", "binomial germination counts")
box("scen", 67, 28, 15.5, 12, "Scenario tensor", "success $v_{b,h}^{(m)}$\n\nwind risk $\\hat{P}_{b,h}$")
box("milp", 67, 4.5, 15.5, 17.5, "Chance-constrained\nMILP", "max scenario-mean\nfertilised flowers s.t.\nfleet, battery, pollen,\n$\\hat{P}_{b,h}\\leq\\varepsilon$", fill=g)
box("exec", 87.5, 4.5, 15.5, 17.5, "Execute day 1", "sorties $x_{b,h}$ flown;\nnew observations,\nupdated pools")

arrow(R("ens"), (22.5, 37.0))
arrow(R("sens"), (22.5, 31.0))
arrow(R("down"), L("pcf"))
arrow(R("pcf"), L("scen"))
arrow(B("trial"), T("pcf"), ls="--")
ax.text(B("trial")[0] - 0.7, 42.1, "maximum-likelihood fit", fontsize=5.6, style="italic", ha="right", va="center")
arrow(B("scen"), T("milp"))
ax.text(B("scen")[0] + 0.7, 24.5, "$M$ scenarios", fontsize=5.8, style="italic", ha="left", va="center")
arrow(R("phen"), (67, 10.0))
ax.text(42.0, 10.8, "receptive pools $R_{b,d}$", fontsize=5.8, style="italic", ha="center")
arrow(R("milp"), L("exec"))
# rolling-horizon feedback: exec -> phenology and sensors (loop below boxes)
ax.plot([95.25, 95.25, 9.0], [4.5, 1.2, 1.2], color="black", lw=0.8)
arrow((9.0, 1.2), (9.0, 3.8))
ax.text(52, 1.7, "rolling horizon: next morning, re-plan with new forecast, sensor data and pools", fontsize=5.8, style="italic", ha="center", va="bottom")
# legend
ax.plot([71, 74], [49.5, 49.5], color="black", lw=0.8); ax.text(74.6, 49.5, "decision flow", va="center", fontsize=5.8)
ax.plot([71, 74], [46.8, 46.8], color="black", lw=0.8, ls="--"); ax.text(74.6, 46.8, "offline model fitting", va="center", fontsize=5.8)
ax.add_patch(FancyBboxPatch((71.2, 43.3), 2.6, 1.6, boxstyle="round,pad=0.1", fc=g, ec="black", lw=0.5)); ax.text(74.6, 44.1, "core contribution", va="center", fontsize=5.8)
fig.savefig("figs/fig1_architecture.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print("ok")
