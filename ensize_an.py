# %% [markdown]
# ## 11. Ensemble-size analysis and figure

# %%
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
E = json.load(open("results/ensize.json"))
Ms = sorted(set(e["M"] for e in E))
def ms(v): v = np.array(v, float); return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
out = {}
for M in Ms:
    rows = [e for e in E if e["M"] == M]
    out[str(M)] = {k: ms([r[k] for r in rows]) for k in ["fruit_set", "viol_rate", "solve_s", "sorties", "pollen_per_fruit"]}
    out[str(M)]["n"] = len(rows); out[str(M)]["kmax"] = int(np.floor(0.10 * M + 1e-9))
# paired differences against M=16 for each season
base = {(e["rep"], e["season"]): e for e in E if e["M"] == 16}
for M in Ms:
    d = [e["fruit_set"] - base[(e["rep"], e["season"])]["fruit_set"] for e in E if e["M"] == M and (e["rep"], e["season"]) in base]
    out[str(M)]["d_fs_vs16"] = ms(d)
N = json.load(open("numbers.json")); N["ensize"] = out; json.dump(N, open("numbers.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw=dict(wspace=0.45))
x = np.arange(len(Ms)); se = lambda key, M: out[str(M)][key][1] / np.sqrt(out[str(M)]["n"])
def bar(a, key, scale, ylab, fmt):
    m = np.array([out[str(M)][key][0] for M in Ms]) * scale; s = np.array([se(key, M) for M in Ms]) * scale
    a.errorbar(x, m, yerr=s, fmt=fmt, color="k", ms=4, lw=0.8, capsize=2.5, mfc="white", zorder=3)
    a.plot(x, m, color="k", lw=0.6, ls="-")
    a.set_xticks(x); a.set_xticklabels([str(M) for M in Ms]); a.set_xlabel("Ensemble members $M$"); a.set_ylabel(ylab)
    return m, s
m, s = bar(ax[0], "viol_rate", 100, "Sorties above $w_{\\max}$ (%)", "o")
for i, M in enumerate(Ms):
    ax[0].text(x[i] + 0.12, m[i] + 0.05, f"$k$={out[str(M)]['kmax']}", ha="left", va="bottom", fontsize=6.3)
ax[0].set_ylim(0, max(m + s) * 1.15); ax[0].set_xlim(-0.4, len(Ms) - 0.4)
bar(ax[1], "d_fs_vs16", 100, "Fruit set relative to $M$=16 (points)", "s")
ax[1].axhline(0, color="k", lw=0.5, ls=":")
bar(ax[2], "solve_s", 1, "MILP solve time per plan (s)", "^")
for a, t in zip(ax, "abc"):
    a.text(-0.30, 1.04, f"({t})", transform=a.transAxes, fontsize=8, fontweight="bold")
fig.savefig("figs/fig11_ensemble.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
print("ensize fig ok")
