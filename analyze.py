"""Aggregates results into numbers.json and grayscale figures."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
R = json.load(open("results/results.json"))
N = {"n_rep": len(R)}
def ms(v): v = np.array([x for x in v if x is not None], float); return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
REG = ["nominal", "heatwave", "cold_spell"]; FC = ["pcf", "gbt", "mlp", "clim"]
# ---------- forecast skill ----------
N["skill"] = {}
for reg in REG:
    N["skill"][reg] = {}
    for f in FC:
        rows = [s[f] for r in R for s in r["skill"][reg]]
        N["skill"][reg][f] = {k: ms([x[k] for x in rows]) for k in ["brier", "logloss", "mae", "mae_d1", "mae_d2", "mae_oor"]}
        N["skill"][reg][f]["n_oor"] = int(np.sum([x["n_oor"] for x in rows]))
N["skill_no_sensor"] = {k: ms([r["skill_no_sensor"]["pcf"][k] for r in R]) for k in ["brier", "mae"]}
N["skill_with_sensor_nom0"] = {k: ms([r["skill"]["nominal"][0]["pcf"][k] for r in R]) for k in ["brier", "mae"]}
th = np.array([r["theta"] for r in R]); N["theta"] = [[float(a), float(b)] for a, b in zip(th.mean(0), th.std(0, ddof=1))]
# ---------- policies ----------
POL = ["calendar", "greedy", "milp_mean", "milp_scen", "milp_oracle"]
S = [s for r in R for s in r["seasons"]]
def get(pol, reg=None, key="fruit_set"):
    return np.array([s[key] for s in S if s["policy"] == pol and (reg is None or s["regime"] == reg)], float)
N["pol"] = {}
for reg in REG + [None]:
    kk = reg or "all"; N["pol"][kk] = {}
    for p in POL:
        N["pol"][kk][p] = {k: ms(get(p, reg, k)) for k in ["fruit_set", "pollen_per_fruit", "sorties", "viol_rate", "solve_s", "nat_fert", "art_fert"]}
        N["pol"][kk][p]["gap"] = ms((get("milp_oracle", reg) - get(p, reg)) / get("milp_oracle", reg))
# ---------- ablation ----------
A = [a for r in R for a in r["ablation"]]
N["abl"] = {}
for reg in REG + [None]:
    kk = reg or "all"; N["abl"][kk] = {}
    for tag in sorted(set(a["tag"] for a in A)):
        rows = [a for a in A if a["tag"] == tag and (reg is None or a["regime"] == reg)]
        N["abl"][kk][tag] = {k: ms([a[k] for a in rows]) for k in ["fruit_set", "pollen_per_fruit", "viol_rate", "sorties"]}
# ---------- statistics ----------
def holm(p):
    p = np.array(p); o = np.argsort(p); adj = np.empty_like(p); run = 0; m = len(p)
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj
def rbc(d):
    nz = d[d != 0]
    if not len(nz): return 0.0
    rk = np.argsort(np.argsort(np.abs(nz))) + 1; return float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
stats = []
for key, lab in [("fruit_set", "fruit set"), ("viol_rate", "violation rate"), ("pollen_per_fruit", "pollen per fruit")]:
    a = get("milp_scen", None, key)
    for b in ["calendar", "greedy", "milp_mean"]:
        bb = get(b, None, key); d = a - bb
        p = wilcoxon(a, bb).pvalue if np.any(d != 0) else 1.0
        stats.append(dict(metric=lab, base=b, mean_diff=float(d.mean()), p=float(p), rbc=rbc(d), n=int(len(d))))
# forecaster comparison on per-season MAE (all regimes) and out-of-range MAE
for f in ["gbt", "mlp"]:
    a = np.array([s["pcf"]["mae"] for r in R for reg in REG for s in r["skill"][reg]]); b = np.array([s[f]["mae"] for r in R for reg in REG for s in r["skill"][reg]])
    stats.append(dict(metric="forecast MAE", base=f, mean_diff=float((a - b).mean()), p=float(wilcoxon(a, b).pvalue), rbc=rbc(a - b), n=int(len(a))))
    a = np.array([s["pcf"]["mae_oor"] for r in R for reg in ["heatwave", "cold_spell"] for s in r["skill"][reg] if s["pcf"]["mae_oor"] is not None])
    b = np.array([s[f]["mae_oor"] for r in R for reg in ["heatwave", "cold_spell"] for s in r["skill"][reg] if s[f]["mae_oor"] is not None])
    stats.append(dict(metric="out-of-range MAE", base=f, mean_diff=float((a - b).mean()), p=float(wilcoxon(a, b).pvalue), rbc=rbc(a - b), n=int(len(a))))
adj = holm([s["p"] for s in stats])
for s, x in zip(stats, adj): s["p_holm"] = float(x)
N["stats"] = stats
# ---------- frontier ----------
if os.path.exists("results/frontier.json"):
    FR = json.load(open("results/frontier.json"))
    N["frontier"] = {}
    for fam in ["mean+margin", "scenario", "hybrid"]:
        lv = sorted(set(x["level"] for x in FR if x["family"] == fam))
        N["frontier"][fam] = [dict(level=l, fruit_set=ms([x["fruit_set"] for x in FR if x["family"] == fam and x["level"] == l]),
                                   viol=ms([x["viol_rate"] for x in FR if x["family"] == fam and x["level"] == l]),
                                   pollen=ms([x["pollen_per_fruit"] for x in FR if x["family"] == fam and x["level"] == l]),
                                   n=int(sum(1 for x in FR if x["family"] == fam and x["level"] == l))) for l in lv]
json.dump(N, open("numbers.json", "w"), indent=1)

# =================== figures ===================
def save(fig, name): fig.savefig(f"figs/{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
LAB = {"pcf": "PCF (proposed)", "gbt": "Gradient-boosted trees", "mlp": "MLP", "clim": "Climatology"}
PL = {"calendar": "Calendar", "greedy": "Threshold", "milp_mean": "Mean-forecast MILP", "milp_scen": "Scenario CC-MILP", "milp_oracle": "Oracle MILP"}

# Fig. 2: example heatwave season trace (block 0)
tr = R[0]["trace"]; h = np.arange(len(tr["T"])) / 24
fig, axs = plt.subplots(3, 1, figsize=(7.16, 3.2), sharex=True, gridspec_kw=dict(hspace=0.15))
axs[0].plot(h, tr["T"], color="black", lw=0.6); axs[0].set_ylabel("$T$ (°C)", fontsize=7)
axs[0].axhline(R[0]["curves"]["train_T"][1], color="0.4", ls=":", lw=0.7); axs[0].text(0.1, R[0]["curves"]["train_T"][1] + 0.6, "99th percentile of trial temperatures", fontsize=6.2, color="0.3")
axs[1].plot(h, tr["wind"], color="black", lw=0.6); axs[1].axhline(4.5, color="0.4", ls=":", lw=0.7); axs[1].set_ylabel("$w$ (m s$^{-1}$)", fontsize=7)
axs[1].text(0.1, 4.7, "$w_{\\max}$", fontsize=6.5, color="0.3")
axs[2].fill_between(h, 0, tr["succ"], color="0.75", lw=0); axs[2].plot(h, tr["succ"], color="black", lw=0.5); axs[2].set_ylabel("true $v$", fontsize=7)
axs[2].set_xlabel("Day of bloom season", fontsize=7); axs[2].set_ylim(0, 1)
for ax in axs: ax.tick_params(labelsize=6.5)
save(fig, "fig2_trace")

# Fig. 3: temperature response curves
c = R[0]["curves"]; T = np.array(c["T"])
fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.axvspan(c["train_T"][0], c["train_T"][1], color="0.92", lw=0)
ax.text((c["train_T"][0] + c["train_T"][1]) / 2, 0.97, "range of trial temperatures", ha="center", fontsize=6.2, color="0.3")
ax.plot(T, c["truth"], color="black", lw=1.6, label="Simulator truth (unknown to models)")
ax.plot(T, c["pcf"], color="black", lw=0.9, ls="--", label=LAB["pcf"])
ax.plot(T, c["gbt"], color="0.45", lw=0.9, ls="-.", label=LAB["gbt"])
ax.plot(T, c["mlp"], color="0.6", lw=0.9, ls=":", label=LAB["mlp"])
ax.set_xlabel("Air temperature (°C), RH = 60%, $w$ = 1.5 m s$^{-1}$", fontsize=7); ax.set_ylabel("Success probability $v$", fontsize=7); ax.set_ylim(0, 1.05)
ax.legend(fontsize=5.9, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
save(fig, "fig3_response")

# Fig. 4: reliability diagrams pooled
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.85), gridspec_kw=dict(wspace=0.35))
for ax, f in zip(axs, FC):
    pts = np.array([b for r in R for reg in REG for s in r["skill"][reg] for b in s[f]["calib"]])
    ed = np.linspace(0, 1, 11); xs, ys, ws = [], [], []
    for i in range(10):
        m = (pts[:, 0] >= ed[i]) & (pts[:, 0] < ed[i + 1] + (i == 9))
        if m.any(): w = pts[m, 2]; xs.append(np.average(pts[m, 0], weights=w)); ys.append(np.average(pts[m, 1], weights=w)); ws.append(w.sum())
    ax.plot([0, 1], [0, 1], ls=":", color="0.4", lw=0.7); ax.plot(xs, ys, color="black", marker="o", ms=2.5, lw=0.9)
    ws = np.array(ws); ax.bar(xs, ws / ws.sum() * 0.6, width=0.07, color="0.82", edgecolor="0.5", lw=0.3)
    ax.set_title(LAB[f], fontsize=7); ax.set_xlabel("Predicted $v$", fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0].set_ylabel("Observed germination", fontsize=7)
save(fig, "fig4_calibration")

# Fig. 5: forecast error by regime incl. out-of-range
fig, axs = plt.subplots(1, 2, figsize=(7.16, 1.9), gridspec_kw=dict(wspace=0.22))
x = np.arange(3); wb = 0.2; cols = ["0.15", "0.55", "0.8", "white"]
for i, f in enumerate(FC):
    m = [N["skill"][reg][f]["mae"][0] for reg in REG]; e = [N["skill"][reg][f]["mae"][1] for reg in REG]
    axs[0].bar(x + (i - 1.5) * wb, m, wb, yerr=e, color=cols[i], edgecolor="black", lw=0.4, capsize=1.2, error_kw=dict(lw=0.5), label=LAB[f])
    m2 = [N["skill"][reg][f]["mae_oor"][0] for reg in ["heatwave", "cold_spell"]]; e2 = [N["skill"][reg][f]["mae_oor"][1] for reg in ["heatwave", "cold_spell"]]
    axs[1].bar(np.arange(2) + (i - 1.5) * wb, m2, wb, yerr=e2, color=cols[i], edgecolor="black", lw=0.4, capsize=1.2, error_kw=dict(lw=0.5))
axs[0].set_xticks(x); axs[0].set_xticklabels(["Nominal", "Heatwave", "Cold spell"]); axs[0].set_ylabel("MAE of $v$ (all hours)", fontsize=7)
axs[1].set_xticks(range(2)); axs[1].set_xticklabels(["Heatwave", "Cold spell"]); axs[1].set_ylabel("MAE of $v$ (out-of-range hours)", fontsize=7)
axs[0].legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(1.05, 1.22))
save(fig, "fig5_forecast_error")

# Fig. 6: fruit set and violation by policy and regime
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw=dict(wspace=0.25))
cols = ["white", "0.85", "0.6", "0.15", "0.4"]; hat = ["", "////", "", "", "xxxx"]; wb = 0.16
for i, p in enumerate(POL):
    m = [N["pol"][reg][p]["fruit_set"][0] * 100 for reg in REG]; e = [N["pol"][reg][p]["fruit_set"][1] * 100 for reg in REG]
    axs[0].bar(x + (i - 2) * wb, m, wb, yerr=e, color=cols[i], hatch=hat[i], edgecolor="black", lw=0.4, capsize=1.2, error_kw=dict(lw=0.5), label=PL[p])
    m = [N["pol"][reg][p]["viol_rate"][0] * 100 for reg in REG]; e = [N["pol"][reg][p]["viol_rate"][1] * 100 for reg in REG]
    axs[1].bar(x + (i - 2) * wb, m, wb, yerr=e, color=cols[i], hatch=hat[i], edgecolor="black", lw=0.4, capsize=1.2, error_kw=dict(lw=0.5))
for ax in axs: ax.set_xticks(x); ax.set_xticklabels(["Nominal", "Heatwave", "Cold spell"])
axs[1].set_ylim(bottom=0); axs[0].set_ylabel("Fruit set (%)", fontsize=7); axs[1].set_ylabel("Sorties above $w_{\\max}$ (%)", fontsize=7)
axs[0].set_ylim(30, 85)
axs[0].legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(1.1, 1.22))
save(fig, "fig6_policies")

# Fig. 7: risk-return frontier
if "frontier" in N:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    OFF = {("mean+margin", 0.0): (0, 9), ("mean+margin", 0.5): (12, 10), ("mean+margin", 1.0): (-14, -12), ("mean+margin", 1.5): (12, 0),
           ("scenario", 0.05): (12, 0), ("scenario", 0.1): (16, -16), ("scenario", 0.2): (0, 11), ("scenario", 0.35): (0, -10), ("scenario", 0.5): (0, -10),
           ("hybrid", 0.05): (-16, 8), ("hybrid", 0.1): (16, -10), ("hybrid", 0.2): (-14, 10), ("hybrid", 0.35): (10, -12), ("hybrid", 0.5): (-4, -11)}
    sty = {"mean+margin": ("0.5", "--", "s", "Mean MILP, wind margin $\\delta$"), "scenario": ("black", "-", "o", "Scenario CC-MILP, risk $\\varepsilon$"), "hybrid": ("0.3", ":", "^", "Mean objective + CC, $\\varepsilon$")}
    for fam, (col, ls, mk, lab) in sty.items():
        pts = N["frontier"][fam]
        xs = [p["viol"][0] * 100 for p in pts]; ys = [p["fruit_set"][0] * 100 for p in pts]
        ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.2, lw=0.9, mfc="white" if fam != "scenario" else "black", label=lab)
        for p, x_, y_ in zip(pts, xs, ys):
            off = OFF.get((fam, round(p["level"], 2)), (4, -8))
            ax.annotate(f"{p['level']:g}", (x_, y_), textcoords="offset points", xytext=off, fontsize=5.4, color="0.15",
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.35, color="0.45", shrinkA=0, shrinkB=2.2))
    ax.set_xlabel("Sorties flown above $w_{\\max}$ (%)", fontsize=7); ax.set_ylabel("Fruit set (%)", fontsize=7)
    ax.legend(fontsize=5.9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    save(fig, "fig7_frontier")
print("done", len(R))

# ---------- wind-exceedance reliability, sortie allocation and pollen budget ----------
if os.path.exists("results/windcal.json"):
    WC = json.load(open("results/windcal.json")); N["windcal"] = WC
    EX = json.load(open("results/extra.json")) if os.path.exists("results/extra.json") else None
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw=dict(wspace=0.42))
    rel = np.array(WC["rel"])
    axs[0].plot([0, 1], [0, 1], ls=":", color="0.4", lw=0.7); axs[0].plot(rel[:, 0], rel[:, 1], color="black", marker="o", ms=2.6, lw=0.9)
    axs[0].set_xlabel("Ensemble exceedance $\\hat{P}_{b,h}$", fontsize=7); axs[0].set_ylabel("Observed exceedance", fontsize=7); axs[0].set_title("(a) reliability of $\\hat{P}$", fontsize=7.2)
    if EX:
        PL2 = {"calendar": "Calendar", "greedy": "Threshold", "milp_mean": "Mean MILP", "milp_scen": "Scenario CC-MILP"}
        sty = {"calendar": ("0.65", ":"), "greedy": ("0.45", "-."), "milp_mean": ("0.3", "--"), "milp_scen": ("black", "-")}
        N["alloc"] = {}
        for p in PL2:
            L = [row for e in EX["log"] if e["policy"] == p for row in e["log"]]
            L = np.array(L) if L else np.zeros((0, 5))
            hist = np.zeros(24)
            for hh, s_, T_, w_, n_ in L: hist[int(hh)] += n_
            hist = hist / max(1, hist.sum())
            axs[1].plot(range(6, 20), hist[6:20] * 100, color=sty[p][0], ls=sty[p][1], lw=1.0, label=PL2[p])
            ws = L[:, 4] if len(L) else np.ones(1)
            N["alloc"][p] = dict(mean_success=float(np.average(L[:, 1], weights=ws)) if len(L) else None, mean_T=float(np.average(L[:, 2], weights=ws)) if len(L) else None,
                                 share_low=float(np.sum(ws[L[:, 1] < 0.3]) / ws.sum()) if len(L) else None)
        axs[1].set_xlabel("Hour of day", fontsize=7); axs[1].set_ylabel("Share of sorties (%)", fontsize=7); axs[1].set_title("(b) when sorties are flown", fontsize=7.2)
        axs[1].legend(fontsize=5.6, loc="upper right")
        for i, p in enumerate(PL2):
            L = np.array([row for e in EX["log"] if e["policy"] == p for row in e["log"]])
            vals = np.repeat(L[:, 1], L[:, 4].astype(int))
            bp = axs[2].boxplot([vals], positions=[i], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.9))
            for b_ in bp["boxes"]: b_.set_facecolor(["white", "0.85", "0.6", "0.3"][i]); b_.set_linewidth(0.6)
        axs[2].set_xticks(range(4)); axs[2].set_xticklabels(["Cal.", "Thr.", "Mean", "CC"], fontsize=6.5)
        axs[2].set_ylabel("True $v$ at sortie time", fontsize=7); axs[2].set_title("(c) realised success of sorties", fontsize=7.2)
        B = EX["budget"]; N["budget"] = {}
        for pd in [25.0, 60.0]:
            for p in ["greedy", "milp_scen"]:
                rows = [b_ for b_ in B if b_["pollen_day"] == pd and b_["policy"] == p]
                if rows: N["budget"][f"{p}|{pd:g}"] = {k: ms([r[k] for r in rows]) for k in ["fruit_set", "pollen_per_fruit", "sorties", "viol_rate"]}
        for p in ["greedy", "milp_scen"]:
            rows = [s for s in S if s["policy"] == p and s in [x for r in R[:1] for x in r["seasons"]]]
            N["budget"][f"{p}|40"] = {k: ms([r[k] for r in rows]) for k in ["fruit_set", "pollen_per_fruit", "sorties", "viol_rate"]}
    save(fig, "fig8_allocation")
    json.dump(N, open("numbers.json", "w"), indent=1)
    print("allocation figure done")

# ---------- Brier decomposition (Murphy) from binned forecasts ----------
N["murphy"] = {}
for reg in REG:
    N["murphy"][reg] = {}
    for fc in FC:
        pts = np.array([b for r in R for s in r["skill"][reg] for b in s[fc]["calib"]])
        n = pts[:, 2]; o_bar = np.average(pts[:, 1], weights=n)
        rel_ = float(np.sum(n * (pts[:, 0] - pts[:, 1]) ** 2) / n.sum()); res_ = float(np.sum(n * (pts[:, 1] - o_bar) ** 2) / n.sum())
        N["murphy"][reg][fc] = dict(rel=rel_, res=res_, unc=float(o_bar * (1 - o_bar)))
# ---------- response factors: fitted vs simulator ----------
th = np.array(N["theta"])[:, 0]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.85), gridspec_kw=dict(wspace=0.38))
D = np.linspace(0, 4, 200)
for T_, ls in [(20, "-"), (30, "--")]:
    axs[0].plot(D, np.exp(-np.maximum(0, D - 0.9) * (0.55 + 0.03 * max(T_ - 20, 0))), color="0.55", ls=ls, lw=1.0, label=f"simulator, {T_} °C")
axs[0].plot(D, np.exp(-th[3] * np.maximum(0, D - th[4])), color="black", lw=1.2, label="PCF fit")
axs[0].set_xlabel("Vapour-pressure deficit $D$ (kPa)", fontsize=7); axs[0].set_ylabel("$g_V$", fontsize=7); axs[0].legend(fontsize=5.6, loc="lower left")
w = np.linspace(0, 8, 200)
axs[1].plot(w, 1 / (1 + np.exp((w - 4.2) / 0.55)), color="0.55", lw=1.0, label="simulator")
axs[1].plot(w, 1 / (1 + np.exp((w - th[5]) / th[6])), color="black", lw=1.2, ls="--", label="PCF fit")
axs[1].axvline(4.5, color="0.4", ls=":", lw=0.7); axs[1].text(4.6, 0.85, "$w_{\\max}$", fontsize=6.3)
axs[1].set_xlabel("Wind speed $w$ (m s$^{-1}$)", fontsize=7); axs[1].set_ylabel("$g_W$", fontsize=7); axs[1].legend(fontsize=5.6, loc="lower left")
# per-season paired comparison
for k, (b_, mk) in enumerate([("calendar", "o"), ("greedy", "s"), ("milp_mean", "^")]):
    a_ = get("milp_scen", None, "fruit_set") * 100; bb = get(b_, None, "fruit_set") * 100
    axs[2].scatter(bb, a_, s=9, marker=mk, facecolor="white" if k == 0 else ("0.6" if k == 1 else "black"), edgecolor="black", lw=0.5, label={"calendar": "vs calendar", "greedy": "vs threshold", "milp_mean": "vs mean MILP"}[b_])
lo_, hi_ = 30, 85
axs[2].plot([lo_, hi_], [lo_, hi_], color="0.4", ls=":", lw=0.7); axs[2].set_xlim(lo_, hi_); axs[2].set_ylim(lo_, hi_)
axs[2].set_xlabel("Comparator fruit set (%)", fontsize=7); axs[2].set_ylabel("Scenario CC-MILP fruit set (%)", fontsize=7); axs[2].legend(fontsize=5.6, loc="lower right")
axs[0].set_title("(a) desiccation factor", fontsize=7.2); axs[1].set_title("(b) deposition factor", fontsize=7.2); axs[2].set_title("(c) season-level pairs", fontsize=7.2)
save(fig, "fig9_factors_pairs")
json.dump(N, open("numbers.json", "w"), indent=1)
print("murphy and factors done")
