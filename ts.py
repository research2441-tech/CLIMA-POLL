# %% [markdown]
# ## 12. Example forecast trajectories during a heatwave (Fig. 7)

# %%
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
import climapoll as C
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(500)
train = [C.orchard(np.random.default_rng(i), "nominal") for i in range(8)]
T, RH, W, S, Rn, hod, K, Nn = C.trial_data(train, rng); X = C.feats(T, RH, W, S, Rn, hod)
models = dict(pcf=C.PCF().fit(T, RH, W, Rn, K, Nn), gbt=C.GBT().fit(X, K, Nn))
hi = np.percentile(T, 99.5)
O = C.orchard(np.random.default_rng(102), "heatwave"); rng = np.random.default_rng(11)
ev = np.where(O["st"]["event"] > 3)[0]; d0 = max(0, int(ev[0]) - 1) if len(ev) else 4
tt, tru, pc_, gb_, Tb = [], [], [], [], []
for day in range(d0, min(d0 + 5, C.DAYS - 1)):
    F = C.ensemble_forecast(O, day, rng); h0, Hh = F["h0"], F["H"]; dl = C.daylight_idx(h0, Hh); dl = dl[dl < 24]; hours = h0 + dl
    oT, ex, oR = C.sensor_offsets(O, day, rng); b = 0
    Tm = F["T"][:, dl] + oT[b]; Wm = F["wind"][:, dl] * ex[b]; Rm = np.clip(F["RH"][:, dl] + oR[b], 8, 100); Sm = F["solar"][:, dl]; Rr = F["rain"][:, dl]
    for name, L in [("pcf", pc_), ("gbt", gb_)]:
        L.append(models[name].predict(Tm, Rm, Wm, Sm, Rr, np.broadcast_to(hours % 24, Tm.shape)).mean(0))
    tt.append(hours / 24); tru.append(O["blk"]["succ"][b, hours]); Tb.append(O["blk"]["T"][b, hours])
tt, tru, pc_, gb_, Tb = map(np.concatenate, (tt, tru, pc_, gb_, Tb))
fig, ax = plt.subplots(figsize=(7.16, 1.9))
oor = Tb > hi
for i in range(len(tt)):
    if oor[i]: ax.axvspan(tt[i] - 0.5 / 24, tt[i] + 0.5 / 24, color="0.88", lw=0)
for d in np.unique(np.floor(tt)):
    m = np.floor(tt) == d
    ax.plot(tt[m], tru[m], color="black", lw=1.4, label="truth" if d == np.floor(tt[0]) else None)
    ax.plot(tt[m], pc_[m], color="black", lw=0.9, ls="--", label="PCF day-ahead" if d == np.floor(tt[0]) else None)
    ax.plot(tt[m], gb_[m], color="0.5", lw=0.9, ls="-.", label="GBT day-ahead" if d == np.floor(tt[0]) else None)
ax.set_xlabel("Day of season (daylight hours only; grey bands: temperature above the trial range)", fontsize=7); ax.set_ylabel("Success probability $v$", fontsize=7); ax.set_ylim(0, 1)
ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
fig.savefig("figs/fig10_heatwave_ts.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
json.dump(dict(mae_pcf_oor=float(np.abs(pc_ - tru)[oor].mean()) if oor.any() else None, mae_gbt_oor=float(np.abs(gb_ - tru)[oor].mean()) if oor.any() else None,
               n_oor=int(oor.sum()), days=[int(d0), int(min(d0 + 5, C.DAYS - 1))]), open("results/ts.json", "w"))
print("ok", oor.sum())
