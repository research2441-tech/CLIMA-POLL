# %% [markdown]
# ## 11. Reliability of the ensemble wind-exceedance estimate used by the chance constraint

# %%
import json, numpy as np
import climapoll as C
OUT = C.OUT; rows = []
for rep in range(2 if C.FAST else 5):
    for i, reg in enumerate(C.REGIMES_TEST):
        O = C.orchard(np.random.default_rng(1000 * rep + 100 + i), reg); rng = np.random.default_rng(4242 + 10 * rep + i)
        for day in range(C.DAYS - 1):
            F = C.ensemble_forecast(O, day, rng); h0, Hh = F["h0"], F["H"]; dl = C.daylight_idx(h0, Hh); hours = h0 + dl
            oT, ex, oR = C.sensor_offsets(O, day, rng)
            Wm = F["wind"][:, None, dl] * ex[None, :, None]
            P = (Wm > C.W_MAX).mean(0); Pm = (Wm.mean(0) > C.W_MAX).astype(float)
            truth = (O["blk"]["wind"][:, hours] > C.W_MAX).astype(float)
            lead = np.broadcast_to(dl // 24, P.shape)
            rows.append(np.stack([P.ravel(), Pm.ravel(), truth.ravel(), lead.ravel()], 1))
A = np.concatenate(rows)
bins = np.linspace(0, 1, 9); rel = []
for k in range(8):
    m = (A[:, 0] >= bins[k]) & (A[:, 0] < bins[k + 1] + (k == 7))
    if m.any(): rel.append([float(A[m, 0].mean()), float(A[m, 2].mean()), int(m.sum())])
out = dict(rel=rel, base=float(A[:, 2].mean()),
           admitted_scen=float(A[A[:, 0] <= 0.10, 2].mean()), n_admitted_scen=int((A[:, 0] <= 0.10).sum()),
           admitted_mean=float(A[A[:, 1] == 0, 2].mean()), n_admitted_mean=int((A[:, 1] == 0).sum()),
           admitted_eps0=float(A[A[:, 0] == 0, 2].mean()), n=int(len(A)),
           brier_ens=float(np.mean((A[:, 0] - A[:, 2]) ** 2)), brier_mean=float(np.mean((A[:, 1] - A[:, 2]) ** 2)))
json.dump(out, open(f"{OUT}/windcal.json", "w")); print(out)
