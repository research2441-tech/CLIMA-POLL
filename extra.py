# %% [markdown]
# ## 10. Where and when sorties are flown, and sensitivity to the pollen budget

# %%
import json, numpy as np
import climapoll as C
OUT = C.OUT; ex = dict(log=[], budget=[])
rep = 0; rng = np.random.default_rng(500 + rep)
train = [C.orchard(np.random.default_rng(1000 * rep + i), "nominal") for i in range(8)]
test = [C.orchard(np.random.default_rng(1000 * rep + 100 + i), r) for i, r in enumerate(C.REGIMES_TEST)]
T, RH, W, S, Rn, hod, K, Nn = C.trial_data(train, rng)
models = dict(pcf=C.PCF().fit(T, RH, W, Rn, K, Nn))
for i, O in enumerate(test):
    for pol in ["calendar", "greedy", "milp_mean", "milp_scen"]:
        r = C.run_season(O, pol, models, 9000 + i, log=True)
        ex["log"].append(dict(policy=pol, regime=O["regime"], season=i, log=r["log"]))
    for pd in [25.0, 60.0]:
        for pol in ["greedy", "milp_scen"]:
            r = C.run_season(O, pol, models, 9000 + i, pollen_day=pd); r.update(policy=pol, regime=O["regime"], season=i, pollen_day=pd); ex["budget"].append(r)
    print("extra season", i, flush=True)
    json.dump(ex, open(f"{OUT}/extra.json", "w"))
print("extra done")
