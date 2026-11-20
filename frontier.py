# %% [markdown]
# ## 9. Risk-return frontier: fruit set versus wind-limit violations

# %%
import json, numpy as np, time
import climapoll as C
OUT = C.OUT; fr = []
for rep in range(2 if C.FAST else 3):
    rng = np.random.default_rng(500 + rep)
    train = [C.orchard(np.random.default_rng(1000 * rep + i), "nominal") for i in range(8)]
    test = [C.orchard(np.random.default_rng(1000 * rep + 100 + i), r) for i, r in enumerate(C.REGIMES_TEST)]
    T, RH, W, S, Rn, hod, K, Nn = C.trial_data(train, rng)
    models = dict(pcf=C.PCF().fit(T, RH, W, Rn, K, Nn))
    for i, O in enumerate(test):
        seed = 9000 + 100 * rep + i
        for m in [0.0, 0.5, 1.0, 1.5]:
            r = C.run_season(O, "milp_mean", models, seed, margin=m); r.update(rep=rep, season=i, regime=O["regime"], family="mean+margin", level=m); fr.append(r)
        for e in [0.05, 0.10, 0.20, 0.35, 0.50]:
            r = C.run_season(O, "milp_scen", models, seed, eps=e); r.update(rep=rep, season=i, regime=O["regime"], family="scenario", level=e); fr.append(r)
            r = C.run_season(O, "milp_hybrid", models, seed, eps=e); r.update(rep=rep, season=i, regime=O["regime"], family="hybrid", level=e); fr.append(r)
        print(f"frontier rep {rep} season {i} done", flush=True)
        json.dump(fr, open(f"{OUT}/frontier.json", "w"))
print("frontier done")
