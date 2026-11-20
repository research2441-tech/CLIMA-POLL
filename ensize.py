# %% [markdown]
# ## 10. Ensemble size and empirical risk of the chance constraint

# %%
import json, numpy as np, time
import climapoll as C
OUT = C.OUT; res = []
_orig_ens = C.ensemble_forecast
for rep in range(1 if C.FAST else 3):
    rng = np.random.default_rng(500 + rep)
    train = [C.orchard(np.random.default_rng(1000 * rep + i), "nominal") for i in range(8)]
    test = [C.orchard(np.random.default_rng(1000 * rep + 100 + i), r) for i, r in enumerate(C.REGIMES_TEST)]
    T, RH, W, S, Rn, hod, K, Nn = C.trial_data(train, rng)
    models = dict(pcf=C.PCF().fit(T, RH, W, Rn, K, Nn))
    for i, O in enumerate(test):
        seed = 9000 + 100 * rep + i
        for M in [4, 8, 16, 32, 64]:
            C.ensemble_forecast = (lambda O_, d_, r_, m=M: _orig_ens(O_, d_, r_, m))
            t0 = time.time()
            r = C.run_season(O, "milp_scen", models, seed, eps=0.10)
            r.update(rep=rep, season=i, regime=O["regime"], M=M, wall=time.time() - t0); res.append(r)
        C.ensemble_forecast = _orig_ens
        print(f"ensize rep {rep} season {i} done", flush=True)
        json.dump(res, open(f"{OUT}/ensize.json", "w"))
print("ensize done")
