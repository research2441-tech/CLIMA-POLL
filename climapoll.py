# %% [markdown]
# # Microclimate-aware pollination-window forecasting and chance-constrained drone pollination scheduling
# Reproduces all numbers, tables and figures of the manuscript. Runs on the free Colab tier (CPU is sufficient).
# Set FAST=1 in the environment for a quick smoke test.

# %%
import os, json, time, math, warnings
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from scipy.sparse import lil_matrix
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
FAST = os.environ.get("FAST", "0") == "1"
OUT = os.environ.get("OUT", "results"); os.makedirs(OUT, exist_ok=True)
N_REP = 2 if FAST else 5
DAYS = 14; HOURS = np.arange(6, 20)          # daylight operating hours 06:00-19:00
NH = len(HOURS); NB = 8                        # blocks per orchard
M_ENS = 16; LEAD_DAYS = 2                      # ensemble members, planning horizon (days)

# %% [markdown]
# ## 1. Stochastic weather generator (station) with heatwaves and cold spells

# %%
def esat(T):           # saturation vapour pressure, kPa (Tetens)
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def weather_season(rng, regime="nominal"):
    """Hourly station weather for DAYS days (24 h each)."""
    H24 = 24 * DAYS; hod = np.arange(H24) % 24; day = np.arange(H24) // 24
    base = 13.0 + 0.18 * np.arange(DAYS)                      # spring warming trend
    anom = np.zeros(DAYS); a = rng.normal(0, 1.5)
    for d in range(DAYS): a = 0.7 * a + rng.normal(0, 1.6); anom[d] = a
    cloud = np.zeros(DAYS); c = rng.uniform(0.2, 0.7)
    for d in range(DAYS): c = np.clip(0.6 * c + 0.4 * rng.beta(1.6, 1.8), 0, 1); cloud[d] = c
    event = np.zeros(DAYS)
    if regime in ("heatwave", "cold_spell"):
        s = rng.integers(3, DAYS - 5); L = rng.integers(3, 6)
        mag = rng.uniform(10.0, 13.0) if regime == "heatwave" else -rng.uniform(8.0, 10.0)
        prof = np.sin(np.linspace(0.2, np.pi - 0.2, L))
        event[s:s + L] = mag * prof[: len(event[s:s + L])]
    Tmean = base + anom + event
    amp = 7.5 * (1 - 0.6 * cloud) + (1.5 if regime == "heatwave" else 0) * (event > 0)
    T = Tmean[day] + amp[day] * np.sin(2 * np.pi * (hod - 9) / 24)
    e = 0.0
    noise = np.zeros(H24)
    for h in range(H24): e = 0.8 * e + rng.normal(0, 0.45); noise[h] = e
    T = T + noise
    spread = np.clip(5.0 + 2.5 * rng.standard_normal(DAYS) * 0.6 - 3.5 * cloud + (2.5 if regime == "heatwave" else 0) * (event > 1), 1.0, 14.0)
    Td = Tmean[day] - spread[day] + 0.8 * rng.standard_normal(H24) * 0.5
    Td = np.minimum(Td, T - 0.2)
    RH = 100 * esat(Td) / esat(T)
    wmean = rng.weibull(2.0, DAYS) * 2.9 + 0.7
    w = wmean[day] * (1 + 0.45 * np.sin(2 * np.pi * (hod - 10) / 24))
    ew = 0.0; wn = np.zeros(H24)
    for h in range(H24): ew = 0.7 * ew + rng.normal(0, 0.45); wn[h] = ew
    wind = np.clip(w + wn, 0.1, None)
    solar = np.clip(np.sin(np.pi * (hod - 6) / 13), 0, None) * 900 * (1 - 0.7 * cloud[day])
    rain = (rng.random(H24) < 0.12 * (cloud[day] > 0.65)) & (hod > 5)
    for h in np.where(rain)[0][:]:                                 # rain episodes last a few hours
        rain[h:h + rng.integers(1, 4)] = True
    return dict(T=T, RH=np.clip(RH, 8, 100), wind=wind, solar=solar, rain=rain.astype(float), Tmean=Tmean, regime=regime, event=event)

# %% [markdown]
# ## 2. Orchard microclimate, phenology and pollen physiology (ground-truth simulator)

# %%
def vpd(T, RH): return esat(T) * (1 - RH / 100.0)

def true_success(T, RH, wind, solar, rain, cult):
    """Probability that pollen applied in this hour germinates on a receptive stigma (truth, not known to models)."""
    Tmin, Topt, Tmax = cult["Tmin"], cult["Topt"], cult["Tmax"]
    Tc = np.clip(T, Tmin + 1e-3, Tmax - 1e-3)
    gT = ((Tmax - Tc) / (Tmax - Topt)) * ((Tc - Tmin) / (Topt - Tmin)) ** ((Topt - Tmin) / (Tmax - Topt))
    gT = np.where((T <= Tmin) | (T >= Tmax), 0.0, gT)
    D = vpd(T, RH)
    gV = np.exp(-np.maximum(0, D - 0.9) * (0.55 + 0.03 * np.maximum(T - 20, 0)))   # desiccation stronger when hot
    dep = 1.0 / (1.0 + np.exp((wind - 4.2) / 0.55))                                  # spray deposition vs wind
    uv = 1 - 0.12 * np.clip(solar / 900, 0, 1)                                       # UV damage (unmodelled by forecaster)
    return np.clip(0.92 * gT * gV * dep * uv * (1 - rain), 0, 1)

def orchard(rng, regime):
    st = weather_season(rng, regime)
    off_T = rng.normal(0, 0.9, NB); expo = rng.uniform(0.7, 1.25, NB); off_RH = rng.normal(0, 4, NB)
    cult = dict(Tmin=rng.uniform(3, 6), Topt=rng.uniform(20, 23), Tmax=rng.uniform(33, 36))
    blk = {}
    for k in ["T", "RH", "wind", "solar", "rain"]:
        if k == "T": blk[k] = st["T"][None] + off_T[:, None] + 0.4 * rng.standard_normal((NB, st["T"].size)) * 0.3
        elif k == "RH": blk[k] = np.clip(st["RH"][None] + off_RH[:, None], 8, 100)
        elif k == "wind": blk[k] = st["wind"][None] * expo[:, None]
        else: blk[k] = np.repeat(st[k][None], NB, 0)
    blk["succ"] = true_success(blk["T"], blk["RH"], blk["wind"], blk["solar"], blk["rain"], cult)
    # phenology: thermal-time driven anthesis; receptive window shortens with temperature
    n_fl = rng.integers(900, 1300, NB)
    gdh = np.cumsum(np.maximum(blk["T"] - 4.5, 0), axis=1)                           # growing degree hours
    thr = np.array([rng.normal(1500 + 220 * b / NB, 280, n) for b, n in enumerate(n_fl)], dtype=object)
    return dict(st=st, blk=blk, off_T=off_T, expo=expo, off_RH=off_RH, cult=cult, n_fl=n_fl, gdh=gdh, thr=thr, regime=regime)

class Phenology:
    """Flower-level receptivity clock per block; tracks fertilisation state."""
    def __init__(s, O, rng):
        s.O = O; s.open_h = []; s.end_h = []; s.fert = []
        Tm = O["blk"]["T"]
        for b in range(NB):
            thr = np.sort(O["thr"][b]); gd = O["gdh"][b]
            oh = np.searchsorted(gd, thr).astype(float)                              # hour of anthesis
            oh = np.minimum(oh, gd.size + 100)
            # receptive window (hours): 60 h at 15 C mean, shorter when warm
            Tw = np.array([Tm[b, int(min(h, gd.size - 1)):int(min(h + 48, gd.size))].mean() if h < gd.size else 15 for h in oh])
            L = 60.0 * np.exp(-0.055 * (Tw - 15.0)) * rng.uniform(0.85, 1.15, oh.size)
            s.open_h.append(oh + 6.0); s.end_h.append(oh + 6.0 + L); s.fert.append(np.zeros(oh.size, bool))
    def pool(s, b, h):
        m = (s.open_h[b] <= h) & (s.end_h[b] > h) & ~s.fert[b]; return m
    def count(s, b, h): return int(s.pool(b, h).sum())

# %% [markdown]
# ## 3. Forecast products: station ensemble forecasts and in-orchard sensor history

# %%
LEAD_SD = dict(T=(0.8, 0.045), RH=(4.0, 0.16), wind=(0.55, 0.02))

def ensemble_forecast(O, day, rng, m=M_ENS):
    """Station ensemble for hours [day*24, (day+LEAD_DAYS)*24). Returns dict var -> (m, H) arrays (station level)."""
    st = O["st"]; h0 = day * 24; H = LEAD_DAYS * 24; h1 = min(h0 + H, 24 * DAYS); Hh = h1 - h0
    out = {}
    lead = np.arange(Hh)
    for v in ["T", "RH", "wind"]:
        a, b = LEAD_SD[v]; sd = a + b * lead
        # systematic forecast error shared by members (unknown to the user) + member spread
        sys_err = np.zeros(Hh); e = rng.normal(0, 1)
        for k in range(Hh): e = 0.97 * e + rng.normal(0, 0.25); sys_err[k] = e
        centre = st[v][h0:h1] + 0.8 * sd * sys_err
        mem = np.zeros((m, Hh))
        for j in range(m):
            e = rng.normal(0, 1); path = np.zeros(Hh)
            for k in range(Hh): e = 0.95 * e + rng.normal(0, 0.31); path[k] = e
            mem[j] = centre + 0.9 * sd * path
        if v == "RH": mem = np.clip(mem, 8, 100)
        if v == "wind": mem = np.clip(mem, 0.1, None)
        out[v] = mem
    out["solar"] = np.repeat(st["solar"][h0:h1][None], m, 0) * rng.uniform(0.85, 1.15, (m, 1))
    p_rain = np.clip(st["rain"][h0:h1] * 0.7 + 0.08, 0, 1)
    out["rain"] = (rng.random((m, Hh)) < p_rain).astype(float)
    out["h0"] = h0; out["H"] = Hh
    return out

def sensor_offsets(O, day, rng, window=72, use=True):
    """Block offsets estimated from in-orchard sensors (block minus station) over the last `window` hours."""
    if not use or day == 0:
        return np.zeros(NB), np.ones(NB), np.zeros(NB)
    h1 = day * 24; h0 = max(0, h1 - window)
    bT = O["blk"]["T"][:, h0:h1] + rng.normal(0, 0.3, (NB, h1 - h0)); sT = O["st"]["T"][h0:h1]
    bW = O["blk"]["wind"][:, h0:h1] * rng.uniform(0.95, 1.05, (NB, 1)); sW = O["st"]["wind"][h0:h1]
    bR = O["blk"]["RH"][:, h0:h1] + rng.normal(0, 2, (NB, h1 - h0)); sR = O["st"]["RH"][h0:h1]
    oT = (bT - sT).mean(1); ex = (bW * sW).sum(1) / (sW * sW).sum(); oR = (bR - sR).mean(1)
    return oT, ex, oR

# %% [markdown]
# ## 4. Pollination-success forecasters
# * PCF (proposed): physiologically constrained factorised model; cardinal-temperature beta x VPD desiccation x wind deposition x rain.
# * Baselines: hour-of-day climatology, gradient-boosted trees, MLP on the same inputs.

# %%
def feats(T, RH, W, S, Rn, hod):
    T, RH, W, S, Rn, hod = np.broadcast_arrays(T, RH, W, S, Rn, hod)
    return np.stack([T, RH, vpd(T, RH), W, S / 900, Rn, np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], -1)

class PCF:
    """theta = (Tmin, Topt, Tmax, a_v, d0, w50, s_w, p_max)."""
    def __init__(s): s.th = np.array([4.0, 21.0, 34.0, 0.6, 1.0, 4.0, 0.6, 0.9])
    @staticmethod
    def f(th, T, RH, W, Rn):
        Tmin, Topt, Tmax, av, d0, w50, sw, pm = th
        Tmin = min(Tmin, Topt - 1); Tmax = max(Tmax, Topt + 1)
        Tc = np.clip(T, Tmin + 1e-3, Tmax - 1e-3)
        gT = ((Tmax - Tc) / (Tmax - Topt)) * ((Tc - Tmin) / (Topt - Tmin)) ** ((Topt - Tmin) / (Tmax - Topt))
        gT = np.where((T <= Tmin) | (T >= Tmax), 0.0, gT)
        gV = np.exp(-max(av, 0) * np.maximum(0, vpd(T, RH) - d0))
        dep = 1 / (1 + np.exp((W - w50) / max(sw, 0.05)))
        return np.clip(1 / (1 + np.exp(-pm)) * gT * gV * dep * (1 - Rn), 1e-5, 1 - 1e-5)
    def fit(s, T, RH, W, Rn, k, n):
        def nll(th):
            p = s.f(th, T, RH, W, Rn); return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p)) / n.sum()
        best = None
        for init in [s.th, s.th + np.array([1, -1, 1, 0.2, 0.2, -0.3, 0.1, 0.3])]:
            r = minimize(nll, init, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-3, fatol=1e-7))
            if best is None or r.fun < best.fun: best = r
        s.th = best.x; return s
    def predict(s, T, RH, W, S, Rn, hod): T, RH, W, Rn = np.broadcast_arrays(T, RH, W, Rn); return s.f(s.th, T, RH, W, Rn)

class Clim:
    def fit(s, hod, k, n):
        s.p = np.array([k[hod == h].sum() / max(1, n[hod == h].sum()) for h in range(24)]); return s
    def predict(s, T, RH, W, S, Rn, hod): T, hod = np.broadcast_arrays(T, hod); return np.clip(s.p[hod.astype(int)], 1e-4, 1 - 1e-4)

class GBT:
    def fit(s, X, k, n):
        Xr = np.repeat(X, 2, 0); y = np.tile([1, 0], len(X)); w = np.stack([k, n - k], 1).ravel()
        s.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=0).fit(Xr, y, sample_weight=w); return s
    def predict(s, T, RH, W, S, Rn, hod): return np.clip(s.m.predict_proba(feats(T, RH, W, S, Rn, hod).reshape(-1, 8))[:, 1].reshape(np.broadcast_shapes(np.shape(T), np.shape(RH), np.shape(W), np.shape(S))), 1e-4, 1 - 1e-4)

class MLP:
    def fit(s, X, k, n, seed=0):
        s.sc = StandardScaler().fit(X)
        Xr = np.repeat(s.sc.transform(X), 2, 0); y = np.tile([1, 0], len(X)); w = np.stack([k, n - k], 1).ravel()
        idx = np.random.default_rng(seed).choice(len(Xr), size=min(len(Xr), 60000), p=w / w.sum())
        s.m = MLPClassifier((64, 64), max_iter=300, random_state=seed, early_stopping=True).fit(Xr[idx], y[idx]); return s
    def predict(s, T, RH, W, S, Rn, hod):
        return np.clip(s.m.predict_proba(s.sc.transform(feats(T, RH, W, S, Rn, hod).reshape(-1, 8)))[:, 1].reshape(np.broadcast_shapes(np.shape(T), np.shape(RH), np.shape(W), np.shape(S))), 1e-4, 1 - 1e-4)

def trial_data(orchards, rng, per_hour=2, grains=25):
    """Controlled pollen-viability trials: hourly germination counts on detached stigmas in each block."""
    T, RH, W, S, Rn, hod, K, Nn = [], [], [], [], [], [], [], []
    for O in orchards:
        B = O["blk"]; H = B["T"].shape[1]
        for b in range(NB):
            hh = np.array([h for h in range(H) if 6 <= h % 24 < 20])
            for _ in range(per_hour):
                p = B["succ"][b, hh]; k = rng.binomial(grains, p)
                T.append(B["T"][b, hh] + rng.normal(0, 0.3, hh.size)); RH.append(B["RH"][b, hh]); W.append(B["wind"][b, hh])
                S.append(B["solar"][b, hh]); Rn.append(B["rain"][b, hh]); hod.append(hh % 24); K.append(k); Nn.append(np.full(hh.size, grains))
    c = lambda L: np.concatenate(L)
    return c(T), c(RH), c(W), c(S), c(Rn), c(hod), c(K).astype(float), c(Nn).astype(float)

# %% [markdown]
# ## 5. Mission scheduling
# Expected fertilised flowers of block b on day d: R_bd * (1 - exp(-kappa * sum_h x_bh v_bh)), concave in sorties.
# Scenario (ensemble) average, outer-approximated by tangent cuts; wind chance constraint by scenario counting.

# %%
KAPPA = 0.35; X_MAX = 3; FLEET_PER_H = 4; SORTIES_PER_DAY = 24; POLLEN_G = 2.5; POLLEN_PER_DAY = 40.0
W_MAX = 4.5; EPS = 0.10
TANG = np.array([0.0, 0.2, 0.45, 0.8, 1.2, 1.8, 2.6, 4.0])

def solve_schedule(V, R, wind_ok, days_in_h, eps_cost=1e-3):
    """V: (S, NB, Hp) success per scenario; R: (NB, Dp) receptive pools per plan day; wind_ok: (NB, Hp) bool;
    days_in_h: (Hp,) day index (0..Dp-1) of each plan hour. Returns integer sorties x (NB, Hp)."""
    S, nb, Hp = V.shape; Dp = R.shape[1]
    nx = nb * Hp; nt = S * nb * Dp; nvar = nx + nt
    ix = lambda b, h: b * Hp + h; it = lambda s, b, d: nx + (s * nb + b) * Dp + d
    c = np.zeros(nvar)
    for s in range(S):
        for b in range(nb):
            for d in range(Dp): c[it(s, b, d)] = -R[b, d] / S
    c[:nx] += eps_cost * POLLEN_G
    rows = []; lo = []; hi = []
    A = lil_matrix((nt * len(TANG) + Hp + Dp * 2, nvar))
    r = 0
    for s in range(S):
        for b in range(nb):
            for d in range(Dp):
                hs = np.where(days_in_h == d)[0]
                for z0 in TANG:          # t <= phi(z0) + phi'(z0) (z - z0), phi(z) = 1 - exp(-kappa z)
                    g = KAPPA * np.exp(-KAPPA * z0); A[r, it(s, b, d)] = 1.0
                    for h in hs: A[r, ix(b, h)] = -g * V[s, b, h]
                    lo.append(-np.inf); hi.append((1 - np.exp(-KAPPA * z0)) - g * z0); r += 1
    for h in range(Hp):
        for b in range(nb): A[r, ix(b, h)] = 1.0
        lo.append(0); hi.append(FLEET_PER_H); r += 1
    for d in range(Dp):
        hs = np.where(days_in_h == d)[0]
        for b in range(nb):
            for h in hs: A[r, ix(b, h)] = 1.0
        lo.append(0); hi.append(SORTIES_PER_DAY); r += 1
        for b in range(nb):
            for h in hs: A[r, ix(b, h)] = POLLEN_G
        lo.append(0); hi.append(POLLEN_PER_DAY); r += 1
    A = A[:r].tocsr()
    ub = np.concatenate([np.where(wind_ok.ravel(), X_MAX, 0), np.ones(nt)])
    integ = np.concatenate([np.ones(nx), np.zeros(nt)])
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(np.zeros(nvar), ub),
               options=dict(time_limit=20, mip_rel_gap=1e-3))
    if res.x is None: return np.zeros((nb, Hp), int)
    return np.round(res.x[:nx]).astype(int).reshape(nb, Hp)

# %% [markdown]
# ## 6. Season simulation under a scheduling policy

# %%
def daylight_idx(h0, H):
    return np.array([k for k in range(H) if 6 <= (h0 + k) % 24 < 20])

def run_season(O, policy, models, rng_seed, eps=EPS, use_sensors=True, scenarios=True, forecaster="pcf", margin=0.0, log=False, pollen_day=None):
    rng = np.random.default_rng(rng_seed)
    ph = Phenology(O, np.random.default_rng(rng_seed + 1))
    B = O["blk"]; total = int(sum(O["n_fl"]))
    sorties = 0; viol = 0; fert = 0; pollen = 0.0; solve_t = []; LOG = []
    global POLLEN_PER_DAY
    _pd_saved = POLLEN_PER_DAY
    if pollen_day is not None: POLLEN_PER_DAY = pollen_day
    for day in range(DAYS):
        F = ensemble_forecast(O, day, rng)
        h0, Hh = F["h0"], F["H"]; dl = daylight_idx(h0, Hh); hours = h0 + dl
        oT, ex, oR = sensor_offsets(O, day, rng, use=use_sensors)
        hod = hours % 24
        # block-level ensemble
        Tm = F["T"][:, None, dl] + oT[None, :, None]; Wm = F["wind"][:, None, dl] * ex[None, :, None]
        Rm = np.clip(F["RH"][:, None, dl] + oR[None, :, None], 8, 100); Sm = F["solar"][:, None, dl]; Rn = F["rain"][:, None, dl] * np.ones((1, NB, 1))
        days_in_h = dl // 24
        Dp = int(days_in_h.max()) + 1
        # receptive pools expected per plan day (flowers receptive at noon of that day, not yet fertilised)
        R = np.array([[ph.count(b, h0 + 24 * d + 12) for d in range(Dp)] for b in range(NB)], float)
        x = np.zeros((NB, dl.size), int)
        if policy == "calendar":                       # fixed daily round: 2 sorties per block from 08:00, fleet-limited
            for b in range(NB):
                hh = day * 24 + 8 + b // (FLEET_PER_H // 2)            # 2 blocks per hour, 08:00-11:00
                k = np.where(hours == hh)[0]
                if k.size: x[b, k[0]] = 2
        elif policy == "greedy":
            mT, mW, mR = Tm.mean(0), Wm.mean(0), Rm.mean(0)
            ok = (mT > 12) & (mT < 28) & (mW < W_MAX - 0.8) & (Rn.mean(0) < 0.3) & (vpd(mT, mR) < 1.8)
            used_day = 0; used_p = 0.0
            for k, h in enumerate(hours):
                if h // 24 != day: continue
                cap = FLEET_PER_H
                order = np.argsort(-R[:, 0])
                for b in order:
                    if cap <= 0 or used_day >= SORTIES_PER_DAY or used_p + POLLEN_G > POLLEN_PER_DAY: break
                    if ok[b, k] and R[b, 0] > 0 and x[b, :].sum() < 3:
                        n_ = min(2, cap); x[b, k] += n_; cap -= n_; used_day += n_; used_p += n_ * POLLEN_G
        else:
            mdl = models[forecaster]
            if policy == "milp_hybrid":                 # mean-forecast objective, scenario chance constraint
                mT, mW, mR = Tm.mean(0), Wm.mean(0), Rm.mean(0)
                V = mdl.predict(mT, mR, mW, Sm.mean(0), (Rn.mean(0) > 0.5).astype(float), np.broadcast_to(hod, mT.shape))[None]
                wind_ok = (Wm > W_MAX).mean(0) <= eps
                t0 = time.time(); x = solve_schedule(V, R, wind_ok, days_in_h); solve_t.append(time.time() - t0)
            if policy in ("milp_mean", "milp_scen", "milp_oracle"):
                if policy == "milp_oracle":
                    V = B["succ"][None, :, hours]; wind_ok = B["wind"][:, hours] <= W_MAX
                elif policy == "milp_mean" or not scenarios:
                    mT, mW, mR = Tm.mean(0), Wm.mean(0), Rm.mean(0)
                    V = mdl.predict(mT, mR, mW, Sm.mean(0), (Rn.mean(0) > 0.5).astype(float), np.broadcast_to(hod, mT.shape))[None]
                    wind_ok = mW <= W_MAX - margin
                else:
                    V = mdl.predict(Tm, Rm, Wm, Sm, Rn, np.broadcast_to(hod, Tm.shape))
                    wind_ok = (Wm > W_MAX).mean(0) <= eps
                t0 = time.time(); x = solve_schedule(V, R, wind_ok, days_in_h); solve_t.append(time.time() - t0)
            # execute only the first day of the plan
            x[:, days_in_h > 0] = 0
        # ----- execute with true weather -----
        for k, h in enumerate(hours):
            if h // 24 != day: continue
            for b in range(NB):
                n = int(x[b, k])
                if n == 0: continue
                sorties += n; pollen += n * POLLEN_G
                if log: LOG.append((int(h % 24), float(B["succ"][b, h]), float(B["T"][b, h]), float(B["wind"][b, h]), n))
                if B["wind"][b, h] > W_MAX: viol += n
                m = ph.pool(b, h); idx = np.where(m)[0]
                if idx.size == 0: continue
                p = 1 - np.exp(-KAPPA * n * B["succ"][b, h])
                hit = idx[rng.random(idx.size) < p]; ph.fert[b][hit] = True; fert += hit.size
        # background insect pollination (small, weather dependent)
        for h in range(day * 24 + 6, day * 24 + 20):
            for b in range(NB):
                idx = np.where(ph.pool(b, h))[0]
                if idx.size: hit = idx[rng.random(idx.size) < 0.004 * B["succ"][b, h]]; ph.fert[b][hit] = True
    nat = sum(int(f.sum()) for f in ph.fert) - fert
    POLLEN_PER_DAY = _pd_saved
    extra = dict(log=LOG) if log else {}
    return dict(**extra, fruit_set=sum(int(f.sum()) for f in ph.fert) / total, art_fert=fert, nat_fert=nat, sorties=sorties, viol=viol,
                viol_rate=viol / max(1, sorties), pollen=pollen, pollen_per_fruit=pollen / max(1, fert), solve_s=float(np.mean(solve_t)) if solve_t else 0.0)

# %% [markdown]
# ## 7. Forecast evaluation helpers

# %%
def brier(p, k, n): return float(np.sum(n * (p * p) - 2 * p * k + k) / n.sum())   # per-grain Brier score
def logloss(p, k, n): p = np.clip(p, 1e-5, 1 - 1e-5); return float(-np.sum(k * np.log(p) + (n - k) * np.log(1 - p)) / n.sum())
def mae_true(p, ptrue): return float(np.mean(np.abs(p - ptrue)))

def calib_curve(p, k, n, bins=10):
    e = np.linspace(0, 1, bins + 1); out = []
    for i in range(bins):
        m = (p >= e[i]) & (p < e[i + 1] + (i == bins - 1))
        if m.any(): out.append([float(p[m].mean()), float(k[m].sum() / n[m].sum()), int(m.sum())])
    return out

TRAIN_T = [None, None]
def forecast_skill(O, models, rng, use_sensors=True):
    """Day-ahead forecast skill against true block success probabilities and simulated trial outcomes."""
    P = {k: [] for k in models}; truth = []; Ks = []; Ns = []; lead = []; Tt = []
    for day in range(DAYS - 1):
        F = ensemble_forecast(O, day, rng); h0, Hh = F["h0"], F["H"]; dl = daylight_idx(h0, Hh); hours = h0 + dl
        oT, ex, oR = sensor_offsets(O, day, rng, use=use_sensors); hod = hours % 24
        Tm = F["T"][:, None, dl] + oT[None, :, None]; Wm = F["wind"][:, None, dl] * ex[None, :, None]
        Rm = np.clip(F["RH"][:, None, dl] + oR[None, :, None], 8, 100); Sm = F["solar"][:, None, dl]; Rn = F["rain"][:, None, dl] * np.ones((1, NB, 1))
        pt = O["blk"]["succ"][:, hours]
        for name, mdl in models.items():
            P[name].append(mdl.predict(Tm, Rm, Wm, Sm, Rn, np.broadcast_to(hod, Tm.shape)).mean(0))
        Tt.append(O["blk"]["T"][:, hours]); truth.append(pt); k = rng.binomial(25, pt); Ks.append(k); Ns.append(np.full(pt.shape, 25)); lead.append(np.broadcast_to(dl // 24, pt.shape))
    c = lambda L: np.concatenate([x.ravel() for x in L])
    t, K, Nn, ld, Ttrue = c(truth), c(Ks).astype(float), c(Ns).astype(float), c(lead), c(Tt)
    out = {}
    for name in models:
        p = c(P[name])
        oor = (Ttrue > TRAIN_T[1]) | (Ttrue < TRAIN_T[0]) if TRAIN_T[0] is not None else np.zeros_like(t, bool)
        out[name] = dict(brier=brier(p, K, Nn), logloss=logloss(p, K, Nn), mae=mae_true(p, t), mae_d1=mae_true(p[ld == 0], t[ld == 0]),
                         mae_d2=mae_true(p[ld == 1], t[ld == 1]), calib=calib_curve(p, K, Nn),
                         mae_oor=mae_true(p[oor], t[oor]) if oor.any() else None, n_oor=int(oor.sum()), brier_oor=brier(p[oor], K[oor], Nn[oor]) if oor.any() else None)
    return out

# %% [markdown]
# ## 8. Replicate driver

# %%
REGIMES_TEST = ["nominal", "nominal", "heatwave", "heatwave", "cold_spell", "cold_spell"]
POLICIES = ["calendar", "greedy", "milp_mean", "milp_scen", "milp_oracle"]

def replicate(rep, log=print):
    t0 = time.time(); rng = np.random.default_rng(500 + rep)
    train = [orchard(np.random.default_rng(1000 * rep + i), "nominal") for i in range(8)]
    test = [orchard(np.random.default_rng(1000 * rep + 100 + i), r) for i, r in enumerate(REGIMES_TEST)]
    T, RH, W, S, Rn, hod, K, Nn = trial_data(train, rng)
    X = feats(T, RH, W, S, Rn, hod); TRAIN_T[0], TRAIN_T[1] = float(np.percentile(T, 0.5)), float(np.percentile(T, 99.5))
    models = dict(pcf=PCF().fit(T, RH, W, Rn, K, Nn), clim=Clim().fit(hod, K, Nn), gbt=GBT().fit(X, K, Nn), mlp=MLP().fit(X, K, Nn, seed=rep))
    log(f"[rep {rep}] forecasters fitted in {time.time()-t0:.0f}s; trial records {len(K)}; PCF theta={np.round(models['pcf'].th,2)}")
    res = dict(rep=rep, theta=models["pcf"].th.tolist(), skill={}, seasons=[], ablation=[])
    # forecast skill per regime
    for i, O in enumerate(test):
        sk = forecast_skill(O, models, np.random.default_rng(7000 + 10 * rep + i))
        res["skill"].setdefault(O["regime"], []).append(sk)
    sk_nos = forecast_skill(test[0], {"pcf": models["pcf"]}, np.random.default_rng(7000 + 10 * rep), use_sensors=False)
    res["skill_no_sensor"] = sk_nos
    # policies
    for i, O in enumerate(test):
        for pol in POLICIES:
            r = run_season(O, pol, models, rng_seed=9000 + 100 * rep + i)
            r.update(policy=pol, regime=O["regime"], season=i); res["seasons"].append(r)
        # ablations on the proposed scheduler
        for tag, kw in [("gbt_forecaster", dict(forecaster="gbt")), ("mlp_forecaster", dict(forecaster="mlp")), ("no_sensors", dict(use_sensors=False)),
                        ("eps_0.00", dict(eps=0.0)), ("eps_0.20", dict(eps=0.2)), ("eps_0.35", dict(eps=0.35))]:
            r = run_season(O, "milp_scen", models, rng_seed=9000 + 100 * rep + i, **kw); r.update(tag=tag, regime=O["regime"], season=i); res["ablation"].append(r)
        log(f"[rep {rep}] season {i} ({O['regime']}) done, {time.time()-t0:.0f}s")
    # example season traces for figures (rep 0 only)
    if rep == 0:
        O = test[2]
        res["trace"] = dict(T=O["blk"]["T"][0].tolist(), wind=O["blk"]["wind"][0].tolist(), succ=O["blk"]["succ"][0].tolist(), RH=O["blk"]["RH"][0].tolist())
        grid = np.linspace(-2, 42, 89)
        res["curves"] = dict(T=grid.tolist(), pcf=models["pcf"].predict(grid, np.full_like(grid, 60), np.full_like(grid, 1.5), np.full_like(grid, 400), np.zeros_like(grid), np.full_like(grid, 12)).tolist(),
                             gbt=models["gbt"].predict(grid, np.full_like(grid, 60), np.full_like(grid, 1.5), np.full_like(grid, 400), np.zeros_like(grid), np.full_like(grid, 12)).tolist(),
                             mlp=models["mlp"].predict(grid, np.full_like(grid, 60), np.full_like(grid, 1.5), np.full_like(grid, 400), np.zeros_like(grid), np.full_like(grid, 12)).tolist(),
                             truth=true_success(grid, np.full_like(grid, 60), np.full_like(grid, 1.5), np.full_like(grid, 400), np.zeros_like(grid), test[0]["cult"]).tolist(),
                             train_T=[float(np.percentile(T, 1)), float(np.percentile(T, 99))])
    log(f"[rep {rep}] finished in {time.time()-t0:.0f}s")
    return res

if __name__ == "__main__":
    allres = []
    for rep in range(N_REP):
        allres.append(replicate(rep)); json.dump(allres, open(f"{OUT}/results.json", "w"))
    print("finished")
