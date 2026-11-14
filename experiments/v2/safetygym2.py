"""Safety-Gymnasium experiment, v2: two alphabets (generic k-means vs. cost-relevant) and several training seeds / tasks.

python safetygym2.py fit  ENV DIR     pilot baseline episodes -> alphabet definitions (alphabet.npz)
python safetygym2.py gen  ENV DIR     rollouts for baseline + agents -> pools.npz
python safetygym2.py analyse ENV DIR  certificates -> summary.json
DIR contains lam{0.1,0.5,1.0,2.0,5.0}/ (OmniSafe PPOLag runs); the lam5.0 policy is the baseline.

Cost-relevant alphabet: the window of 16 consecutive observation dimensions (a lidar block) whose max over the window best
predicts the per-step cost on the *baseline pilot* (R^2 of the 8-bin predictor with the quantile edges below), binned at the pilot quantiles
{.5,.8,.9,.95,.97,.98,.99}, times 3 k-means clusters of the remaining dimensions: K = 24. Everything is fixed from baseline pilot
data before any agent is rolled out. The pilot mixes noiseless baseline episodes with baseline episodes under action noise sigma=1.5
(a value not used for any evaluated agent) because the noiseless baseline almost never incurs cost.
"""
import os, sys, json, glob
os.environ.pop("MUJOCO_GL", None); os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool

T = 1000; LAMS = [0.1, 0.5, 1.0, 2.0, 5.0]
NOISES = (0.1, 0.3, 0.6, 1.0)
AGENTS = [(f"lam{l}", l, 0.0) for l in LAMS[:-1]] + [(f"basenoise{s}", 5.0, s) for s in NOISES]
QS = np.array([.5, .8, .9, .95, .97, .98, .99]); KC = 16; KH = 24
ENVNAME = DIR = None


class Policy:
    def __init__(self, lam, sigma):
        import torch
        torch.set_num_threads(1)
        ck = sorted(glob.glob(f"{DIR}/lam{lam}/**/epoch-*.pt", recursive=True), key=lambda p: int(p.split("epoch-")[-1][:-3]))[-1]
        c = torch.load(ck, map_location="cpu"); pi = c["pi"]; on = c["obs_normalizer"]
        self.W = [(pi[f"mean.{i}.weight"].numpy(), pi[f"mean.{i}.bias"].numpy()) for i in (0, 2, 4)]
        self.mu = on["_mean"].numpy(); self.sd = np.maximum(on["_std"].numpy(), 1e-8); self.sigma = sigma

    def act(self, obs, rng):
        x = (obs - self.mu) / self.sd
        for i, (w, b) in enumerate(self.W):
            x = x @ w.T + b
            if i < 2: x = np.tanh(x)
        if self.sigma > 0: x = x + self.sigma * rng.standard_normal(x.shape)
        return np.clip(x, -1, 1).astype(np.float32)


def encode(o, A):
    """symbols of one observation under both alphabets."""
    zc = int(np.argmin(((A["centers"] - o) ** 2).sum(1)))
    if A["win"] < 0: return zc, -1
    m = o[A["win"]:A["win"] + 16].max()
    b = int(np.digitize(m, A["edges"]))
    rest = np.delete(o, np.arange(A["win"], A["win"] + 16))
    zr = int(np.argmin(((A["rcenters"] - rest) ** 2).sum(1)))
    return zc, b * 3 + zr


def roll(args):
    name, lam, sigma, n, seed, A, pilot = args
    import safety_gymnasium
    pol = Policy(lam, sigma); env = safety_gymnasium.make(ENVNAME); rng = np.random.default_rng(seed)
    cc = np.zeros((n, KC), np.int32); ch = np.zeros((n, KH), np.int32); sc = np.zeros((n, KC)); sh = np.zeros((n, KH)); costs = np.zeros(n)
    O = []; C = []
    for e in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        for t in range(T):
            o = np.asarray(obs, dtype=np.float32)
            a = pol.act(o, rng)
            obs, _, c, term, trunc, _ = env.step(a); c = float(c)
            if pilot:
                if t % 5 == 0: O.append(o); C.append(c)
            else:
                zc, zh = encode(o, A); cc[e, zc] += 1; sc[e, zc] += c
                ch[e, zh] += 1; sh[e, zh] += c
            costs[e] += c
            if term or trunc: break
    return name, cc, ch, sc, sh, costs, (np.array(O), np.array(C)) if pilot else None


def fit(env, d):
    global ENVNAME, DIR; ENVNAME, DIR = env, d
    from sklearn.cluster import KMeans
    parts = Pool(10).map(roll, [("fit", 5.0, (0.0 if i % 2 == 0 else 1.5), 5, 900 + i, None, True) for i in range(10)])
    O = np.concatenate([p[6][0] for p in parts]); C = np.concatenate([p[6][1] for p in parts])
    best = (-1, None, None)
    for w in range(0, O.shape[1] - 15):
        m = O[:, w:w + 16].max(1); q = np.unique(np.quantile(m, QS)); b = np.digitize(m, q)
        rates = np.array([C[b == i].mean() if (b == i).any() else 0 for i in range(len(q) + 1)])
        r2 = 1 - np.mean((C - rates[b]) ** 2) / max(C.var(), 1e-12)
        if r2 > best[0]: best = (r2, w, None)
    r2, w, _ = best
    m = O[:, w:w + 16].max(1); edges = np.unique(np.quantile(m, QS))
    edges = np.pad(edges, (0, 7 - len(edges)), mode="edge")  # keep 8 bins even if quantiles tie
    rest = np.delete(O, np.arange(w, w + 16), axis=1)
    rk = KMeans(3, n_init=5, random_state=0).fit(rest); ck = KMeans(KC, n_init=5, random_state=0).fit(O)
    np.savez(f"{d}/alphabet.npz", centers=ck.cluster_centers_, rcenters=rk.cluster_centers_, win=w, edges=edges, r2=r2, pilot_cost_rate=C.mean())
    print("window", w, "R2", round(r2, 3), "pilot cost rate", round(float(C.mean()), 4))


def onset(env, d):
    """Re-roll the same pilot as `fit` (same seeds) and choose the onset thresholds: tau such that the pilot cost rate rises by >= 0.05
    between risk level tau and tau+1. Depends on baseline pilot data only (never on agent rollouts or on the certificate episodes)."""
    global ENVNAME, DIR; ENVNAME, DIR = env, d
    A = dict(np.load(f"{d}/alphabet.npz")); w = int(A["win"]); edges = A["edges"]
    parts = Pool(10).map(roll, [("fit", 5.0, (0.0 if i % 2 == 0 else 1.5), 5, 900 + i, None, True) for i in range(10)])
    O = np.concatenate([p[6][0] for p in parts]); C = np.concatenate([p[6][1] for p in parts])
    lv = np.digitize(O[:, w:w + 16].max(1), edges)
    g = np.array([C[lv == l].mean() if (lv == l).any() else 0.0 for l in range(8)])
    on = [int(tau) for tau in range(7) if g[tau + 1] - g[tau] >= 0.05]
    # per-pilot-policy (sigma=0 vs sigma=1.5) level cost rates: a sufficiency check between two policies on pilot data
    lv0 = np.concatenate([np.digitize(p[6][0][:, w:w + 16].max(1), edges) for i, p in enumerate(parts) if i % 2 == 0]); C0 = np.concatenate([p[6][1] for i, p in enumerate(parts) if i % 2 == 0])
    lv1 = np.concatenate([np.digitize(p[6][0][:, w:w + 16].max(1), edges) for i, p in enumerate(parts) if i % 2 == 1]); C1 = np.concatenate([p[6][1] for i, p in enumerate(parts) if i % 2 == 1])
    r0 = [float(C0[lv0 == l].mean()) if (lv0 == l).sum() >= 20 else None for l in range(8)]
    r1 = [float(C1[lv1 == l].mean()) if (lv1 == l).sum() >= 20 else None for l in range(8)]
    json.dump(dict(onset=on, pilot_level_rates=g.tolist(), pilot_visits=[int((lv == l).sum()) for l in range(8)],
                   rates_sigma0=r0, rates_sigma15=r1, visits_sigma0=[int((lv0 == l).sum()) for l in range(8)], visits_sigma15=[int((lv1 == l).sum()) for l in range(8)]), open(f"{d}/onset.json", "w"))
    print("onset", on, np.round(g, 3))


def gen(env, d):
    global ENVNAME, DIR; ENVNAME, DIR = env, d
    A = dict(np.load(f"{d}/alphabet.npz")); A = {k: (int(v) if k == "win" else v) for k, v in A.items()}
    pols = [("base", 5.0, 0.0)] + AGENTS; jobs = []
    for k, (nm, l, s) in enumerate(pols):
        for blk in range(4): jobs.append((f"{nm}#c{blk}", l, s, 100, 100 * k + blk, A, False))
        for blk in range(3): jobs.append((f"{nm}#t{blk}", l, s, 100, 7000 + 100 * k + blk, A, False))
    with Pool(8, initializer=_init, initargs=(env, d)) as p: res = p.map(roll, jobs, chunksize=1)
    out = {}
    for nm, cc, ch, sc, sh, co, _ in res:
        for k, v in dict(cc=cc, ch=ch, sc=sc, sh=sh, co=co).items(): out[f"{nm}|{k}"] = v
    np.savez_compressed(f"{d}/pools.npz", **out)


def _init(e, d):
    global ENVNAME, DIR; ENVNAME, DIR = e, d


def analyse(env, d):
    sys.path.insert(0, os.path.dirname(__file__)); from cert import certificate, ks_certificate, ks_certificate_eb
    P = np.load(f"{d}/pools.npz"); names = ["base"] + [a[0] for a in AGENTS]
    get = lambda nm, tag, blks, k: np.concatenate([P[f"{nm}#{tag}{b}|{k}"] for b in blks])
    res = {}
    for alph, ck, sk, K in (("kmeans", "cc", "sc", KC), ("cost-relevant", "ch", "sh", KH)):
        bc = get("base", "c", range(4), ck); bs = get("base", "c", range(4), sk); bco = get("base", "c", range(4), "co")
        for nm in names:
            assert (get(nm, "c", range(4), sk) <= get(nm, "c", range(4), ck) + 1e-9).all()
            assert (get(nm, "c", range(4), ck).sum(1) == T).all()
        cbar = bs.sum(0) / np.maximum(bc.sum(0), 1)
        rows = []
        for nm in names[1:]:
            ac = get(nm, "c", range(4), ck); asum = get(nm, "c", range(4), sk)
            tco = get(nm, "t", range(3), "co"); Jt = float(tco.mean()); Jse = float(tco.std(ddof=1) / np.sqrt(len(tco)))
            bt = get("base", "t", range(3), "co"); JBse = float(bt.std(ddof=1) / np.sqrt(len(bt)))
            c = certificate(ac, bc, bco, T); dA = ac.sum(0) / ac.sum()
            ok = (ac.sum(0) >= 200) & (bc.sum(0) >= 200)
            rA = asum.sum(0) / np.maximum(ac.sum(0), 1)
            ks = {}
            if alph == "cost-relevant":
                lv = np.arange(KH) // 3
                ks = ks_certificate(ac, bc, bco, T, lv)
                top = ks_certificate(ac, bc, bco, T, lv, onset=[5, 6])        # FIXED onset {5,6}: a misspecification example
                ks.update({"top2_" + k: v for k, v in top.items()})
                oj = json.load(open(f"{d}/onset.json")); on = oj["onset"]; g = oj["pilot_level_rates"]
                eta_hat = sum(max(0.0, g[tau + 1] - g[tau]) for tau in range(7) if tau not in on); eta_bar = eta_hat + 0.05   # total rise outside onset set + margin
                sel = ks_certificate(ac, bc, bco, T, lv, onset=on, eta=eta_bar)       # onset selected from the baseline pilot, with slack eta_bar
                ks.update({"sel_" + k: v for k, v in sel.items()}); ks["onset"] = on; ks["eta_bar"] = eta_bar
                sel0 = ks_certificate(ac, bc, bco, T, lv, onset=on)                     # same without slack (eta = 0)
                ks.update({"sel0_" + k: v for k, v in sel0.items()})
                ks["sel_eb"] = ks_certificate_eb(ac, bc, bco, T, lv, onset=on, eta=eta_bar)["ks_eb_bound"]
                ks["sel0_eb"] = ks_certificate_eb(ac, bc, bco, T, lv, onset=on)["ks_eb_bound"]
                ks["all_eb"] = ks_certificate_eb(ac, bc, bco, T, lv)["ks_eb_bound"]
                lvc = lambda cnt, sm: np.array([sm[:, lv == l].sum() / max(cnt[:, lv == l].sum(), 1) for l in range(8)])
                rb, ra = lvc(bc, bs), lvc(ac, asum)
                ks["mono_base"] = bool((np.diff(rb) >= -1e-9).all()); ks["mono_agent"] = bool((np.diff(ra) >= -1e-9).all())
                ks["level_rates_base"] = rb.tolist(); ks["level_rates_agent"] = ra.tolist()
            drift = {}
            if alph == "cost-relevant":
                lv = np.arange(KH) // 3
                cntA = np.array([ac[:, lv == l].sum() for l in range(8)]); cntB = np.array([bc[:, lv == l].sum() for l in range(8)])
                gA = np.array([asum[:, lv == l].sum() / max(cntA[l], 1) for l in range(8)]); gB = np.array([bs[:, lv == l].sum() / max(cntB[l], 1) for l in range(8)])
                dAl = cntA / cntA.sum(); both = (cntA > 0) & (cntB > 0)
                drift = dict(drift=float(T * (dAl[both] * (gA - gB)[both]).sum()), drift_unseen_mass=float(dAl[~both].sum()),
                             pool_J_A=float(T * (dAl * gA).sum()))
            rows.append(dict(**ks, **drift, J_se=Jse, JB_se=JBse, agent=nm, J_true=Jt, JB_true=float(get("base", "t", range(3), "co").mean()), tv_hat=c["tv_hat"], plugin=c["plugin_bound"],
                             plugin_jsd=c["plugin_bound_via_jsd"], finite=c["finite_sample_bound"], aware=float(T * (dA * cbar).sum()),
                             maxrate=float(np.abs(rA - cbar)[ok].max()) if ok.any() else None, K=K))
        res[alph] = rows
    bt = get("base", "t", range(3), "co"); bc_ = get("base", "c", range(4), "co")
    base = dict(cert_mean=float(bc_.mean()), truth_mean=float(bt.mean()), sd_cert=float(bc_.std(ddof=1)), sd_truth=float(bt.std(ddof=1)),
                z=float((bt.mean() - bc_.mean()) / np.sqrt(bc_.var(ddof=1) / len(bc_) + bt.var(ddof=1) / len(bt))))
    json.dump(dict(T=T, env=env, rows=res, baseline=base), open(f"{d}/summary2.json", "w"), indent=1)
    for alph, rows in res.items():
        print("==", alph)
        for r in rows:
            print(f"{r['agent']:14s} J={r['J_true']:6.1f} aware={r['aware']:6.1f} tv={r['tv_hat']:.3f} plugin={r['plugin']:7.1f} finite={r['finite']:7.1f} maxrate={r['maxrate']}")


if __name__ == "__main__":
    {"fit": fit, "onset": onset, "gen": gen, "analyse": analyse}[sys.argv[1]](sys.argv[2], sys.argv[3])
