"""CartPole experiment for the occupancy-TV certificate.

Cost is a known function of the discretised symbol (so the cost-sufficiency assumption holds exactly).
Symbols: (theta bin x theta_dot bin) grid  +  an absorbing TERM symbol (pole fell) with cost 1.
Usage: python cartpole.py gen OUT_DIR   (rolls out pools; heavy)  |  python cartpole.py analyse OUT_DIR
"""
import sys, json, math, os
import numpy as np
from multiprocessing import Pool

T = 200
TH_BINS, TD_BINS = 10, 3
K = TH_BINS * TD_BINS + 1
TERM = K - 1
TH_EDGES = np.linspace(-0.21, 0.21, TH_BINS + 1)[1:-1]
TD_EDGES = np.array([-0.5, 0.5])
TH_CENTERS = (np.linspace(-0.21, 0.21, TH_BINS + 1)[:-1] + np.linspace(-0.21, 0.21, TH_BINS + 1)[1:]) / 2


def symbol(obs):
    i = int(np.digitize(obs[2], TH_EDGES)); j = int(np.digitize(obs[3], TD_EDGES))
    return i * TD_BINS + j


def cost_of_symbol():
    c = np.zeros(K)
    for i in range(TH_BINS):
        for j in range(TD_BINS):
            c[i * TD_BINS + j] = min(1.0, max(0.0, (abs(TH_CENTERS[i]) - 0.05) / 0.10))
    c[TERM] = 1.0
    return c


COST = cost_of_symbol()

# agent specs: (name, gain on theta_dot, action-noise eps, bias)
def level_of_symbol():
    lv = np.zeros(K, dtype=int)
    for i in range(TH_BINS):
        for j in range(TD_BINS):
            lv[i * TD_BINS + j] = int(abs(i - (TH_BINS - 1) / 2) - 0.5)   # |theta| level 0..4 (cost nondecreasing in it)
    lv[TERM] = 5
    return lv


LEVELS = level_of_symbol()
AGENTS = [("base", 0.5, 0.0, 0.0)] + [(f"eps{e}", 0.5, e, 0.0) for e in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)] + \
         [("gain0.1", 0.1, 0.0, 0.0), ("gain0.25", 0.25, 0.0, 0.0), ("lazy", 0.5, 0.0, 0.03)]


def rollout_pool(args):
    name, gain, eps, bias, n, seed = args
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    counts = np.zeros((n, K), dtype=np.int32); costs = np.zeros(n)
    for e in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        for t in range(T):
            counts[e, symbol(obs)] += 1; costs[e] += COST[symbol(obs)]
            a = int(obs[2] + gain * obs[3] + bias > 0)
            if rng.random() < eps:
                a = int(rng.integers(2))
            obs, _, term, trunc, _ = env.step(a)
            if term:
                rem = T - t - 1
                counts[e, TERM] += rem; costs[e] += rem * COST[TERM]
                break
    return name, counts, costs


def gen(out):
    os.makedirs(out, exist_ok=True)
    N = 10000
    jobs = []
    for k, (nm, g, e, b) in enumerate(AGENTS):
        jobs.append((nm, g, e, b, N, 1000 + k))          # block pool (for repeated certificate draws)
        jobs.append((nm + "@truth", g, e, b, N, 5000 + k))  # independent pool for ground truth
    with Pool(12) as p:
        res = p.map(rollout_pool, jobs)
    for nm, c, co in res:
        np.savez_compressed(os.path.join(out, nm + ".npz"), counts=c, costs=co)


def analyse(out):
    sys.path.insert(0, os.path.dirname(__file__))
    from cert import certificate, ks_certificate, ks_certificate_eb
    ld = lambda nm: np.load(os.path.join(out, nm + ".npz"))
    base = ld("base"); rows = []
    for n in (100, 500, 2000):
        reps = 10000 // n if n < 2000 else 5
        reps = min(reps, 20)
        for nm, *_ in AGENTS[1:]:
            ag, tr = ld(nm), ld(nm + "@truth")
            J_true = float(tr["costs"].mean()); JB_true = float(ld("base@truth")["costs"].mean())
            stats = []
            for r in range(reps):
                sl = slice(r * n, (r + 1) * n)
                c = certificate(ag["counts"][sl], base["counts"][sl], base["costs"][sl], T)
                c.update(ks_certificate(ag["counts"][sl], base["counts"][sl], base["costs"][sl], T, LEVELS))
                c["ks_eb"] = ks_certificate_eb(ag["counts"][sl], base["counts"][sl], base["costs"][sl], T, LEVELS)["ks_eb_bound"]
                c["aware"] = float((ag["counts"][sl].sum(0) / ag["counts"][sl].sum() * COST).sum() * T)
                stats.append(c)
            m = lambda k: float(np.mean([s[k] for s in stats]))
            rows.append(dict(n=n, agent=nm, J_true=J_true, JB_true=JB_true, tv_hat=m("tv_hat"), slack=m("slack"),
                             plugin=m("plugin_bound"), plugin_jsd=m("plugin_bound_via_jsd"), finite=m("finite_sample_bound"),
                             aware=m("aware"), ks_hat=m("ks_hat"), ks_plugin=m("ks_plugin"), ks_finite=m("ks_finite"), ks_eb=m("ks_eb"), viol_ks_eb=float(np.mean([J_true > s["ks_eb"] for s in stats])),
                             viol_ks_plugin=float(np.mean([J_true > s["ks_plugin"] for s in stats])),
                             viol_ks_finite=float(np.mean([J_true > s["ks_finite"] for s in stats])),
                             viol_plugin=float(np.mean([J_true > s["plugin_bound"] for s in stats])),
                             viol_plugin_jsd=float(np.mean([J_true > s["plugin_bound_via_jsd"] for s in stats])),
                             viol_finite=float(np.mean([J_true > s["finite_sample_bound"] for s in stats])), reps=reps))
    json.dump(dict(T=T, K=K, rows=rows), open(os.path.join(out, "summary.json"), "w"), indent=1)
    for r in rows:
        print(f"n={r['n']:5d} {r['agent']:9s} J={r['J_true']:6.1f} JB={r['JB_true']:5.1f} aware={r['aware']:6.1f} plugin={r['plugin']:6.1f} "
              f"jsd={r['plugin_jsd']:6.1f} finite={r['finite']:6.1f} ks_plug={r['ks_plugin']:6.1f} ks_fin={r['ks_finite']:6.1f} viol(p/j/f)={r['viol_plugin']:.2f}/{r['viol_plugin_jsd']:.2f}/{r['viol_finite']:.2f}")


if __name__ == "__main__":
    {"gen": gen, "analyse": analyse}[sys.argv[1]](sys.argv[2])
