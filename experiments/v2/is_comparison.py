"""Comparison with importance-sampling off-policy evaluation (the approach behind high-confidence off-policy evaluation,
Thomas et al. 2015) on CartPole. IS needs the action probabilities of both policies and data from the baseline only; the
occupancy certificate needs rollouts of both but no action probabilities. Policies here are epsilon-noise controllers so that
both have full support and explicit probabilities: pi(a|s) = (1-eps) 1[a = a*(s)] + eps/2.
python is_comparison.py DIR   (DIR/summary_is.json)
"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import cartpole as cp

T = cp.T


def run_baseline(eps_b, eps_as, n, seed):
    import gymnasium as gym
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(seed)
    G = np.zeros(n); logrho = {e: np.zeros(n) for e in eps_as}
    for i in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        for t in range(T):
            s = cp.symbol(obs); G[i] += cp.COST[s]
            astar = int(obs[2] + 0.5 * obs[3] > 0)
            a = int(rng.integers(2)) if rng.random() < eps_b else astar
            pb = (1 - eps_b) * (a == astar) + eps_b / 2
            for e in eps_as:
                logrho[e][i] += math.log(((1 - e) * (a == astar) + e / 2) / pb)
            obs, _, term, trunc, _ = env.step(a)
            if term:
                G[i] += (T - t - 1) * cp.COST[cp.TERM]
                break
    return G, logrho


def truth(eps, n, seed):
    _, counts, costs = cp.rollout_pool(("x", 0.5, eps, 0.0, n, seed))
    return float(costs.mean()), counts, costs


def main(out):
    from multiprocessing import Pool
    sys.path.insert(0, os.path.dirname(__file__)); from cert import certificate
    eps_b = 0.3; eps_as = [0.05, 0.1, 0.2]; n = 2000
    rows = []
    G, lr = run_baseline(eps_b, eps_as, n, 11)
    _, cB, coB = cp.rollout_pool(("B", 0.5, eps_b, 0.0, n, 12))
    for e in eps_as:
        Jt, cA, _ = truth(e, 10000, 5000 + int(e * 1000))
        lw = lr[e]; w = np.exp(lw - lw.max()); ess = float(w.sum() ** 2 / (w ** 2).sum())
        snis = float((w * G).sum() / w.sum())
        ois = float(np.mean(np.exp(lw) * G)) if lw.max() < 700 else float("nan")
        step_max = (1 - e / 2) / (1 - eps_b / 2)           # largest per-step ratio pi_A(a|s)/pi_B(a|s) (greedy action)
        _, cAn, _ = cp.rollout_pool(("A", 0.5, e, 0.0, n, 13 + int(e * 100)))
        c = certificate(cAn, cB, coB, T)
        rows.append(dict(eps_A=e, eps_B=eps_b, J_true=Jt, J_B=float(G.mean()), ess=ess, snis=snis, ois=ois,
                         log10_max_weight=T * math.log10(step_max),
                         log10_hoeffding_slack=math.log10(T * 10 ** (T * math.log10(step_max)) * math.sqrt(math.log(1 / 0.025) / (2 * n))), ours_plugin=c["plugin_bound"], ours_finite=c["finite_sample_bound"]))
        print(rows[-1])
    json.dump(dict(n=n, T=T, rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True); main(os.path.join(sys.argv[1], "summary_is.json"))
