"""Fresh, episode-bootstrap sensitivity study for the cost-relevant alphabet.

Run with the pinned Safety-Gymnasium environment and released checkpoints:
  python pilot_sensitivity.py --checkpoint-dir PATH --task SafetyCarGoal1-v0 --out PATH

The pilot is independent of the evaluation rollouts. The algorithm (window
scan, quantile bins, 3-cluster residual partition, onset, eta) is held fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import safetygym2 as sg
from cert import ks_certificate_eb


QS = np.array([.5, .8, .9, .95, .97, .98, .99])
T = 1000
SIZES = (10, 25, 50, 100)
THRESHOLDS = (0.025, 0.05, 0.10)


def collect_evaluation(env_name: str, checkpoint_dir: Path, lam: float, n: int, seed: int):
    """Collect fresh full-observation episodes, independent of the pilot."""
    import safety_gymnasium

    sg.ENVNAME, sg.DIR = env_name, str(checkpoint_dir)
    policy = sg.Policy(lam, 0.0)
    env = safety_gymnasium.make(env_name)
    rng = np.random.default_rng(seed)
    obs_all = None
    costs_all = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        if obs_all is None:
            obs_all = np.zeros((n, T, np.asarray(obs).size), dtype=np.float32)
        for t in range(T):
            obs = np.asarray(obs, dtype=np.float32)
            obs_all[i, t] = obs
            action = policy.act(obs, rng)
            obs, _, cost, term, trunc, _ = env.step(action)
            costs_all[i, t] = float(cost)
            if term or trunc:
                # This environment's released experiment uses fixed-length episodes.
                if t != T - 1:
                    raise RuntimeError(f"Unexpected early termination at step {t + 1}")
                break
    env.close()
    return obs_all, costs_all.sum(1)


def collect_pilot(env_name: str, checkpoint_dir: Path, n_each: int, seed: int):
    """Reuse the released rollout function so pilot collection matches v1 exactly."""
    sg.ENVNAME, sg.DIR = env_name, str(checkpoint_dir)
    p0 = sg.roll(("pilot_sigma0", 5.0, 0.0, n_each, seed, None, True))[6]
    p1 = sg.roll(("pilot_sigma15", 5.0, 1.5, n_each, seed + 1, None, True))[6]
    # The pilot recorder samples every fifth step; each episode has T/5 rows.
    m = T // 5
    return (
        p0[0].reshape(n_each, m, -1), p0[1].reshape(n_each, m),
        p1[0].reshape(n_each, m, -1), p1[1].reshape(n_each, m),
    )


def fit_alphabet(pilot, ids0, ids1):
    o0, c0, o1, c1 = pilot
    o = np.concatenate([o0[ids0].reshape(-1, o0.shape[-1]), o1[ids1].reshape(-1, o1.shape[-1])])
    c = np.concatenate([c0[ids0].reshape(-1), c1[ids1].reshape(-1)])
    best = (-np.inf, None)
    for w in range(o.shape[1] - 15):
        risk = o[:, w:w + 16].max(1)
        edges = np.unique(np.quantile(risk, QS))
        level = np.digitize(risk, edges)
        rates = np.array([c[level == i].mean() if np.any(level == i) else 0.0 for i in range(len(edges) + 1)])
        prediction = rates[level]
        r2 = 1 - np.mean((c - prediction) ** 2) / max(float(c.var()), 1e-12)
        if r2 > best[0]:
            best = (float(r2), int(w))
    r2, w = best
    risk = o[:, w:w + 16].max(1)
    edges = np.unique(np.quantile(risk, QS))
    edges = np.pad(edges, (0, max(0, 7 - len(edges))), mode="edge")[:7]
    level = np.digitize(risk, edges)
    rates = np.array([c[level == i].mean() if np.any(level == i) else 0.0 for i in range(8)])
    rest = np.delete(o, np.arange(w, w + 16), axis=1)
    km = KMeans(3, n_init=5, random_state=0).fit(rest)
    return {"win": w, "edges": edges, "r2": r2, "rates": rates, "rcenters": km.cluster_centers_}


def encode_counts(obs: np.ndarray, alphabet: dict) -> np.ndarray:
    n = len(obs)
    counts = np.zeros((n, 24), dtype=np.int32)
    w = alphabet["win"]
    edge = alphabet["edges"]
    rc = alphabet["rcenters"]
    for i in range(n):
        risk = obs[i, :, w:w + 16].max(1)
        b = np.digitize(risk, edge)
        rest = np.delete(obs[i], np.arange(w, w + 16), axis=1)
        z = ((rest[:, None, :] - rc[None, :, :]) ** 2).sum(2).argmin(1)
        counts[i] = np.bincount(b * 3 + z, minlength=24)
    return counts


def true_costs(artifact_dir: Path, name: str) -> np.ndarray:
    with np.load(artifact_dir / "pools.npz", allow_pickle=False) as z:
        return np.concatenate([z[f"{name}#t{k}|co"] for k in range(3)])


def run(args):
    rng = np.random.default_rng(45121115)
    task_dir = args.checkpoint_dir
    artifact_dir = args.artifact_dir
    pilot = collect_pilot(args.task, task_dir, 50, args.seed)
    # These evaluation episodes are separate from all pilot episodes and seeds.
    b_obs, b_cost = collect_evaluation(args.task, task_dir, 5.0, args.eval_n, args.seed + 1000)
    a_obs, a_cost = collect_evaluation(args.task, task_dir, 0.1, args.eval_n, args.seed + 2000)
    if (artifact_dir / "pools.npz").exists():
        jb_ref = true_costs(artifact_dir, "base")
        ja_ref = true_costs(artifact_dir, "lam0.1")
        truth_source = "independent released 300-episode truth pools"
    else:
        # For a newly trained holdout seed, use the independent fresh evaluation
        # rollouts as a descriptive reference; do not treat this as coverage evidence.
        jb_ref, ja_ref = b_cost, a_cost
        truth_source = "fresh evaluation pool also used for certificate inputs; descriptive only"
    ref = fit_alphabet(pilot, np.arange(50), np.arange(50))

    rows = []
    for total in SIZES:
        n0 = total // 2
        n1 = total - n0
        reps = 20
        for rep in range(reps):
            ids0 = rng.integers(0, 50, n0)
            ids1 = rng.integers(0, 50, n1)
            alpha = fit_alphabet(pilot, ids0, ids1)
            ca = encode_counts(a_obs, alpha)
            cb = encode_counts(b_obs, alpha)
            for threshold in THRESHOLDS:
                rises = np.diff(alpha["rates"])
                onset = [int(i) for i, rise in enumerate(rises) if rise >= threshold]
                eta = float(sum(max(0.0, rises[i]) for i in range(7) if i not in onset) + 0.05)
                upper = ks_certificate_eb(ca, cb, b_cost, T, np.arange(24) // 3, onset=onset, eta=eta)["ks_eb_bound"]
                rows.append({
                    "pilot_episodes_total": total,
                    "pilot_episodes_per_condition": [n0, n1],
                    "bootstrap_replicate": rep,
                    "threshold": threshold,
                    "window": alpha["win"],
                    "window_matches_full_100_episode_pilot": alpha["win"] == ref["win"],
                    "edge_mae_vs_full_pilot": float(np.mean(np.abs(alpha["edges"] - ref["edges"]))),
                    "risk_level_rates": alpha["rates"].tolist(),
                    "r2": alpha["r2"],
                    "onset": onset,
                    "eta": eta,
                    "U_EB": float(upper),
                    "J_A_reference": float(ja_ref.mean()),
                    "J_B_reference": float(jb_ref.mean()),
                    "point_undercoverage": float(upper) < float(ja_ref.mean()),
                })
        print(f"pilot total={total} complete ({reps} bootstrap fits)", flush=True)

    json.dump({
        "task": args.task,
        "pilot_seed": args.seed,
        "evaluation_seeds": [args.seed + 1000, args.seed + 2000],
        "evaluation_episodes_per_policy": args.eval_n,
        "bootstrap_replicates_per_size": 20,
        "pilot_sizes_total_episodes": list(SIZES),
        "thresholds": list(THRESHOLDS),
        "full_pilot_reference": {"window": ref["win"], "edges": ref["edges"].tolist(), "r2": ref["r2"], "risk_level_rates": ref["rates"].tolist()},
        "J_A_reference_se": float(ja_ref.std(ddof=1) / np.sqrt(len(ja_ref))),
        "J_B_reference_se": float(jb_ref.std(ddof=1) / np.sqrt(len(jb_ref))),
        "reference_source": truth_source,
        "resampling_note": "Pilot uncertainty is estimated by episode-level bootstrap from 50 fresh episodes in each pilot condition; the 100-episode reference pilot is not an independent validation set. Evaluation rollouts are fresh and independent of the pilot.",
        "results": rows,
    }, args.out.open("w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--task", default="SafetyCarGoal1-v0")
    p.add_argument("--seed", type=int, default=2026111501)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
