"""Variance-adaptive high-confidence OIS comparison on CartPole.

Runs 2,000 behavior-policy episodes once, computes ordinary importance-sampling
returns for three target policies, and applies an empirical-Bernstein upper
bound using the known return range T * max trajectory likelihood ratio.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import is_comparison as isv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--delta", type=float, default=0.05)
    args = p.parse_args()

    eps_b, eps_as = 0.3, (0.05, 0.1, 0.2)
    G, log_weights = isv.run_baseline(eps_b, eps_as, args.n, seed=9112026)
    results = []
    for e in eps_as:
        lw = log_weights[e]
        weights = np.exp(lw)
        x = weights * G
        ratio_max = (1 - e / 2) / (1 - eps_b / 2)
        max_return = isv.T * math.exp(isv.T * math.log(ratio_max))
        upper, radius, variance = isv.empirical_bernstein_upper(x, max_return, delta=args.delta)
        with np.load(args.artifacts / f"eps{e}@truth.npz", allow_pickle=False) as truth:
            true_cost = float(truth["costs"].mean())
        results.append({
            "eps_A": e,
            "eps_B": eps_b,
            "n_behavior_episodes": args.n,
            "delta": args.delta,
            "true_cost_reference": true_cost,
            "ois_estimate": float(x.mean()),
            "ois_empirical_variance": variance,
            "ois_eb_radius": radius,
            "ois_eb_upper": upper,
            "max_trajectory_weight_log10": isv.T * math.log10(ratio_max),
            "max_weighted_return": max_return,
            "covers_reference": upper >= true_cost,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "method": "Unnormalized per-trajectory importance sampling with the Maurer-Pontil empirical-Bernstein upper bound.",
        "confidence": 1 - args.delta,
        "behavior_seed": 9112026,
        "n": args.n,
        "results": results,
    }, args.out.open("w"), indent=2)
    for row in results:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
