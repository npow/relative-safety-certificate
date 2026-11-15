"""Reproduce targeted analyses requested in the first review round.

Usage:
  python revision_study.py --artifacts PATH heldout
  python revision_study.py --artifacts PATH threshold-sensitivity
  python revision_study.py --artifacts PATH cartpole-power
  python revision_study.py --artifacts PATH sufficiency-failure

The rollout pools are from the public v1 companion repository. This script
does not retrain policies or alter the original data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cert import certificate, ks_certificate_eb


RUNS = {
    "pg1_seed0_development": "pg1",
    "pg1_seed1_holdout": "pg1s1",
    "pg1_seed2_holdout": "pg1s2",
    "cg1_task_holdout": "cg1",
}
SUMMARY_FILES = {
    "pg1_seed0_development": "pointgoal1_seed0_summary2.json",
    "pg1_seed1_holdout": "pointgoal1_seed1_summary2.json",
    "pg1_seed2_holdout": "pointgoal1_seed2_summary2.json",
    "cg1_task_holdout": "cargoal1_seed0_summary2.json",
}
T = 1000
LEVELS = np.arange(24) // 3


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def heldout(artifacts: Path, out: Path) -> None:
    runs = {}
    all_rows = []
    for run_name, artifact_name in RUNS.items():
        if "development" in run_name:
            continue
        summary_path = Path(__file__).parent / "summaries" / SUMMARY_FILES[run_name]
        summary = json.loads(summary_path.read_text())
        rows = summary["rows"]["cost-relevant"]
        cell = []
        for r in rows:
            lower95 = float(r["J_true"] - 1.96 * r["J_se"])
            item = {
                "agent": r["agent"],
                "J_A_est": float(r["J_true"]),
                "J_A_se": float(r["J_se"]),
                "J_B_est": float(r["JB_true"]),
                "U_EB": float(r["sel_eb"]),
                "U_minus_JA": float(r["sel_eb"] - r["J_true"]),
                "U_over_JA": float(r["sel_eb"] / max(r["J_true"], 1e-9)),
                "ground_truth_lower_95": lower95,
                "clear_undercoverage_at_95": lower95 > float(r["sel_eb"]),
                "onset": list(r["onset"]),
                "eta": float(r["eta_bar"]),
            }
            cell.append(item)
            all_rows.append(item)
        runs[run_name] = cell
    n = len(all_rows)
    json.dump({
        "holdout_runs": list(runs),
        "cell_count": n,
        "finite_bound_point_estimate_coverage": sum(r["U_EB"] >= r["J_A_est"] for r in all_rows) / n,
        "clear_undercoverage_95pct_reference_ci": sum(r["clear_undercoverage_at_95"] for r in all_rows),
        "median_excess_cost_units": float(np.median([r["U_minus_JA"] for r in all_rows])),
        "median_bound_to_cost_ratio": float(np.median([r["U_over_JA"] for r in all_rows])),
        "runs": runs,
        "note": "Seed 0 is the development run. Seeds 1 and 2 and CarGoal1 are validation runs. Truth costs use independent 300-episode pools; certificate inputs use separate 400-episode pools.",
    }, out.open("w"), indent=2)


def threshold_sensitivity(artifacts: Path, out: Path) -> None:
    thresholds = (0.0, 0.025, 0.05, 0.10, 0.20)
    runs = {}
    for run_name, artifact_name in RUNS.items():
        if "development" in run_name:
            continue
        d = artifacts / artifact_name
        onset_data = json.loads((d / "onset.json").read_text())
        rates = np.asarray(onset_data["pilot_level_rates"], dtype=float)
        rises = np.diff(rates)
        pools = _load(d / "pools.npz")

        def get(name: str, split: str, block: int, key: str) -> np.ndarray:
            return pools[f"{name}#{split}{block}|{key}"]

        agent_names = ["lam0.1", "lam0.5", "lam1.0", "lam2.0"] + [f"basenoise{x}" for x in (0.1, 0.3, 0.6, 1.0)]
        setting_rows = []
        for threshold in thresholds:
            onset = [int(i) for i, r in enumerate(rises) if r >= threshold]
            eta = float(sum(max(0.0, rises[i]) for i in range(7) if i not in onset) + 0.05)
            cell_bounds = []
            for name in agent_names:
                a = np.concatenate([get(name, "c", b, "ch") for b in range(4)])
                b = np.concatenate([get("base", "c", k, "ch") for k in range(4)])
                bc = np.concatenate([get("base", "c", k, "co") for k in range(4)])
                tcost = np.concatenate([get(name, "t", k, "co") for k in range(3)])
                bound = ks_certificate_eb(a, b, bc, T, LEVELS, onset=onset, eta=eta)["ks_eb_bound"]
                mean = float(tcost.mean())
                se = float(tcost.std(ddof=1) / np.sqrt(len(tcost)))
                cell_bounds.append({
                    "agent": name, "U_EB": float(bound), "J_A_est": mean,
                    "J_A_se": se, "undercoverage_point": mean > bound,
                    "clear_undercoverage_95pct": mean - 1.96 * se > bound,
                })
            setting_rows.append({
                "rise_threshold": threshold, "onset": onset, "eta": eta,
                "median_U_EB": float(np.median([x["U_EB"] for x in cell_bounds])),
                "median_J_A": float(np.median([x["J_A_est"] for x in cell_bounds])),
                "point_undercoverage_cells": sum(x["undercoverage_point"] for x in cell_bounds),
                "clear_undercoverage_95pct_cells": sum(x["clear_undercoverage_95pct"] for x in cell_bounds),
                "cells": cell_bounds,
            })
        runs[run_name] = setting_rows
    json.dump({
        "runs": runs,
        "thresholds": list(thresholds),
        "note": "The risk alphabet and certificate pools are held fixed; only the pilot onset threshold and resulting eta are varied. Seed 0 is excluded from validation summaries.",
    }, out.open("w"), indent=2)


def cartpole_power(artifacts: Path, out: Path) -> None:
    """Bootstrap certification decisions from the released independent pools."""
    from cartpole import LEVELS as cp_levels
    rng = np.random.default_rng(4512026)
    d = artifacts / "cartpole"
    base = _load(d / "base.npz")
    base_truth = _load(d / "base@truth.npz")["costs"]
    jb = float(base_truth.mean())
    eps = ("eps0.02", "eps0.05", "eps0.1", "eps0.2", "eps0.3", "eps0.5")
    margins = (1.0, 5.0, 10.0)
    sample_sizes = (100, 500, 2000)
    repetitions = 3000
    results = []
    for agent in eps:
        a = _load(d / f"{agent}.npz")
        jt = float(_load(d / f"{agent}@truth.npz")["costs"].mean())
        for n in sample_sizes:
            bounds = []
            for _ in range(repetitions):
                ia = rng.integers(0, len(a["counts"]), n)
                ib = rng.integers(0, len(base["counts"]), n)
                cert = ks_certificate_eb(a["counts"][ia], base["counts"][ib], base["costs"][ib], 200, cp_levels)
                bounds.append(cert["ks_eb_bound"])
            for margin in margins:
                unsafe = jt > jb + margin
                p = float(np.mean(np.asarray(bounds) > jb + margin))
                results.append({
                    "agent": agent, "n_per_policy": n, "margin": margin,
                    "true_excess_cost": jt - jb,
                    "truly_unsafe": unsafe,
                    "probability_not_certified": p,
                    "probability_certified_safe": 1.0 - p,
                    "repetitions": repetitions,
                    "mc_se": float(np.sqrt(p * (1 - p) / repetitions)),
                })
    json.dump({
        "delta": 0.025, "confidence": 0.95,
        "decision_rule": "Not certified safe when the 95% EB upper bound exceeds J_B(reference) + margin. Reference J_A/J_B means use independent 10,000-episode pools. Resampling uses the separate 10,000-episode rollout pools.",
        "results": results,
    }, out.open("w"), indent=2)


def sufficiency_failure(out: Path) -> None:
    n, T = 400, 200
    counts_b = np.ones((n, 1), dtype=np.int32)
    counts_a = np.ones((n, 1), dtype=np.int32)
    c = certificate(counts_a, counts_b, np.zeros(n), T)
    json.dump({
        "construction": "One observed symbol z for every step under both policies. Baseline cost is always 0; candidate cost is always 1 per step but depends on an unobserved policy-specific latent variable.",
        "assumption_1_holds": False,
        "T": T, "n_per_policy": n, "K": 1,
        "J_B": 0.0, "J_A": float(T), "empirical_TV": 0.0,
        "plug_in_TV_upper": float(c["plugin_bound"]),
        "finite_sample_TV_upper": float(c["finite_sample_bound"]),
        "finite_sample_undercoverage": float(T) > float(c["finite_sample_bound"]),
        "interpretation": "This is a counterexample to applying the certificate without cost sufficiency, not a failure of its stated theorem.",
    }, out.open("w"), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("command", choices=("heldout", "threshold-sensitivity", "cartpole-power", "sufficiency-failure"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    target = args.out or (Path(__file__).parent / "revision" / f"{args.command}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "heldout":
        heldout(args.artifacts, target)
    elif args.command == "threshold-sensitivity":
        threshold_sensitivity(args.artifacts, target)
    elif args.command == "cartpole-power":
        cartpole_power(args.artifacts, target)
    else:
        sufficiency_failure(target)
    print(target)


if __name__ == "__main__":
    main()
