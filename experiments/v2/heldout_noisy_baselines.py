"""Evaluate frozen-policy action-noise candidates on the untouched PointGoal1 seed 3."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from cert import ks_certificate_eb
from heldout_safetygym import (
    T, collect_evaluation_parallel, collect_truth_parallel, encode_counts,
    fit_pilot_alphabet,
)

SIGMAS = (0.1, 0.3, 0.6, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--base-summary", type=Path, default=Path("experiments/v2/revision/heldout-pg1-seed3.json"))
    p.add_argument("--base-raw", type=Path, default=Path("experiments/v2/revision/heldout-pg1-seed3.npz"))
    p.add_argument("--out", type=Path, default=Path("experiments/v2/revision/heldout-pg1-seed3-noisy.json"))
    p.add_argument("--raw-out", type=Path, default=Path("experiments/v2/revision/heldout-pg1-seed3-noisy.npz"))
    args = p.parse_args()

    base = json.loads(args.base_summary.read_text())
    with np.load(args.base_raw) as z:
        # Refit from the archived pilot episodes using the exact frozen v1 rule.
        pilot = (z["pilot_sigma0_observations"], z["pilot_sigma0_costs"],
                 z["pilot_sigma15_observations"], z["pilot_sigma15_costs"])
        alpha = fit_pilot_alphabet(pilot, np.arange(50), np.arange(50))
        b_counts = z["b_certificate_counts"].copy()
        b_cost = z["b_certificate_costs"].copy()
        b_truth = z["b_truth_costs"].copy()
        b_lengths = z["b_certificate_lengths"].copy()
        b_truth_lengths = z["b_truth_lengths"].copy()
        original_a_counts = z["a_certificate_counts"].copy()
        original_a_truth = z["a_truth_costs"].copy()
    assert alpha["win"] == base["alphabet"]["window"]
    assert np.allclose(alpha["edges"], base["alphabet"]["edges"])
    onset = base["selected_certificate"]["onset"]
    eta = base["selected_certificate"]["eta"]
    levels = np.r_[np.arange(24) // 3, 0]
    seed = int(base["collection_seed"])
    rows, raw = [], {}
    original_all = ks_certificate_eb(original_a_counts, b_counts, b_cost, T, levels,
                                     onset=list(range(7)), eta=0.0)
    original_boot_a = np.asarray(base["candidate_truth_95ci_bootstrap"])
    original = {
        "candidate": "trained lambda=0.1 policy",
        "sigma": None,
        "certificate_episodes": len(original_a_counts),
        "truth_episodes": len(original_a_truth),
        "baseline_reference_mean_shared": float(b_truth.mean()),
        "baseline_reference_95ci_shared": base["baseline_truth_95ci_bootstrap"],
        "candidate_reference_mean": float(original_a_truth.mean()),
        "candidate_reference_se": float(original_a_truth.std(ddof=1) / np.sqrt(len(original_a_truth))),
        "candidate_reference_95ci_bootstrap": original_boot_a.tolist(),
        "excess_reference_mean": float(original_a_truth.mean() - b_truth.mean()),
        "pilot_selected_onset": onset,
        "pilot_selected_eta": eta,
        "pilot_selected_eb_upper_cost": float(base["selected_certificate"]["U_EB"]),
        "pilot_selected_upper_minus_candidate_ci_hi": float(base["selected_certificate"]["U_EB"] - original_boot_a[1]),
        "pilot_selected_covers_candidate_mean": float(original_a_truth.mean()) <= base["selected_certificate"]["U_EB"],
        "pilot_selected_covers_candidate_reference_ci": original_boot_a[1] <= base["selected_certificate"]["U_EB"],
        "all_threshold_onset": list(range(7)),
        "all_threshold_eta": 0.0,
        "all_threshold_eb_upper_cost": float(original_all["ks_eb_bound"]),
        "all_threshold_upper_minus_candidate_ci_hi": float(original_all["ks_eb_bound"] - original_boot_a[1]),
        "all_threshold_covers_candidate_mean": float(original_a_truth.mean()) <= original_all["ks_eb_bound"],
        "all_threshold_covers_candidate_reference_ci": original_boot_a[1] <= original_all["ks_eb_bound"],
        "certificate_early_terminations": base["episode_length_summary"]["candidate_certificate_early_terminations"],
        "truth_early_terminations": base["episode_length_summary"]["candidate_truth_early_terminations"],
        "baseline_pool_shared_across_candidates": True,
        "truth_seeds_disjoint_from_certificate": True,
    }
    rows.append(original)
    for j, sigma in enumerate(SIGMAS):
        cert_seed = seed + 10000 + j * 2000
        truth_seed = seed + 20000 + j * 2000
        a_obs, a_cert_cost, a_lengths = collect_evaluation_parallel(
            base["task"], args.checkpoint_dir, 5.0, 400, cert_seed, sigma=sigma)
        a_truth, a_truth_lengths = collect_truth_parallel(
            base["task"], args.checkpoint_dir, 5.0, 300, truth_seed, sigma=sigma)
        a_counts = encode_counts(a_obs, alpha)
        selected = ks_certificate_eb(a_counts, b_counts, b_cost, T, levels,
                                     onset=onset, eta=eta)
        all_thresholds = ks_certificate_eb(a_counts, b_counts, b_cost, T, levels,
                                           onset=list(range(7)), eta=0.0)
        rng = np.random.default_rng(seed + 30000 + j)
        boot_a = rng.choice(a_truth, size=(10000, len(a_truth)), replace=True).mean(1)
        ci_a = np.quantile(boot_a, [0.025, 0.975]).tolist()
        boot_b = rng.choice(b_truth, size=(10000, len(b_truth)), replace=True).mean(1)
        excess = boot_a - boot_b
        ci_excess = np.quantile(excess, [0.025, 0.975]).tolist()
        jb = float(b_truth.mean()); ja = float(a_truth.mean())
        row = {
            "candidate": "lambda=5 baseline with Gaussian action noise",
            "sigma": sigma,
            "certificate_episodes": len(a_cert_cost),
            "truth_episodes": len(a_truth),
            "certificate_seed_start": cert_seed,
            "truth_seed_start": truth_seed,
            "baseline_reference_mean_shared": jb,
            "baseline_reference_95ci_shared": base["baseline_truth_95ci_bootstrap"],
            "candidate_reference_mean": ja,
            "candidate_reference_se": float(a_truth.std(ddof=1) / np.sqrt(len(a_truth))),
            "candidate_reference_95ci_bootstrap": ci_a,
            "excess_reference_mean": ja - jb,
            "excess_reference_95ci_bootstrap": ci_excess,
            "pilot_selected_onset": onset,
            "pilot_selected_eta": eta,
            "pilot_selected_eb_upper_cost": float(selected["ks_eb_bound"]),
            "pilot_selected_upper_minus_candidate_ci_hi": float(selected["ks_eb_bound"] - ci_a[1]),
            "pilot_selected_covers_candidate_mean": ja <= selected["ks_eb_bound"],
            "pilot_selected_covers_candidate_reference_ci": ci_a[1] <= selected["ks_eb_bound"],
            "all_threshold_onset": list(range(7)),
            "all_threshold_eta": 0.0,
            "all_threshold_eb_upper_cost": float(all_thresholds["ks_eb_bound"]),
            "all_threshold_upper_minus_candidate_ci_hi": float(all_thresholds["ks_eb_bound"] - ci_a[1]),
            "all_threshold_covers_candidate_mean": ja <= all_thresholds["ks_eb_bound"],
            "all_threshold_covers_candidate_reference_ci": ci_a[1] <= all_thresholds["ks_eb_bound"],
            "certificate_early_terminations": int((a_lengths < T).sum()),
            "truth_early_terminations": int((a_truth_lengths < T).sum()),
            "baseline_pool_shared_across_candidates": True,
            "truth_seeds_disjoint_from_certificate": True,
        }
        rows.append(row)
        key = f"sigma_{sigma:g}"
        raw[f"{key}_certificate_counts"] = a_counts
        raw[f"{key}_certificate_costs"] = a_cert_cost
        raw[f"{key}_certificate_lengths"] = a_lengths
        raw[f"{key}_truth_costs"] = a_truth
        raw[f"{key}_truth_lengths"] = a_truth_lengths
        print(json.dumps({k: row[k] for k in (
            "sigma", "candidate_reference_mean", "candidate_reference_95ci_bootstrap",
            "excess_reference_mean", "excess_reference_95ci_bootstrap",
            "pilot_selected_eb_upper_cost", "all_threshold_eb_upper_cost")}), flush=True)

    out = {
        "task": base["task"], "training_seed": base["training_seed"],
        "selection_rule": base["selection_rule"], "collection_seed": seed,
        "candidate_policies": ["lambda=0.1 trained policy", "lambda=5 baseline with Gaussian action noise; no additional training"],
        "certificate_episodes_per_candidate": 400,
        "truth_episodes_per_candidate": 300,
        "baseline_reference_episodes_shared": len(b_truth),
        "baseline_reference_mean_shared": float(b_truth.mean()),
        "baseline_reference_95ci_shared": base["baseline_truth_95ci_bootstrap"],
        "baseline_certificate_episodes_shared": len(b_cost),
        "pilot_selected_bound_is_total_candidate_cost_upper": True,
        "all_threshold_bound_is_total_candidate_cost_upper": True,
        "all_threshold_rule": "Use all seven onset thresholds with eta=0; this requires only nondecreasing cost in the risk level and avoids pilot onset selection.",
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    np.savez_compressed(args.raw_out, **raw)


if __name__ == "__main__":
    main()
