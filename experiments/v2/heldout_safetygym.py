"""Frozen-rule evaluation on a newly trained Safety-Gymnasium seed.

The pilot-only selection rule is fixed before candidate rollouts. Collection is
parallel across CPU workers with disjoint episode seeds. Early terminations are
padded with an explicit absorbing symbol at the lowest risk level.
"""
from __future__ import annotations
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import safetygym2 as sg
from cert import ks_certificate_eb
from pilot_sensitivity import T, QS, fit_alphabet


def _pilot_shard(task):
    env_name, checkpoint_dir, n, seed, sigma = task
    import safety_gymnasium
    import safetygym2 as worker_sg
    worker_sg.ENVNAME, worker_sg.DIR = env_name, checkpoint_dir
    policy = worker_sg.Policy(5.0, sigma)
    env = safety_gymnasium.make(env_name)
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        obs_ep, cost_ep = [], []
        for t in range(T):
            obs = np.asarray(obs, dtype=np.float32)
            action = policy.act(obs, rng)
            next_obs, _, cost, term, trunc, _ = env.step(action)
            if t % 5 == 0:
                obs_ep.append(obs)
                cost_ep.append(float(cost))
            obs = next_obs
            if term or trunc:
                break
        episodes.append((np.asarray(obs_ep), np.asarray(cost_ep)))
    env.close()
    return episodes


def _evaluation_shard(task):
    env_name, checkpoint_dir, lam, n, seed = task
    import safety_gymnasium
    import safetygym2 as worker_sg
    worker_sg.ENVNAME, worker_sg.DIR = env_name, checkpoint_dir
    policy = worker_sg.Policy(lam, 0.0)
    env = safety_gymnasium.make(env_name)
    rng = np.random.default_rng(seed)
    obs_all = None
    costs_all = np.zeros((n, T), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    for i in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        if obs_all is None:
            obs_all = np.full((n, T, np.asarray(obs).size), np.nan, dtype=np.float32)
        for t in range(T):
            obs = np.asarray(obs, dtype=np.float32)
            obs_all[i, t] = obs
            action = policy.act(obs, rng)
            obs, _, cost, term, trunc, _ = env.step(action)
            costs_all[i, t] = float(cost)
            lengths[i] = t + 1
            if term or trunc:
                break
    env.close()
    return obs_all, costs_all.sum(1), lengths


def collect_truth(env_name: str, checkpoint_dir: Path, lam: float, n: int, seed: int):
    import safety_gymnasium
    sg.ENVNAME, sg.DIR = env_name, str(checkpoint_dir)
    policy = sg.Policy(lam, 0.0)
    env = safety_gymnasium.make(env_name)
    rng = np.random.default_rng(seed)
    costs, lengths = np.zeros(n), np.zeros(n, dtype=np.int32)
    for i in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        total = 0.0
        for t in range(T):
            obs = np.asarray(obs, dtype=np.float32)
            action = policy.act(obs, rng)
            obs, _, cost, term, trunc, _ = env.step(action)
            total += float(cost)
            lengths[i] = t + 1
            if term or trunc:
                break
        costs[i] = total
    env.close()
    return costs, lengths


def _truth_shard(task):
    env_name, checkpoint_dir, lam, n, seed = task
    return collect_truth(env_name, Path(checkpoint_dir), lam, n, seed)


def _chunks(total, workers=10):
    base, extra = divmod(total, workers)
    return [base + (i < extra) for i in range(workers) if base + (i < extra) > 0]


def _parallel_map(fn, jobs):
    with ProcessPoolExecutor(max_workers=min(10, len(jobs))) as pool:
        return list(pool.map(fn, jobs))


def collect_pilot_parallel(env_name, checkpoint_dir, n_each, seed):
    sizes = _chunks(n_each, workers=4)
    jobs = []
    for sigma in (0.0, 1.5):
        offset = 0
        for size in sizes:
            jobs.append((env_name, str(checkpoint_dir), size,
                         seed + (100 if sigma else 0) + offset, sigma))
            offset += size
    shards = _parallel_map(_pilot_shard, jobs)
    n_shards = len(sizes)
    eps0 = [episode for chunk in shards[:n_shards] for episode in chunk]
    eps1 = [episode for chunk in shards[n_shards:] for episode in chunk]
    return ([x[0] for x in eps0], [x[1] for x in eps0],
            [x[0] for x in eps1], [x[1] for x in eps1])


def fit_pilot_alphabet(pilot, ids0, ids1):
    o0, c0, o1, c1 = pilot
    obs = np.concatenate([o0[i] for i in ids0] + [o1[i] for i in ids1])[None, :, :]
    costs = np.concatenate([c0[i] for i in ids0] + [c1[i] for i in ids1])[None, :]
    return fit_alphabet((obs, costs, obs, costs), np.array([0]), np.array([0]))


def collect_evaluation_parallel(env_name, checkpoint_dir, lam, n, seed):
    jobs = [(env_name, str(checkpoint_dir), lam, size, seed + i * 10000)
            for i, size in enumerate(_chunks(n))]
    shards = _parallel_map(_evaluation_shard, jobs)
    return tuple(np.concatenate([x[k] for x in shards]) for k in range(3))


def collect_truth_parallel(env_name, checkpoint_dir, lam, n, seed):
    jobs = [(env_name, str(checkpoint_dir), lam, size, seed + i * 10000)
            for i, size in enumerate(_chunks(n))]
    shards = _parallel_map(_truth_shard, jobs)
    return tuple(np.concatenate([x[k] for x in shards]) for k in range(2))


def encode_counts(obs: np.ndarray, alphabet: dict) -> np.ndarray:
    """Encode 24 ordinary cells plus a lowest-risk absorbing symbol."""
    counts = np.zeros((len(obs), 25), dtype=np.int32)
    w, edges, centers = alphabet["win"], alphabet["edges"], alphabet["rcenters"]
    for i in range(len(obs)):
        terminal = np.isnan(obs[i]).any(axis=1)
        counts[i, 24] = int(terminal.sum())
        valid = ~terminal
        risk = obs[i, valid, w:w + 16].max(1)
        level = np.digitize(risk, edges)
        rest = np.delete(obs[i, valid], np.arange(w, w + 16), axis=1)
        cluster = ((rest[:, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)
        ids = level * 3 + cluster
        counts[i, :24] = np.bincount(ids, minlength=24)
    return counts


def _pack_pilot_episodes(episodes):
    n = len(episodes)
    dim = episodes[0][0].shape[1]
    obs = np.full((n, T // 5, dim), np.nan, dtype=np.float32)
    cost = np.full((n, T // 5), np.nan, dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    for i, (o, c) in enumerate(episodes):
        lengths[i] = len(c)
        obs[i, :len(o)] = o
        cost[i, :len(c)] = c
    return obs, cost, lengths


def cost_bound(alpha, a_obs, b_obs, b_cost, threshold):
    a, b = encode_counts(a_obs, alpha), encode_counts(b_obs, alpha)
    rises = np.diff(alpha["rates"])
    onset = [int(i) for i, rise in enumerate(rises) if rise >= threshold]
    eta = float(sum(max(0.0, rises[i]) for i in range(7) if i not in onset) + 0.05)
    levels = np.concatenate([np.arange(24) // 3, np.array([0])])
    result = ks_certificate_eb(a, b, b_cost, T, levels, onset=onset, eta=eta)
    return {"U_EB": float(result["ks_eb_bound"]), "onset": onset, "eta": eta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--task", default="SafetyPointGoal1-v0")
    p.add_argument("--seed", type=int, required=True, help="fresh collection seed")
    p.add_argument("--pilot-n-each", type=int, default=50)
    p.add_argument("--certificate-n", type=int, default=400)
    p.add_argument("--truth-n", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw-out", type=Path, required=True)
    args = p.parse_args()

    # Frozen v1 rule: the alpha fit uses baseline pilot episodes only.
    pilot = collect_pilot_parallel(args.task, args.checkpoint_dir, args.pilot_n_each, args.seed)
    pilot0_obs, pilot0_cost, pilot0_len = _pack_pilot_episodes(list(zip(pilot[0], pilot[1])))
    pilot1_obs, pilot1_cost, pilot1_len = _pack_pilot_episodes(list(zip(pilot[2], pilot[3])))
    n0 = n1 = args.pilot_n_each
    alpha = fit_pilot_alphabet(pilot, np.arange(n0), np.arange(n1))
    b_obs, b_cert_cost, b_cert_lengths = collect_evaluation_parallel(args.task, args.checkpoint_dir, 5.0, args.certificate_n, args.seed + 1000)
    a_obs, a_cert_cost, a_cert_lengths = collect_evaluation_parallel(args.task, args.checkpoint_dir, 0.1, args.certificate_n, args.seed + 2000)
    b_truth, b_truth_lengths = collect_truth_parallel(args.task, args.checkpoint_dir, 5.0, args.truth_n, args.seed + 3000)
    a_truth, a_truth_lengths = collect_truth_parallel(args.task, args.checkpoint_dir, 0.1, args.truth_n, args.seed + 4000)

    thresholds = (0.0, 0.025, 0.05, 0.10, 0.20)
    threshold_rows = [dict(rise_threshold=x, **cost_bound(alpha, a_obs, b_obs, b_cert_cost, x)) for x in thresholds]
    rng = np.random.default_rng(args.seed + 5000)
    pilot_sensitivity = []
    for total in (10, 25, 50, 100):
        n_each = total // 2
        for rep in range(20):
            ids0 = rng.integers(0, n0, n_each)
            ids1 = rng.integers(0, n1, total - n_each)
            fit = fit_pilot_alphabet(pilot, ids0, ids1)
            pilot_sensitivity.append({
                "pilot_total": total, "replicate": rep,
                "window": int(fit["win"]),
                "edge_mae_vs_full": float(np.mean(np.abs(fit["edges"] - alpha["edges"]))),
                **cost_bound(fit, a_obs, b_obs, b_cert_cost, 0.05),
            })

    chosen = cost_bound(alpha, a_obs, b_obs, b_cert_cost, 0.05)
    ja, jb = float(a_truth.mean()), float(b_truth.mean())
    se_a = float(a_truth.std(ddof=1) / np.sqrt(args.truth_n))
    se_b = float(b_truth.std(ddof=1) / np.sqrt(args.truth_n))
    boot_a = rng.choice(a_truth, size=(10000, args.truth_n), replace=True).mean(axis=1)
    boot_b = rng.choice(b_truth, size=(10000, args.truth_n), replace=True).mean(axis=1)
    ci_a = np.quantile(boot_a, [0.025, 0.975]).tolist()
    ci_b = np.quantile(boot_b, [0.025, 0.975]).tolist()
    summary = {
        "task": args.task, "training_seed": 3, "collection_seed": args.seed,
        "selection_rule": "Frozen baseline-pilot window scan, seven quantile edges, onset threshold 0.05 and residual-rise eta+0.05.",
        "pilot_episodes_per_condition": args.pilot_n_each,
        "pilot_early_terminations": {
            "sigma0": int((pilot0_len < T // 5).sum()),
            "sigma1.5": int((pilot1_len < T // 5).sum()),
        },
        "alphabet": {"window": int(alpha["win"]), "edges": alpha["edges"].tolist(), "pilot_r2": float(alpha["r2"]), "rates": alpha["rates"].tolist()},
        "certificate_episodes_per_policy": args.certificate_n,
        "truth_episodes_per_policy": args.truth_n,
        "certificate_and_truth_seeds_disjoint": True,
        "terminal_padding": "Early-terminated episodes use a dedicated absorbing symbol at the lowest risk level and zero cost.",
        "episode_length_summary": {
            "baseline_certificate_early_terminations": int((b_cert_lengths < T).sum()),
            "candidate_certificate_early_terminations": int((a_cert_lengths < T).sum()),
            "baseline_truth_early_terminations": int((b_truth_lengths < T).sum()),
            "candidate_truth_early_terminations": int((a_truth_lengths < T).sum()),
        },
        "baseline_truth_mean": jb, "baseline_truth_se": se_b,
        "candidate_truth_mean": ja, "candidate_truth_se": se_a,
        "baseline_truth_95ci_bootstrap": ci_b,
        "candidate_truth_95ci_bootstrap": ci_a,
        "candidate_truth_ci_within_bound": ci_a[1] <= chosen["U_EB"],
        "point_coverage": ja <= chosen["U_EB"],
        "selected_certificate": chosen,
        "threshold_sensitivity": threshold_rows,
        "pilot_size_bootstrap": pilot_sensitivity,
        "pilot_size_bootstrap_replicates_per_size": 20,
        "limitation": "One fresh training seed and one lambda=0.1 candidate; coverage is descriptive, not a repeated-sampling validation of the procedure.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.raw_out,
        a_certificate_counts=encode_counts(a_obs, alpha), b_certificate_counts=encode_counts(b_obs, alpha),
        b_certificate_costs=b_cert_cost, a_certificate_costs=a_cert_cost,
        b_truth_costs=b_truth, a_truth_costs=a_truth,
        b_certificate_lengths=b_cert_lengths, a_certificate_lengths=a_cert_lengths,
        b_truth_lengths=b_truth_lengths, a_truth_lengths=a_truth_lengths,
        pilot_sigma0_observations=pilot0_obs, pilot_sigma0_costs=pilot0_cost,
        pilot_sigma0_sample_lengths=pilot0_len,
        pilot_sigma15_observations=pilot1_obs, pilot_sigma15_costs=pilot1_cost,
        pilot_sigma15_sample_lengths=pilot1_len,
        risk_edges=alpha["edges"], risk_window=np.asarray([alpha["win"]]))
    keep = ("task", "training_seed", "pilot_episodes_per_condition", "certificate_episodes_per_policy", "truth_episodes_per_policy", "baseline_truth_mean", "candidate_truth_mean", "candidate_truth_95ci_bootstrap", "selected_certificate", "point_coverage", "candidate_truth_ci_within_bound")
    print(json.dumps({k: summary[k] for k in keep}, indent=2))


if __name__ == "__main__":
    main()
