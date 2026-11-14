# Occupancy-Based Certificates for the Relative Safety of RL Agents

Upper-bounds the expected episodic cost of a new agent `A` relative to a trusted baseline `B` from rollouts alone:

    J(A) <= J(B) + T * TV(d_A, d_B)                     (cost sufficiency; paper, Assumption 1)
    J(A) <= J(B) + T * max_{tau in T} [d_A(b>tau) - d_B(b>tau)]_+   (cost nondecreasing in a risk level b; Theorem 4)

with finite-sample versions (Hoeffding and empirical-Bernstein) that treat episodes as the independent units.
Paper source: `paper/main.tex` (+ `paper/refs.bib`); the Markdown package for publish.fun is in `publishfun/`.

## Layout

- `experiments/v2/cert.py` — certificates (TV, JSD, KS+, finite-sample slacks, empirical Bernstein).
- `experiments/v2/cartpole.py` — Constrained CartPole (`gen`, `analyse`); `is_comparison.py` — importance-sampling comparison.
- `experiments/v2/train_ppolag.py`, `safetygym2.py` — Safety-Gymnasium (`fit`, `onset`, `gen`, `analyse`).
- `experiments/v2/make_tables.py` — writes the LaTeX tables; `check_numbers.py` — re-derives every number quoted in the paper.
- `experiments/v2/summaries/` — summary JSONs behind the tables. `tests/test_cert.py` — unit tests.
- `archive/v1/` — withdrawn first draft; not part of the paper.
- `artifacts/` (when published) — rollout pools, alphabet files and trained checkpoints (about 15 MB).

## Reproduce

Python 3.10; versions in `experiments/v2/requirements.txt` (safety-gymnasium needs a `pygame>=2.5` override).

    python experiments/v2/cartpole.py gen data/cartpole && python experiments/v2/cartpole.py analyse data/cartpole
    python experiments/v2/is_comparison.py data/cartpole
    for s in 0 1 2; do for lam in 0.1 0.5 1.0 2.0 5.0; do
      python experiments/v2/train_ppolag.py SafetyPointGoal1-v0 $lam $s 100000 data/pg1s$s/lam$lam; done; done
    for lam in 0.1 0.5 1.0 2.0 5.0; do python experiments/v2/train_ppolag.py SafetyCarGoal1-v0 $lam 0 100000 data/cg1/lam$lam; done
    for d in "SafetyPointGoal1-v0 data/pg1s0" "SafetyPointGoal1-v0 data/pg1s1" "SafetyPointGoal1-v0 data/pg1s2" "SafetyCarGoal1-v0 data/cg1"; do
      for step in fit onset gen analyse; do python experiments/v2/safetygym2.py $step $d; done; done
    python experiments/v2/make_tables.py && python experiments/v2/check_numbers.py && pytest -q tests

## Artifacts

`artifacts/` holds the rollout pools, alphabet/onset files and the final PPO-Lagrangian checkpoints used in the paper
(`artifacts/pg1` = PointGoal1 seed 0, `pg1s1`, `pg1s2`, `cg1` = CarGoal1 seed 0, `cartpole`). To re-run the analyses without
re-training, copy them to `data/` (`pg1` -> `data/pg1s0`, others by name) and run the `analyse` steps above.
