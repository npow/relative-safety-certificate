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

## First-round revision analyses

The requested review analyses and their machine-readable outputs are in `experiments/v2/revision/`. With the original artifact pools in `data/`:

    python experiments/v2/revision_study.py --artifacts artifacts heldout --out experiments/v2/revision/heldout.json
    python experiments/v2/revision_study.py --artifacts artifacts threshold-sensitivity --out experiments/v2/revision/threshold-sensitivity.json
    python experiments/v2/revision_study.py --artifacts artifacts cartpole-power --out experiments/v2/revision/cartpole-power.json
    python experiments/v2/revision_study.py --artifacts artifacts sufficiency-failure --out experiments/v2/revision/sufficiency-failure.json
    python experiments/v2/is_eb.py --artifacts artifacts/cartpole --out experiments/v2/revision/is-eb.json

`pilot_sensitivity.py` reruns the pilot-size bootstrap using a released checkpoint. `heldout_safetygym.py` evaluates the frozen pilot-selection rule on the additional PointGoal1 seed-3 checkpoints; its pilot, certificate and independent truth samples are kept separate. These sensitivity and held-out results are described as diagnostics and a case study, not as a broad coverage validation. Run `python experiments/v2/check_numbers.py` to audit the reported values.

## Revision release artifacts

The public `artifacts/pg1s3/` directory contains the seed-3 baseline (`lam5.0`) and candidate (`lam0.1`) checkpoints. The accompanying `experiments/v2/revision/heldout-pg1-seed3.npz` preserves the pilot, certificate and independent truth episodes; `heldout-pg1-seed3.json` records the frozen selection and summary. Reproduce the held-out case with:

    python experiments/v2/heldout_safetygym.py --checkpoint-dir artifacts/pg1s3 --seed 2026111515 --out experiments/v2/revision/heldout-pg1-seed3.json --raw-out experiments/v2/revision/heldout-pg1-seed3.npz


## Round-two seed-3 analysis

`experiments/v2/heldout_noisy_baselines.py` reuses the archived seed-3 pilot and shared baseline pools, then evaluates the four Gaussian-noise candidates. Each candidate gets a separate certificate pool (400 episodes) and independent truth pool (300 episodes). It reports both the frozen pilot-selected bound and the robust all-threshold bound. Reproduce with:

    python experiments/v2/heldout_noisy_baselines.py --checkpoint-dir artifacts/pg1s3

