"""Train one PPOLag policy on a Safety-Gymnasium task (used for the Safety-Gym stress test).

Usage: python train_ppolag.py ENV LAMBDA_INIT SEED STEPS OUT_DIR
"""
import os, sys, json
os.environ.pop("MUJOCO_GL", None)
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)
import omnisafe

env, lam, seed, steps, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
spe = 2000
agent = omnisafe.Agent("PPOLag", env, custom_cfgs={
    "seed": seed,
    "train_cfgs": {"total_steps": steps, "torch_threads": 2, "device": "cpu"},
    "logger_cfgs": {"log_dir": out, "save_model_freq": max(1, steps // spe // 5)},
    "algo_cfgs": {"steps_per_epoch": spe},
    "lagrange_cfgs": {"cost_limit": 25.0, "lagrangian_multiplier_init": lam},
})
agent.learn()
json.dump({"env": env, "lambda_init": lam, "seed": seed, "steps": steps}, open(os.path.join(out, "meta.json"), "w"))
