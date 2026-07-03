"""
Train a PPO agent to drive the CarRacingEnv.

Usage:
    python train.py --timesteps 200000

For a quick smoke test (a few minutes, won't drive well but proves the
pipeline works end-to-end):
    python train.py --timesteps 20000
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from car_env import CarRacingEnv


def make_env():
    return CarRacingEnv(render_mode="rgb_array", randomize_track_each_episode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--out", type=str, default="ppo_driver")
    parser.add_argument("--logdir", type=str, default="tb_logs")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path="checkpoints",
        name_prefix="ppo_driver",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True)
    model.save(args.out)
    print(f"Saved trained model to {args.out}.zip")
    print(f"View training curves with: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
