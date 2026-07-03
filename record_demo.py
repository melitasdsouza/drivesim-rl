"""
Load a trained PPO model and record an MP4 of it driving, for your
portfolio / README demo.

Usage:
    python record_demo.py --model ppo_driver.zip --out demo.mp4 --episodes 1
"""

import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO

from car_env import CarRacingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_driver.zip")
    parser.add_argument("--out", type=str, default="demo.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = CarRacingEnv(render_mode="rgb_array", randomize_track_each_episode=True)

    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            frames.append(env.render())
        print(f"Episode {ep}: reward={total_reward:.1f}, lap_completed={info.get('lap_completed')}")

    env.close()
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved video to {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
