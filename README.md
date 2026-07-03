# DriveSim: A From-Scratch 2D Autonomous Driving Simulator + RL Agent

A lightweight driving environment (custom physics, procedural tracks, lidar-style
sensors, and rendering — all built from scratch in numpy/pygame) paired with a
PPO agent trained to drive it.

**Result:** after only 50,000 training steps (a few minutes on CPU), the agent
already completes full laps on procedurally generated tracks it wasn't trained on.
See `demo_50k.mp4`.

## Why this project

Built to demonstrate the intersection of computer graphics (custom renderer,
sensor simulation, procedural generation) and reinforcement learning (reward
design, policy training, generalization) — the combination underlying tools
like NVIDIA's Isaac Gym, Tesla's simulation stack for autonomy, and general
RL research.

## What's in the box

| File | Purpose |
|---|---|
| `car_env.py` | The environment: procedural track generation, car kinematics, lidar raycasting sensors, and pygame rendering (works headless). |
| `train.py` | Trains a PPO agent (Stable-Baselines3) across 8 parallel environments. |
| `record_demo.py` | Loads a trained model and renders an MP4 of it driving. |
| `checkpoints/` | Saved model checkpoints every 10k steps (10k → 50k). |
| `ppo_driver_pretrained.zip` | The 50k-step checkpoint used for the demo video. |
| `demo_50k.mp4` | Recorded lap from the pretrained agent. |
| `tb_logs/` | TensorBoard training logs. |

## Quickstart

```bash
pip install gymnasium stable-baselines3 pygame numpy imageio imageio-ffmpeg tensorboard

# Train from scratch (increase --timesteps for a better driver; 200k-500k
# gives noticeably smoother, faster laps than the 50k checkpoint included here)
python train.py --timesteps 200000

# Watch training progress
tensorboard --logdir tb_logs

# Record a video of the trained agent
python record_demo.py --model ppo_driver.zip --out demo.mp4

# Or just use the included pretrained checkpoint directly:
python record_demo.py --model ppo_driver_pretrained.zip --out demo.mp4
```

## Design notes (for your writeup / interview talking points)

**Environment design.** The track is generated procedurally each episode by
perturbing a circle's radius with a few random sine harmonics — this keeps
turns smooth and drivable while still varying every episode, so the agent has
to learn to *drive*, not memorize one track layout.

**Observation space.** 9 lidar-style raycasts fanned across the car's forward
180°, plus current speed and signed lateral offset from the track centerline.
This mirrors how real AV stacks fuse range sensors with vehicle state — and
keeps the observation low-dimensional enough to train fast on CPU (no image
input needed, unlike pixel-based RL environments).

**Reward shaping — this is the part worth discussing in an interview.** The
reward is progress along the track's centerline (checkpoint index delta per
step), not raw distance traveled or speed. Early versions that rewarded speed
directly caused the agent to floor it into walls; rewarding raw position
caused it to farm reward by oscillating near the start line. Tying reward to
*monotonic progress along the track's checkpoint sequence*, plus a large
terminal penalty for leaving the track and a small per-step time penalty
(discourage stalling), fixed both failure modes. This is a textbook example
of reward hacking and why reward shaping — not just "add more training" — is
usually the actual bottleneck in applied RL.

**Generalization.** Because the track is re-randomized every episode, a high
reward can't come from memorizing a fixed path — the agent has to use its
sensors. That's the property to highlight if asked "how do you know it
learned to drive vs. memorized something."

## Suggested next steps (good "future work" section)

- Train longer (200k-500k+ steps) for smoother, faster driving.
- Add other vehicles / obstacles to avoid — turns this into a more genuine
  autonomy problem instead of pure lane-keeping.
- Swap the hand-rolled raycasting for a real 2D physics engine (Box2D via
  pymunk) to get proper collision response instead of episode termination.
- Try a recurrent policy (PPO + LSTM) and compare sample efficiency against
  the current stateless MLP — natural ablation to report.
- Port the renderer to a 3D engine (even a basic OpenGL/three.js scene) for
  a stronger graphics signal if targeting graphics-heavy roles.
