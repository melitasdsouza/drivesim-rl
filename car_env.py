"""
CarRacingLite: a from-scratch 2D driving environment.

- Procedurally generated closed-loop track (randomized radius per angle)
- Simple car kinematics (position, heading, speed) with steering + throttle
- Lidar-style raycast sensors measuring distance to track walls
- Reward = progress along track centerline, penalized for going off-track
- Pygame rendering (works headless via SDL "dummy" video driver)

This is intentionally dependency-light (numpy + pygame + gymnasium) so it's
easy to read, modify, and explain in a writeup.
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Allow headless rendering (no display) for training on servers/containers.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402


def generate_track(n_points=64, base_radius=220, radius_noise=60, width=40, seed=None):
    """Procedurally generate a closed-loop track as a centerline + width.

    Returns:
        centerline: (n_points, 2) array of (x, y) points around origin
        width: scalar track width (constant along the loop, kept simple)
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Smooth random radius variation via a small number of sine harmonics
    # (cheap substitute for Perlin noise -> avoids sharp, undrivable turns).
    radius = np.full(n_points, base_radius, dtype=float)
    for k in range(1, 4):
        amp = radius_noise / k
        phase = rng.uniform(0, 2 * np.pi)
        radius += amp * np.sin(k * angles + phase)

    centerline = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return centerline, width


class CarRacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, n_rays=9, ray_length=200, seed=None,
                 randomize_track_each_episode=True):
        super().__init__()
        self.render_mode = render_mode
        self.n_rays = n_rays
        self.ray_length = ray_length
        self.randomize_track_each_episode = randomize_track_each_episode
        self._rng = np.random.default_rng(seed)

        # Observation: n_rays lidar distances (normalized 0-1) + speed + heading error to centerline
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_rays + 2,), dtype=np.float32
        )
        # Action: [steering (-1..1), throttle (-1..1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.width_px, self.height_px = 800, 800
        self.screen = None
        self.clock = None

        self.max_speed = 6.0
        self.accel = 0.25
        self.friction = 0.05
        self.max_steer_rate = 0.08  # radians per step

        self.track_width = 40
        self._build_track()

    # ---------- track geometry ----------

    def _build_track(self):
        seed = None if self.randomize_track_each_episode else 0
        self.centerline, self.track_width = generate_track(
            n_points=96, base_radius=220, radius_noise=60, width=40, seed=seed
        )
        # Precompute segment vectors for nearest-point / boundary queries
        self._seg_next = np.roll(self.centerline, -1, axis=0)
        self._seg_vec = self._seg_next - self.centerline
        self._seg_len = np.linalg.norm(self._seg_vec, axis=1)
        self.n_checkpoints = len(self.centerline)

    def _closest_segment(self, pos):
        """Return (index of closest centerline segment, signed lateral offset, progress along segment)."""
        p = np.asarray(pos)
        to_p = p[None, :] - self.centerline
        seg_unit = self._seg_vec / (self._seg_len[:, None] + 1e-8)
        proj = np.sum(to_p * seg_unit, axis=1)
        proj_clamped = np.clip(proj, 0, self._seg_len)
        closest_pts = self.centerline + seg_unit * proj_clamped[:, None]
        dists = np.linalg.norm(p[None, :] - closest_pts, axis=1)
        idx = int(np.argmin(dists))
        lateral = dists[idx]
        # sign: left/right of segment direction
        seg_dir = seg_unit[idx]
        normal = np.array([-seg_dir[1], seg_dir[0]])
        signed_lateral = np.dot(p - closest_pts[idx], normal)
        return idx, signed_lateral, lateral

    def _ray_hit_distance(self, origin, angle):
        """March a ray outward from origin and find distance to track boundary."""
        step = 4.0
        dist = 0.0
        direction = np.array([math.cos(angle), math.sin(angle)])
        while dist < self.ray_length:
            point = origin + direction * dist
            _, _, lateral = self._closest_segment(point)
            if lateral > self.track_width / 2:
                return dist
            dist += step
        return self.ray_length

    # ---------- gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_track_each_episode:
            self._build_track()

        start = self.centerline[0]
        nxt = self.centerline[1]
        heading = math.atan2(nxt[1] - start[1], nxt[0] - start[0])

        self.pos = start.copy()
        self.heading = heading
        self.speed = 0.0
        self.progress_idx = 0
        self.steps = 0
        self.max_steps = 1200
        self.cum_progress = 0.0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        steer, throttle = float(action[0]), float(action[1])
        steer = np.clip(steer, -1, 1)
        throttle = np.clip(throttle, -1, 1)

        self.heading += steer * self.max_steer_rate
        self.speed += throttle * self.accel
        self.speed -= np.sign(self.speed) * self.friction
        self.speed = np.clip(self.speed, -self.max_speed / 2, self.max_speed)

        self.pos = self.pos + self.speed * np.array(
            [math.cos(self.heading), math.sin(self.heading)]
        )

        idx, signed_lateral, lateral = self._closest_segment(self.pos)

        # progress reward: how far around the loop we've advanced since last step
        delta = idx - self.progress_idx
        if delta < -self.n_checkpoints / 2:
            delta += self.n_checkpoints  # wrapped around start/finish
        elif delta > self.n_checkpoints / 2:
            delta -= self.n_checkpoints  # went backwards across the seam
        self.progress_idx = idx
        self.cum_progress += delta

        off_track = lateral > self.track_width / 2
        reward = float(delta) * 1.0
        reward -= 0.01  # small time penalty to discourage idling
        terminated = False
        if off_track:
            reward -= 10.0
            terminated = True

        self.steps += 1
        truncated = self.steps >= self.max_steps
        lap_completed = self.cum_progress >= self.n_checkpoints
        if lap_completed:
            reward += 50.0
            terminated = True

        obs = self._get_obs()
        info = {"lateral_offset": lateral, "lap_completed": lap_completed}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ray_angles = np.linspace(-math.pi / 2, math.pi / 2, self.n_rays) + self.heading
        rays = np.array([self._ray_hit_distance(self.pos, a) for a in ray_angles])
        rays_norm = (rays / self.ray_length) * 2 - 1  # normalize to [-1, 1]

        _, signed_lateral, _ = self._closest_segment(self.pos)
        lateral_norm = np.clip(signed_lateral / (self.track_width / 2), -1, 1)
        speed_norm = np.clip(self.speed / self.max_speed, -1, 1)

        obs = np.concatenate([rays_norm, [speed_norm, lateral_norm]]).astype(np.float32)
        return obs

    # ---------- rendering ----------

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                os.environ.pop("SDL_VIDEODRIVER", None)
                self.screen = pygame.display.set_mode((self.width_px, self.height_px))
            else:
                self.screen = pygame.Surface((self.width_px, self.height_px))
            self.clock = pygame.time.Clock()

        surf = self.screen
        surf.fill((30, 30, 35))

        cx, cy = self.width_px / 2, self.height_px / 2

        def to_screen(p):
            return int(cx + p[0]), int(cy + p[1])

        # draw track as a thick closed line along the centerline (robust to
        # concave curves, unlike filling inner/outer polygons directly)
        pts = [to_screen(p) for p in self.centerline]
        pygame.draw.lines(surf, (60, 60, 70), True, pts, int(self.track_width))
        for p in pts:
            pygame.draw.circle(surf, (60, 60, 70), p, int(self.track_width / 2))
        pygame.draw.lines(surf, (110, 110, 120), True, pts, 1)

        # draw sensor rays
        ray_angles = np.linspace(-math.pi / 2, math.pi / 2, self.n_rays) + self.heading
        for a in ray_angles:
            d = self._ray_hit_distance(self.pos, a)
            end = self.pos + d * np.array([math.cos(a), math.sin(a)])
            pygame.draw.line(surf, (255, 200, 80), to_screen(self.pos), to_screen(end), 1)

        # draw car as a rotated triangle
        L = 12
        tip = self.pos + L * np.array([math.cos(self.heading), math.sin(self.heading)])
        left = self.pos + L * 0.6 * np.array(
            [math.cos(self.heading + 2.5), math.sin(self.heading + 2.5)]
        )
        right = self.pos + L * 0.6 * np.array(
            [math.cos(self.heading - 2.5), math.sin(self.heading - 2.5)]
        )
        pygame.draw.polygon(surf, (240, 60, 60), [to_screen(tip), to_screen(left), to_screen(right)])

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            arr = pygame.surfarray.array3d(surf)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
