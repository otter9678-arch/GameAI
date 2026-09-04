"""
GameAI — Self-learning game-playing AI framework.

LEGAL NOTICE — READ BEFORE USING
================================
Whether an AI is allowed to play a game depends entirely on that game's
terms of service. Some games explicitly allow bots (single-player, sandbox,
open-source games). Most multiplayer games PROHIBIT automation of any kind
(Destiny 2, Valorant, Fortnite, Overwatch, etc.) — using an AI there will
get the account permanently banned and may violate computer-fraud law.

This framework is intended for:
  * Single-player / offline games
  * Games that explicitly allow bots (Minecraft servers with bot policy,
    open-ai gym environments, retro emulators, chess, racing sims with
    mod support)
  * Your own homebrew games

It is NOT intended for online multiplayer with anti-cheat. You are
responsible for checking each game's ToS before using it.

Architecture:
  ScreenWatcher  -> captures frames (mss, fast)
  InputInjector  -> sends keyboard/mouse (pynput)
  Agent          -> PyTorch policy network (CNN + actor-critic, PPO)
  Trainer        -> collects experience, learns constantly
  Save/Load      -> models persist between sessions
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import mss              # fast screen capture
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from pynput.keyboard import Controller as KbController, Key
    from pynput.mouse import Controller as MouseController, Button
except ImportError as e:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "  pip install torch torchvision mss numpy pynput opencv-python\n"
        f"Details: {e}"
    )


# ---------- Models ----------

class PolicyNetwork(nn.Module):
    """Simple CNN + actor-critic head. Takes a downscaled grayscale frame,
    outputs action logits + value estimate."""

    def __init__(self, in_channels: int = 1, num_actions: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        return self.policy(x), self.value(x)


# ---------- Environment ----------

@dataclass
class GameConfig:
    window_title: str
    capture_region: Optional[tuple] = None   # (left, top, width, height); None = full screen
    keys: list = None                        # action keys, e.g. ['w','a','s','d','space']
    mouse_click: bool = False
    frame_scale: tuple = (84, 84)            # downscaled input to network
    episode_seconds: int = 60


class GameEnv:
    """Wraps screen capture + input injection into a gym-like API."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.kb = KbController()
        self.mouse = MouseController()
        self.sct = mss.mss()

    def reset(self):
        """Start of an episode — no-op for now (games need UI automation)."""
        return self._get_frame()

    def step(self, action_idx: int):
        """Apply action, capture next frame, return (frame, reward, done)."""
        keys = self.cfg.keys or []
        if action_idx < len(keys):
            key = keys[action_idx]
            try:
                if key == "space": self.kb.press(Key.space); self.kb.release(Key.space)
                elif key == "click": self.mouse.click(Button.left)
                else: self.kb.press(key); self.kb.release(key)
            except Exception:
                pass

        frame = self._get_frame()
        # Reward shaping: override per-game. Default is small negative for time
        # (encourages the agent to finish fast) — real reward needs game hooks.
        reward = -0.01
        done = False
        return frame, reward, done, {}

    def _get_frame(self):
        region = self.cfg.capture_region
        if region is None:
            monitor = self.sct.monitors[1]  # primary monitor
            region = {"top": monitor["top"], "left": monitor["left"],
                      "width": monitor["width"], "height": monitor["height"]}
        shot = self.sct.grab(region)
        # Convert BGRA -> grayscale, downscale
        arr = np.array(shot, dtype=np.float32)
        gray = np.dot(arr[..., :3], [0.114, 0.587, 0.299])
        import cv2
        gray = cv2.resize(gray, self.cfg.frame_scale, interpolation=cv2.INTER_AREA)
        return gray / 255.0


# ---------- Agent + PPO-style training ----------

class Agent:
    def __init__(self, num_actions: int, lr: float = 3e-4):
        self.net = PolicyNetwork(num_actions=num_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.num_actions = num_actions

    def choose_action(self, frame):
        import torch
        x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits, value = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs.detach().numpy()[0], value.detach().item()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        if __import__("os").path.exists(path):
            self.net.load_state_dict(torch.load(path))
            print(f"[GameAI] Loaded model from {path}")


class Trainer:
    """Collects episodes and does simple policy-gradient updates.
    For serious results, switch to stable-baselines3 PPO — this is the skeleton."""

    def __init__(self, env: GameEnv, agent: Agent, save_path="gameai_model.pt"):
        self.env, self.agent, self.save_path = env, agent, save_path

    def train(self, episodes: int = 10, steps_per_episode: int = 300):
        import torch
        self.agent.load(self.save_path)
        for ep in range(episodes):
            frame = self.env.reset()
            total_reward, done, steps = 0.0, False, 0
            log_probs, values, rewards = [], [], []
            while not done and steps < steps_per_episode:
                action, logp, value = self.agent.choose_action(frame)
                frame, reward, done, _ = self.env.step(action)
                log_probs.append(logp[action])
                values.append(value)
                rewards.append(reward)
                total_reward += reward
                steps += 1
            # Simple REINFORCE update
            loss = -torch.sum(torch.stack(log_probs) *
                              torch.tensor(rewards, dtype=torch.float32))
            self.agent.optimizer.zero_grad()
            loss.backward()
            self.agent.optimizer.step()
            self.agent.save(self.save_path)
            print(f"[GameAI] Episode {ep + 1}: steps={steps} reward={total_reward:.2f} "
                  f"(model saved to {self.save_path})")


# ---------- Main ----------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gameai.py <window_title> <key1,key2,...> [episodes]")
        print("Example (single-player game using WASD + space):")
        print("  python gameai.py 'My Game' w,a,s,d,space 20")
        print("\nLEGAL: Only use on games that allow bots (check ToS).")
        sys.exit(1)

    title = sys.argv[1]
    keys = sys.argv[2].split(",")
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    cfg = GameConfig(window_title=title, keys=keys)
    env = GameEnv(cfg)
    agent = Agent(num_actions=len(keys))
    trainer = Trainer(env, agent)
    trainer.train(episodes=episodes)