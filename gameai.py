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
  * Games that explicitly allow bots
  * Your own homebrew games

It is NOT intended for online multiplayer with anti-cheat. You are
responsible for checking each game's ToS before using it.

Architecture:
  games.py       -> game registry (built-ins + user-registered + generic)
  GameEnv        -> mss screen capture + pynput input injection, gym-like API
  PolicyNetwork  -> CNN actor-critic over stacked grayscale frames (PyTorch)
  Trainer        -> endless self-learning loop; resumes + keeps best model
  CLI            -> list / register / unregister / train / play / selftest

Any window title can be used directly: unrecognized names get a generic
single-player profile automatically (see games.generic_profile).

Usage:
  python gameai.py list
  python gameai.py register <name> --window "Window Title" [--keys w,a,s,d,space]
  python gameai.py unregister <name>
  python gameai.py train <game> [--episodes N] [--steps N] [--dry-run]
  python gameai.py play  <game> [--seconds N] [--dry-run]
  python gameai.py selftest
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

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
        "  python install_deps.py\n"
        f"Details: {e}"
    )

import games as gamereg

# ---------- Model ----------

class PolicyNetwork(nn.Module):
    """CNN + actor-critic head over stacked grayscale frames."""

    def __init__(self, in_channels: int = 4, num_actions: int = 8):
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

class GameEnv:
    """Screen capture + input injection, gym-like API.

    Rewards (fully screen-derived, so ANY single-player game works):
      * motion  — positive when the scene changes (exploration/progress)
      * damage  — negative when the configured UI region turns red
      * time    — small negative per step to discourage idling

    dry_run=True makes step() skip input injection (capture-only mode).
    """

    def __init__(self, profile: gamereg.GameProfile, dry_run: bool = False):
        self.cfg = profile
        self.dry_run = dry_run
        self.kb = KbController()
        self.mouse = MouseController()
        self.sct = mss.MSS()
        self._monitor = self.sct.monitors[1]  # primary monitor
        self._last_gray: Optional[np.ndarray] = None
        self._last_red: float = 0.0
        self._win_rect: Optional[Tuple[int, int, int, int]] = None

    # ----- window targeting (Win32, no external deps) -----

    def resolve_window(self) -> Tuple[int, int, int, int]:
        """Locate the game window by title substring and return (l, t, w, h).
        Falls back to the full primary monitor if not found."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found: List[Tuple[int, int, int, int]] = []
        title_q = self.cfg.window_title.lower()

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _enum_cb(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if not length:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if title_q not in buf.value.lower():
                return True
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                w, h = rect.right - rect.left, rect.bottom - rect.top
                if w > 200 and h > 200:
                    found.append((rect.left, rect.top, w, h))
                    return False
            return True

        user32.EnumWindows(_enum_cb, 0)

        if found:
            self._win_rect = found[0]
            print(f"[GameAI] Window '{self.cfg.window_title}' at {self._win_rect}")
        else:
            m = self._monitor
            print(f"[GameAI] Window '{self.cfg.window_title}' not found; "
                  f"using full primary screen")
            self._win_rect = (m["left"], m["top"], m["width"], m["height"])
        return self._win_rect

    # ----- actions -----

    def _press(self, key: str) -> None:
        try:
            special = {
                "space": Key.space, "shift": Key.shift, "ctrl": Key.ctrl,
                "alt": Key.alt, "tab": Key.tab, "esc": Key.esc,
                "enter": Key.enter, "capslock": Key.caps_lock,
            }
            if key == "click":
                self.mouse.click(Button.left)
            elif key == "rclick":
                self.mouse.click(Button.right)
            elif key in special:
                self.kb.press(special[key]); self.kb.release(special[key])
            else:
                self.kb.press(key); self.kb.release(key)
        except Exception:
            pass

    def step(self, action_idx: int):
        """Apply action, capture next frame, return (gray, reward, done, info)."""
        keys = self.cfg.keys
        if not self.dry_run and action_idx < len(keys):
            self._press(keys[action_idx])
            time.sleep(self.cfg.hold_seconds)

        frame = self._get_frame()
        reward = self._reward(frame)
        return frame, reward, False, {}

    # ----- vision + reward (color-aware damage detection) -----

    def _get_frame(self) -> np.ndarray:
        l, t, w, h = self._win_rect or self.resolve_window()
        shot = self.sct.grab({"top": t, "left": l, "width": w, "height": h})
        arr = np.array(shot, dtype=np.float32)  # BGRA
        # Damage detection needs color; do it at native resolution.
        self._last_red = self._red_ratio(arr)
        # Grayscale for the network (coefficients match BGRA channel order).
        gray = np.dot(arr[..., :3], [0.114, 0.587, 0.299])
        import cv2
        gray = cv2.resize(gray, tuple(self.cfg.frame_scale),
                          interpolation=cv2.INTER_AREA)
        return gray / 255.0

    def _red_ratio(self, arr: np.ndarray) -> float:
        reg = self.cfg.damage_region
        if reg is None:
            return 0.0
        l, t, r, b = reg
        h, w = arr.shape[:2]
        crop = arr[int(t * h):int(b * h), int(l * w):int(r * w)]
        if crop.size == 0:
            return 0.0
        blue, green, red = crop[..., 0], crop[..., 1], crop[..., 2]
        return float(np.mean((red > 120) & (red > green * 1.5) & (red > blue * 1.5)))

    def _reward(self, gray: np.ndarray) -> float:
        r = -0.01  # time cost
        if self._last_gray is not None:
            motion = float(np.mean(np.abs(gray - self._last_gray)))
            r += self.cfg.motion_gain * motion
        r -= self.cfg.damage_penalty * self._last_red
        self._last_gray = gray
        return r

    def close(self):
        try:
            self.sct.close()
        except Exception:
            pass


class FrameStack:
    """Stacks the last N frames as input channels: (N, H, W)."""

    def __init__(self, n: int):
        self.n = n
        self.buf: List[np.ndarray] = []

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.buf = [frame] * self.n
        return np.stack(self.buf, axis=0)

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.buf.append(frame)
        self.buf = self.buf[-self.n:]
        return np.stack(self.buf, axis=0)


# ---------- Agent ----------

class Agent:
    def __init__(self, num_actions: int, frame_stack: int = 4, lr: float = 3e-4,
                 device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = PolicyNetwork(in_channels=frame_stack, num_actions=num_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.num_actions = num_actions

    def choose_action(self, stacked, deterministic: bool = False):
        x = torch.tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        probs = torch.softmax(logits, dim=-1)[0]
        if deterministic:
            action = int(torch.argmax(probs).item())
        else:
            action = int(torch.multinomial(probs, 1).item())
        # logp keeps its graph so the policy-gradient update can backprop.
        logp = torch.log(probs[action] + 1e-8)
        return action, logp, value.detach().item()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.net.load_state_dict(
                torch.load(path, map_location=self.device))
            print(f"[GameAI] Loaded model from {path} (device: {self.device})")


# ---------- Trainer (endless self-learning) ----------

class Trainer:
    """Self-learning loop: trains forever (or for --episodes), resuming from
    the saved model, and keeps the best checkpoint by per-step average reward."""

    def __init__(self, env, agent: Agent, save_path: str):
        self.env, self.agent = env, agent
        self.save_path = save_path
        self.best_path = save_path.replace(".pt", "_best.pt")

    def train(self, episodes: Optional[int] = None, steps_per_episode: int = 600,
              dry_run: bool = False, log_every: int = 25):
        self.agent.load(self.save_path)
        stack = FrameStack(self.env.cfg.frame_stack)
        ep, best_avg = 0, -float("inf")
        while episodes is None or ep < episodes:
            ep += 1
            stacked = stack.reset(self.env._get_frame())
            deadline = time.time() + self.env.cfg.episode_seconds
            logps, rewards = [], []
            total_reward, steps = 0.0, 0
            while steps < steps_per_episode and time.time() < deadline:
                action, logp, _ = self.agent.choose_action(stacked)
                frame, reward, _, _ = self.env.step(action)
                stacked = stack.push(frame)
                logps.append(logp)
                rewards.append(reward)
                total_reward += reward
                steps += 1
                if not dry_run and steps % log_every == 0:
                    print(f"  ep{ep} step{steps}: r={reward:+.3f} total={total_reward:.2f}")

            avg = total_reward / max(steps, 1)
            self._update(logps, rewards, dry_run)
            if avg > best_avg:
                best_avg = avg
                if not dry_run:
                    self.agent.save(self.best_path)
            if not dry_run:
                self.agent.save(self.save_path)
            print(f"[GameAI] Episode {ep}: steps={steps} reward={total_reward:.2f} "
                  f"avg={avg:.3f} best_avg={best_avg:.3f}")

    def _update(self, logps, rewards, dry_run: bool):
        """REINFORCE with discounted returns + advantage normalization."""
        if dry_run or not logps:
            return
        n = len(rewards)
        returns = [sum(rewards[i:]) * (0.99 ** i) for i in range(n)]
        logp_t = torch.stack(logps)
        R = torch.tensor(returns, dtype=torch.float32,
                         device=logp_t.device)  # match logp device (cuda/cpu)
        R = (R - R.mean()) / (R.std() + 1e-8)
        loss = -(logp_t * R).sum()
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()


# ---------- play (inference-only) ----------

def play(profile: gamereg.GameProfile, seconds: int = 30, dry_run: bool = False):
    env = GameEnv(profile, dry_run=dry_run)
    agent = Agent(num_actions=len(profile.keys), frame_stack=profile.frame_stack)
    agent.load(_model_path(profile.name))
    stack = FrameStack(profile.frame_stack)
    stacked = stack.reset(env._get_frame())
    print(f"[GameAI] Playing {profile.name} for {seconds}s "
          f"({'DRY RUN — no input' if dry_run else 'LIVE INPUT'})")
    deadline = time.time() + seconds
    while time.time() < deadline:
        action, _, _ = agent.choose_action(stacked, deterministic=True)
        label = profile.keys[action] if action < len(profile.keys) else "?"
        if dry_run:
            print(f"  would press: {label}")
            time.sleep(0.2)
            stacked = stack.push(env._get_frame())
        else:
            print(f"  action: {label}")
            frame, _, _, _ = env.step(action)
            stacked = stack.push(frame)
    env.close()


# ---------- CLI ----------

def _model_path(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
    return os.path.join(gamereg.HERE, f"model_{safe}.pt")


def _resolve_profile(name: str) -> gamereg.GameProfile:
    if gamereg.is_denied(name):
        raise SystemExit(
            f"REFUSED: '{name}' is an online multiplayer game with anti-cheat. "
            "GameAI does not support Destiny 2, Valorant, Fortnite, or Overwatch "
            "— automation there violates ToS and can violate computer-fraud law.")
    try:
        return gamereg.resolve(name)
    except SystemExit:
        # Not registered — treat the string as a window title and build a
        # generic single-player profile on the fly.
        print(f"[GameAI] '{name}' not registered; using generic single-player profile")
        return gamereg.generic_profile(name)


def cmd_list(args):
    table = gamereg.list_games()
    print(f"{'NAME':<24} {'WINDOW TITLE':<30} ACTIONS")
    for _, p in sorted(table.items()):
        print(f"{p.name:<24} {p.window_title:<30} {','.join(p.keys)}")


def cmd_register(args):
    prof = gamereg.GameProfile(
        name=args.name,
        window_title=args.window or args.name,
        keys=[k.strip() for k in args.keys.split(",")] if args.keys else
             ["w", "a", "s", "d", "space", "shift", "e", "click"],
        aliases=[a.strip() for a in args.alias.split(",")] if args.alias else [],
        motion_gain=args.motion_gain,
        damage_region=tuple(args.damage_region) if args.damage_region else None,
        damage_penalty=args.damage_penalty,
        episode_seconds=args.episode_seconds,
    )
    gamereg.register(prof, force=args.force)


def cmd_unregister(args):
    gamereg.unregister(args.name)


def cmd_train(args):
    profile = _resolve_profile(args.game)
    print(f"[GameAI] Training on '{profile.name}' "
          f"({'DRY RUN — no input' if args.dry_run else 'LIVE'}) — "
          f"keys: {','.join(profile.keys)}")
    print("[GameAI] Put the game window in active gameplay (not a menu). "
          "Training starts in 5s...")
    if not args.dry_run:
        time.sleep(5)
    env = GameEnv(profile, dry_run=args.dry_run)
    agent = Agent(num_actions=len(profile.keys), frame_stack=profile.frame_stack)
    trainer = Trainer(env, agent, _model_path(profile.name))
    try:
        trainer.train(episodes=args.episodes, steps_per_episode=args.steps,
                      dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n[GameAI] Interrupted — models saved so far are kept.")
    finally:
        env.close()


def cmd_play(args):
    profile = _resolve_profile(args.game)
    play(profile, seconds=args.seconds, dry_run=args.dry_run)


# ---------- controls (study the game's own on-screen key hints) ----------

_OCR_PS = os.path.join(gamereg.HERE, "ocr.ps1")  # vendored Windows OCR helper
_KNOWN_KEYS = {"W", "A", "S", "D", "E", "F", "Z", "Q", "R", "C", "X",
               "SPACE", "TAB", "SHIFT", "LEFTSHIFT", "CTRL", "ALT",
               "MOUSE1", "MOUSE2", "MOUSE4", "MOUSE5", "WHEELUP", "WHEELDOWN"}


def _ocr_lines(path: str) -> List[str]:
    import subprocess
    if not os.path.exists(_OCR_PS):
        raise SystemExit(f"OCR helper not found: {_OCR_PS}")
    out = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", _OCR_PS, path],
        capture_output=True, text=True, timeout=120).stdout
    lines = []
    for ln in out.splitlines():
        if "|" in ln:
            lines.append(ln.partition("|")[2].strip())
    return [l for l in lines if l]


def cmd_controls(args):
    """Capture the game window, OCR its on-screen control hints, and compare
    them with the profile's action space. Read-only: sends no input."""
    import cv2
    profile = _resolve_profile(args.game)
    env = GameEnv(profile, dry_run=True)
    try:
        l, t, w, h = env.resolve_window()
        shot = env.sct.grab({"top": t, "left": l, "width": w, "height": h})
        arr = np.array(shot)[:, :, :3]  # BGR

        stem = os.path.join(gamereg.HERE,
                            f"controls_{_model_path(profile.name)[_model_path(profile.name).find('model_') + 6:-3]}")
        full_png = stem + "_full.png"
        hint_png = stem + "_hints.png"
        cv2.imwrite(full_png, arr)
        # PULSAR-style key-hint strip lives in the bottom-right corner.
        hint = arr[int(h * 0.55):, int(w * 0.55):]
        hint = cv2.resize(hint, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(hint_png, hint)
        print(f"[GameAI] Saved captures: {full_png} , {hint_png}")

        seen = {}
        for src in (full_png, hint_png):
            for line in _ocr_lines(src):
                tok = line.split()
                if len(tok) >= 2 and tok[0].upper() in _KNOWN_KEYS:
                    seen[tok[0].upper()] = " ".join(tok[1:])

        if not seen:
            print("[GameAI] No control hints detected on screen "
                  "(hints usually show during gameplay, not menus).")
            return

        norm = {"SPACE": "space", "LEFTSHIFT": "shift", "SHIFT": "shift",
                "CTRL": "ctrl", "ALT": "alt", "TAB": "tab",
                "MOUSE1": "click", "MOUSE2": "rclick"}
        profile_keys = {k.lower() for k in profile.keys}
        print(f"\n[GameAI] On-screen controls for '{profile.name}':")
        for k, action in sorted(seen.items()):
            mapped = norm.get(k, k.lower())
            status = ("in action space" if mapped in profile_keys
                      else "NOT in action space"
                      + (" (mouse side-button unsupported)" if k.startswith("MOUSE") and k not in norm else ""))
            print(f"  {k:<10} {action:<30} -> {status}")

        missing = sorted({norm.get(k, k.lower()) for k in seen
                          if norm.get(k, k.lower()) not in profile_keys
                          and not k.startswith("MOUSE")})
        if missing:
            print(f"\n[GameAI] Suggested keys to add: {','.join(missing)}")
            print(f"[GameAI] Apply with: python gameai.py register {profile.name} "
                  f"--window \"{profile.window_title}\" "
                  f"--keys {','.join(profile.keys)},{','.join(missing)} --force "
                  f"(built-ins: this exact command; --force overrides the builtin)")
    finally:
        env.close()


def cmd_selftest(args):
    """End-to-end offline test: fake env, tiny net, no input, no files."""
    import numpy as np

    class FakeEnv:
        cfg = gamereg.GameProfile(name="selftest", window_title="selftest",
                                  keys=["w", "a", "s", "d"], frame_stack=2,
                                  frame_scale=(84, 84))

        def _get_frame(self):
            return np.random.rand(84, 84).astype(np.float32)

        def step(self, a):
            return self._get_frame(), float(a) * 0.1, False, {}

        def close(self):
            pass

    trainer = Trainer(FakeEnv(), Agent(num_actions=4, frame_stack=2),
                      os.path.join(gamereg.HERE, "_selftest_model.pt"))
    trainer.train(episodes=2, steps_per_episode=30, dry_run=True)
    print("[GameAI] selftest OK (no input sent, no files written)")


def main():
    ap = argparse.ArgumentParser(description="GameAI — self-learning game AI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list registered games")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("register", help="register a game")
    p.add_argument("name")
    p.add_argument("--window", help="window title substring (default: name)")
    p.add_argument("--keys", help="comma-separated action keys")
    p.add_argument("--alias", help="comma-separated aliases")
    p.add_argument("--motion-gain", type=float, default=8.0)
    p.add_argument("--damage-region", nargs=4, type=float, metavar=("L", "T", "R", "B"),
                   help="normalized (left top right bottom) health-bar region")
    p.add_argument("--damage-penalty", type=float, default=1.0)
    p.add_argument("--episode-seconds", type=int, default=90)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("unregister", help="remove a registered game")
    p.add_argument("name")
    p.set_defaults(func=cmd_unregister)

    p = sub.add_parser("train", help="self-learning training loop")
    p.add_argument("game", help="game name, alias, or window title")
    p.add_argument("--episodes", type=int, default=None,
                   help="episodes to run (default: endless until Ctrl+C)")
    p.add_argument("--steps", type=int, default=600, help="max steps per episode")
    p.add_argument("--dry-run", action="store_true",
                   help="capture + decide but send NO input")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("play", help="run the trained model (inference only)")
    p.add_argument("game", help="game name, alias, or window title")
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("controls", help="study on-screen control hints (read-only)")
    p.add_argument("game", help="game name, alias, or window title")
    p.set_defaults(func=cmd_controls)

    p = sub.add_parser("selftest", help="offline self-test (no input sent)")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()