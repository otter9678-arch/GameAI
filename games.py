"""
games.py — Game registry for GameAI.

Built-in profiles plus user-registered games persisted to games.json
(same directory as this file).

A profile tells GameAI how to drive and score a game:
  window_title    substring matched case-insensitively against window titles
  keys            action list (see gameai._key_for for accepted tokens)
  motion_gain     reward gain for pixel motion (encourages exploration)
  damage_region   normalized (l, t, r, b) crop scanned for red health/damage UI
  damage_penalty  reward subtracted per unit of red ratio in damage_region
  hold_seconds    how long each key is held per step
  episode_seconds episode length in wall-clock seconds

LEGAL: only register games whose terms of service allow bots. Both built-ins
are single-player titles with no anti-cheat and no competitive online play.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(HERE, "games.json")


@dataclass
class GameProfile:
    name: str
    window_title: str
    keys: List[str] = field(default_factory=lambda: ["w", "a", "s", "d"])
    aliases: List[str] = field(default_factory=list)
    motion_gain: float = 8.0
    damage_region: Optional[Tuple[float, float, float, float]] = None
    damage_penalty: float = 1.0
    hold_seconds: float = 0.08
    frame_stack: int = 4
    frame_scale: Tuple[int, int] = (84, 84)
    episode_seconds: int = 60
    notes: str = ""


BUILTINS: Dict[str, GameProfile] = {
    p.name.lower(): p
    for p in [
        GameProfile(
            name="cyberpunk2077",
            window_title="Cyberpunk 2077",
            aliases=["cyberpunk", "cp2077", "cp77"],
            keys=["w", "a", "s", "d", "space", "c", "shift", "click"],
            damage_region=(0.30, 0.86, 0.60, 0.94),  # health bar, bottom center
            episode_seconds=90,
            notes="Single-player, no anti-cheat. Train in a quiet district (empty "
                  "badlands). Pause menus confuse the vision — start training from "
                  "active gameplay.",
        ),
        GameProfile(
            name="starfield",
            window_title="Starfield",
            aliases=["sf"],
            keys=["w", "a", "s", "d", "space", "shift", "e", "click"],
            damage_region=(0.02, 0.88, 0.28, 0.98),  # health, bottom left
            episode_seconds=90,
            notes="Single-player, officially supports mods. Train outdoors away from "
                  "menus and dialogue. Jetpack (space) is only meaningful after "
                  "unlocking it.",
        ),
        GameProfile(
            name="pulsar_lost_colony",
            window_title="PULSAR: Lost Colony",
            aliases=["pulsar", "plc"],
            keys=["w", "a", "s", "d", "space", "shift", "e", "click"],
            damage_region=(0.02, 0.86, 0.30, 0.98),  # health HUD, bottom left
            episode_seconds=120,
            notes="Unity ship sim, moddable via PulsarModLoader (CapBot 2.0 target "
                  "game). LEGAL: solo/offline sessions ONLY — PULSAR has online "
                  "co-op; never run the agent in a crew with other human players. "
                  "Health HUD region is approximate — verify/adjust with "
                  "--damage-region after checking the in-game HUD layout. Start "
                  "training from active ship gameplay, not the station or menus.",
        ),
    ]
}


def generic_profile(window_title: str) -> GameProfile:
    """Build a working profile for ANY single-player game from just its
    window title. Uses a broad common action space (WASD + space + shift +
    E + click) and pure screen-based reward (motion + damage avoidance),
    so nothing game-specific is required."""
    return GameProfile(
        name=window_title,
        window_title=window_title,
        aliases=[],
        keys=["w", "a", "s", "d", "space", "shift", "e", "click"],
        motion_gain=8.0,
        damage_region=None,
        damage_penalty=1.0,
        hold_seconds=0.08,
        frame_stack=4,
        frame_scale=(84, 84),
        episode_seconds=90,
        notes="Generic auto-generated profile for an unregistered single-player "
              "game. Register it (gameai.py register ...) to tune keys, damage "
              "region and rewards.",
    )


def _load_registry() -> Dict[str, GameProfile]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k.lower(): GameProfile(**v) for k, v in raw.items()}


def _save_registry(profiles: Dict[str, GameProfile]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in profiles.items()}, f, indent=2)


def list_games() -> Dict[str, GameProfile]:
    games = dict(BUILTINS)
    games.update(_load_registry())
    return games


def resolve(name: str) -> GameProfile:
    """Find a game by name or alias (case-insensitive)."""
    q = name.strip().lower()
    games = list_games()
    if q in games:
        return games[q]
    for p in games.values():
        if q in [a.lower() for a in p.aliases]:
            return p
        if q in p.name.lower():
            return p
    available = ", ".join(sorted(games))
    raise SystemExit(f"Unknown game '{name}'. Registered games: {available}")


def register(profile: GameProfile, *, force: bool = False) -> None:
    key = profile.name.lower()
    if key in BUILTINS and not force:
        raise SystemExit(f"'{profile.name}' is a built-in game; cannot override it.")
    reg = _load_registry()
    if key in reg and not force:
        raise SystemExit(f"'{profile.name}' is already registered; pass --force to overwrite.")
    reg[key] = profile
    _save_registry(reg)
    print(f"[GameAI] Registered '{profile.name}' -> {REGISTRY_PATH}")


def unregister(name: str) -> None:
    key = name.strip().lower()
    reg = _load_registry()
    if key not in reg:
        raise SystemExit(f"'{name}' is not a registered game (built-ins cannot be removed).")
    del reg[key]
    _save_registry(reg)
    print(f"[GameAI] Unregistered '{name}'")