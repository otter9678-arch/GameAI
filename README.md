# GameAI — Self-Learning Game-Playing AI

Python deep-learning framework that watches the screen, learns to play, and
improves on its own. Runs on the RTX 4090. Ships with profiles for
**Cyberpunk 2077** and **Starfield**, and can drive **any single-player game**
from just its window title.

## ⚠️ Legal — read first

Only use this on games that **allow bots**. Most multiplayer games with
anti-cheat (Destiny 2, Valorant, Fortnite, Overwatch...) **ban accounts** for
automation — using it there violates the ToS and can violate computer-fraud
law. Cyberpunk 2077 and Starfield are single-player titles with no anti-cheat,
so they are fine. Intended for: single-player games, sandbox games, games with
explicit bot policies, retro emulators, your own games.

## Setup

```bash
python install_deps.py
# For GPU on the RTX 4090 (cu124 index lacks Python 3.14 wheels; cu130 works):
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Usage

```bash
python gameai.py list                 # show registered games
python gameai.py selftest             # offline end-to-end test (no input sent)
python gameai.py train cyberpunk  --episodes 20 --dry-run   # watch it think, no input
python gameai.py train cyberpunk  --episodes 20             # live training
python gameai.py train starfield --episodes 20
python gameai.py play  cyberpunk --seconds 30                # inference only
```

**Any other single-player game works without registering** — just pass the
window title:

```bash
python gameai.py train "My Game" w,a,s,d,space 20     # legacy form also supported
python gameai.py train "Elden Ring" --episodes 10
```

Unregistered titles get an auto-generated generic profile (WASD + space +
shift + E + click, screen-based rewards).

To tune a game permanently:

```bash
python gameai.py register eldenring --window "ELDEN RING" \
    --keys w,a,s,d,space,shift,click \
    --damage-region 0.35 0.90 0.65 0.97 --force
python gameai.py unregister eldenring
```

## How it learns (self-learning loop)

- `train` runs **endlessly** until Ctrl+C unless `--episodes` is given.
- Each session resumes from `model_<game>.pt` and keeps improving.
- The best-performing checkpoint is kept separately as `model_<game>_best.pt`.
- Every episode: capture frame stack → pick action → apply key → measure
  reward → policy-gradient update → save.
- Rewards are fully screen-derived, so no game hooks are needed:
  - **motion** — scene change means something is happening (exploring)
  - **damage** — red pixels in the configured health-bar region penalize
  - **time** — small per-step cost discourages standing still

Use `--dry-run` first: the agent captures and decides but never sends input,
so you can sanity-check the reward stream in the log.

## Architecture

- `games.py` — registry: built-ins (Cyberpunk 2077, Starfield), user
  registrations (`games.json`), and a generic fallback for any window title
- `PolicyNetwork` — CNN + actor-critic head over 4 stacked grayscale frames
- `GameEnv` — Win32 window targeting + `mss` capture + `pynput` injection,
  gym-style `step()`; `dry_run` flag makes it capture-only
- `Agent` — sampling, save/load
- `Trainer` — REINFORCE with discounted returns + advantage normalization;
  endless self-learning with best-checkpoint tracking
- Damage detection is color-aware (RGB thresholds), unlike the naive version

## Extending

- Better reward: OCR score readout, pixel-diff region weighting per game
- Production RL: swap `Trainer` for `stable_baselines3.PPO` (already installed
  by `install_deps.py` — wrap `GameEnv` in a gymnasium adapter)
- Vision: add a second head for object detection
- Multi-agent: run several instances with different seeds

## Honest limitations

- Screen-only learning is slow — expect hours to days of training for
  meaningful behavior in complex 3D games
- No game-state API: the agent learns from pixels alone
- Reward shaping is heuristic (motion + damage); it learns "do interesting
  things and don't die", not "complete objectives" unless you add OCR/game
  hooks
- Games with heavy anti-cheat will detect + ban — don't use there
- For serious RL, use SB3 PPO + vectorized envs, not this loop