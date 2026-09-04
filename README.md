# GameAI — Self-Learning Game-Playing AI

Python deep-learning framework that captures the screen, learns to play, and
improves on its own. Runs on the RTX 4090.

## ⚠️ Legal — read first

Only use this on games that **allow bots**. Most multiplayer games with
anti-cheat (Destiny 2, Valorant, Fortnite, Overwatch...) **ban accounts** for
automation — using it there violates the ToS and can violate computer-fraud
law. Intended for: single-player games, sandbox games, games with explicit
bot policies, retro emulators, your own games.

## Setup

```bash
python install_deps.py
# For GPU on the RTX 4090:
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
python gameai.py "My Game" w,a,s,d,space 20
```

That: launches the agent on the game titled "My Game", uses WASD+space as the
action space, trains for 20 episodes. Model persists to `gameai_model.pt` —
the agent resumes learning each session.

## Architecture

- `PolicyNetwork` — CNN + actor-critic head (PyTorch)
- `GameEnv` — mss screen capture + pynput input injection, gym-style `step()`
- `Agent` — sampling, save/load model
- `Trainer` — REINFORCE loop (swap to `stable-baselines3.PPO` for production)
- Reward shaping is game-specific — hook into the game's state or use the
  score pixels as reward.

## Extending

- Better reward: use OCR or pixel-diff heuristics per game
- Vision: add a second model head for object detection
- Multi-agent: run several instances with different seeds
- Real production training: wrap `GameEnv` in `stable_baselines3.PPO`

## Honest limitations

- Screen-only learning is slow (months of training for complex games)
- No game-state API access means the agent learns from pixels alone
- Games with heavy anti-cheat will detect + ban — don't use there
- For serious RL, use SB3 PPO + vectorized envs, not this skeleton