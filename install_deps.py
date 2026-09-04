"""
GameAI dependencies. Install into a venv (recommended) or globally.

NOTE: torch on Windows with CUDA (for the RTX 4090) needs the CUDA wheel index:
  pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
import subprocess, sys

PACKAGES = [
    "torch",              # PyTorch (install CUDA wheel separately for GPU)
    "torchvision",
    "numpy",
    "mss",                # fast screen capture
    "pynput",             # keyboard/mouse injection
    "opencv-python",      # image resize / preprocessing
    "stable-baselines3",  # production-grade PPO (better than the built-in REINFORCE)
    "gymnasium",          # env API
]

def main():
    for pkg in PACKAGES:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    print("\nDone. To enable GPU on the RTX 4090:")
    print("  pip uninstall torch torchvision -y")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

if __name__ == "__main__":
    main()