"""Small one-factor-at-a-time configurable PPO ablation runner."""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42,123,456]); parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--note", default="Change one documented YAML parameter per run and compare with benchmark.py")
    args = parser.parse_args(); print(args.note)
    for seed in args.seeds:
        subprocess.run([sys.executable,"scripts/train.py","--config",args.config,"--seed",str(seed),"--total-steps",str(args.total_steps)],check=True)

if __name__ == "__main__": main()
