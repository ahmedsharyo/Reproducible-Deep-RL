from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("inputs", nargs="+", help="run dirs or benchmark dirs")
    parser.add_argument("--output", default="results/learning_curve.png"); args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for item in args.inputs:
        path = Path(item)
        file = path / "evaluation_curves.csv" if (path / "evaluation_curves.csv").exists() else path / "history.csv"
        data = pd.read_csv(file).dropna(subset=["evaluation_return"])
        if "seed" in data:
            grouped = data.groupby("step")["evaluation_return"].agg(["mean", "std"])
            x, mean, std = grouped.index, grouped["mean"], grouped["std"].fillna(0)
            label = path.name
        else:
            x, mean, std, label = data.step, data.evaluation_return, data.get("evaluation_std", 0), path.name
        ax.plot(x, mean, label=label); ax.fill_between(x, mean-std, mean+std, alpha=.2)
    ax.set(xlabel="Environment steps", ylabel="Evaluation return"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); fig.savefig(target); plt.close(fig)
    print(target)

if __name__ == "__main__": main()
