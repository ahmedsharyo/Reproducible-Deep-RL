from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run one configured experiment for each independent seed.")
    parser.add_argument("--config", required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output", default="results/benchmark"); parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        command = [sys.executable, "scripts/train.py", "--config", args.config, "--seed", str(seed), "--output", str(output)]
        if args.total_steps is not None: command += ["--total-steps", str(args.total_steps)]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        result = json.loads(completed.stdout.strip().splitlines()[-1]); runs.append(result)
    # Preserve exact paths. Aggregation plots can consume each history file.
    (output / "runs.json").write_text(json.dumps({"config": args.config, "seeds": args.seeds, "runs": runs}, indent=2), encoding="utf-8")
    curves = []
    for run in runs:
        frame = pd.read_csv(Path(run["run"]) / "history.csv")
        evals = frame.dropna(subset=["evaluation_return"])[["step", "evaluation_return"]].copy()
        evals["seed"] = int(Path(run["run"]).name.split("_")[-2]); curves.append(evals)
    combined = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    combined.to_csv(output / "evaluation_curves.csv", index=False)
    if not combined.empty:
        aggregate = combined.groupby("step")["evaluation_return"].agg(["mean", "std", "count"]).reset_index()
        aggregate.to_csv(output / "evaluation_aggregate.csv", index=False)
        auc = float(aggregate["mean"].fillna(aggregate["mean"].mean()).sum())
        summary = {"mean_final_evaluation_return": float(aggregate.iloc[-1]["mean"]),
                   "std_final_evaluation_return": float(aggregate.iloc[-1]["std"] or 0.0),
                   "evaluation_auc_step_sum": auc, "seed_count": len(args.seeds),
                   "results_status": "measured; see per-seed run artifacts"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    else:
        print("Runs completed, but no evaluation records were produced.")

if __name__ == "__main__": main()
