# Reproducible Deep RL

Readable PyTorch implementations of DQN for `CartPole-v1` and PPO with a tanh-squashed Gaussian policy for `Pendulum-v1`. The project emphasizes inspectable Bellman targets, terminal versus time-limit handling, deterministic evaluation, and multi-seed experiment records.

## Install

Use Python 3.11 or newer. On macOS, a virtual environment and the platform-specific PyTorch wheel are recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

MPS is selected with `--device mps` when available. `auto` chooses MPS if present and otherwise CPU; CPU can be faster for these small MLPs. CUDA is optional and never required. MP4 recording uses Gymnasium's video wrapper and the imageio FFmpeg extra; install a system FFmpeg if your platform's codec backend needs it.

## Train and evaluate

```bash
python scripts/train.py --config configs/dqn_cartpole.yaml --seed 42
python scripts/train.py --config configs/ppo_pendulum.yaml --seed 42
python scripts/evaluate.py --checkpoint results/RUN/checkpoint.pt --algorithm ppo --environment Pendulum-v1
```

Runs save configuration metadata, JSON/CSV histories, and inference checkpoints under `results/`. PPO's default total budget is 500k steps; use `--total-steps` for a shorter smoke run. Results are not pre-populated and no performance claims are made until measured locally.

## Reproducible experiments

The default independent seeds are `[42, 123, 456]`:

```bash
python scripts/benchmark.py --config configs/dqn_cartpole.yaml --seeds 42 123 456
python scripts/benchmark.py --config configs/ppo_pendulum.yaml --seeds 42 123 456
python scripts/plot_results.py results/benchmark --output results/curve.png
python -m pytest
```

The benchmark records per-seed histories and aggregates evaluation return. For a fair CleanRL comparison, install the optional pinned baseline extra and run its official continuous PPO script with the same seeds, environment, step budget, and evaluation protocol. The CleanRL comparison adapter/table remains incomplete, so this repository currently does not claim measured CleanRL results.

Record a deterministic episode with:

```bash
python scripts/record_video.py --checkpoint results/RUN/checkpoint.pt --algorithm dqn --environment CartPole-v1
```

Use `--algorithm ppo --environment Pendulum-v1` for PPO. Video output goes to `videos/`.

## Architecture

`src/src.deep_rl/agents` contains algorithm updates; `networks` contains the Q and actor-critic MLPs; `buffers` contains replay and GAE utilities. `scripts/` provides CLI entry points. `docs/ppo_explained.md` and `docs/failure_analysis.md` explain the implementation and known diagnostic patterns.

## Method notes and limitations

DQN masks only true termination in the Bellman target. A time-limit truncation is stored with `terminated=False`, so it bootstraps. PPO computes the final-state value on each transition (including truncations) while stopping the GAE trace at episode boundaries. PPO actions are sampled in normalized `[-1,1]`, transformed by tanh, then scaled to environment bounds. The approximate entropy metric is the underlying Gaussian entropy, not exact squashed entropy.

Defaults are sensible starting points, not claimed optima. Full benchmark, CleanRL comparison, ablations, and videos must be run by the user; MuJoCo is optional and not needed for the core tasks.
