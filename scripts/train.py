from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.deep_rl.agents.dqn import DQNAgent, DQNConfig
from src.deep_rl.agents.ppo import PPOAgent, PPOConfig
from src.deep_rl.buffers.rollout import Rollout
from src.deep_rl.utils import choose_device, load_config, seed_everything


def evaluate(agent, env_id: str, seed: int, episodes: int, algo: str, device) -> list[float]:
    env = gym.make(env_id)
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + 100_000 + episode)
        total = 0.0
        while True:
            if algo == "dqn":
                action = agent.act(obs, epsilon=0.0)
            else:
                normalized, _, _ = agent.act(obs, deterministic=True)
                action = normalized * env.action_space.high
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    config = load_config(args.config)
    total_steps = args.total_steps or int(config["total_steps"])
    device = choose_device(args.device or config.get("device", "auto"))
    seed_everything(args.seed)
    env_id, algo = config["environment"], config["algorithm"].lower()
    env = gym.make(env_id)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    run_name = f"{algo}_{env_id}_{args.seed}_{int(time.time())}"
    out = Path(args.output) / run_name
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "seed": args.seed, "device": str(device), "python_torch": torch.__version__}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    history: list[dict] = []
    start = time.perf_counter()

    if algo == "dqn":
        cfg = DQNConfig(**config["dqn"])
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, device)
        episode_return, episode_length, losses = 0.0, 0, []
        for step in range(1, total_steps + 1):
            action = agent.act(obs, agent.epsilon(step))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.replay.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            episode_length += 1
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            if terminated or truncated:
                history.append({"step": step, "episode_return": episode_return, "episode_length": episode_length,
                                "loss": float(np.mean(losses[-100:])) if losses else None, "epsilon": agent.epsilon(step)})
                obs, _ = env.reset()
                episode_return, episode_length = 0.0, 0
            if step % int(config["evaluation_interval"]) == 0:
                returns = evaluate(agent, env_id, args.seed, int(config["evaluation_episodes"]), algo, device)
                history.append({"step": step, "evaluation_return": float(np.mean(returns)), "evaluation_std": float(np.std(returns))})
        torch.save({"model": agent.online.state_dict(), "config": asdict(cfg), "seed": args.seed}, out / "checkpoint.pt")
    elif algo == "ppo":
        cfg = PPOConfig(**config["ppo"])
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg, device)
        completed_returns: list[float] = []
        ep_return = 0.0
        env_steps = 0
        while env_steps < total_steps:
            n = min(cfg.rollout_steps, total_steps - env_steps)
            obs_list, action_list, log_list, reward_list, value_list, term_list, trunc_list, next_value_list = [], [], [], [], [], [], [], []
            for _ in range(n):
                norm_action, log_prob, value = agent.act(obs)
                env_action = norm_action * env.action_space.high
                next_obs, reward, terminated, truncated, _ = env.step(env_action)
                # Value of final observation is used to bootstrap truncations.
                with torch.no_grad():
                    next_value = float(agent.network.value(torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)).item())
                obs_list.append(obs); action_list.append(norm_action); log_list.append(log_prob); reward_list.append(reward)
                value_list.append(value); term_list.append(float(terminated)); trunc_list.append(float(truncated)); next_value_list.append(next_value)
                ep_return += reward
                # Preserve the final observation for bootstrap above, then start a
                # fresh episode only after recording the boundary flags.
                if terminated or truncated:
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                if terminated or truncated:
                    completed_returns.append(ep_return); ep_return = 0.0
            rollout = Rollout(np.asarray(obs_list), np.asarray(action_list), np.asarray(log_list), np.asarray(reward_list, dtype=np.float32),
                              np.asarray(value_list, dtype=np.float32), np.asarray(term_list, dtype=np.float32),
                              np.asarray(trunc_list, dtype=np.float32), np.asarray(next_value_list, dtype=np.float32))
            advantages, returns = rollout.advantages_and_returns(cfg.gamma, cfg.gae_lambda)
            metrics = agent.optimize(rollout.observations, rollout.actions, rollout.log_probs, advantages, returns)
            env_steps += n
            history.append({"step": env_steps, "mean_episode_return": float(np.mean(completed_returns[-20:])) if completed_returns else None,
                            "wall_time": time.perf_counter() - start, **metrics})
            if env_steps % int(config["evaluation_interval"]) < n:
                vals = evaluate(agent, env_id, args.seed, int(config["evaluation_episodes"]), algo, device)
                history.append({"step": env_steps, "evaluation_return": float(np.mean(vals)), "evaluation_std": float(np.std(vals))})
        torch.save({"model": agent.network.state_dict(), "config": asdict(cfg), "seed": args.seed}, out / "checkpoint.pt")
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    env.close()
    elapsed = time.perf_counter() - start
    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if history:
        keys = sorted(set().union(*(row.keys() for row in history)))
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(history)
    print(json.dumps({"run": str(out), "environment_steps": total_steps, "wall_time_seconds": elapsed, "device": str(device)}))


if __name__ == "__main__":
    main()
