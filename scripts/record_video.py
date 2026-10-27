from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from src.deep_rl.agents.dqn import DQNAgent, DQNConfig
from src.deep_rl.agents.ppo import PPOAgent, PPOConfig
from src.deep_rl.utils import choose_device


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--environment", required=True); parser.add_argument("--output", default="videos"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); device = choose_device("auto"); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env = gym.make(args.environment, render_mode="rgb_array"); env = gym.wrappers.RecordVideo(env, video_folder=str(out), episode_trigger=lambda _: True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.algorithm == "dqn":
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, DQNConfig(**ckpt["config"]), device); agent.online.load_state_dict(ckpt["model"])
    else:
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig(**ckpt["config"]), device); agent.network.load_state_dict(ckpt["model"])
    obs, _ = env.reset(seed=args.seed)
    while True:
        if args.algorithm == "dqn": action = agent.act(obs)
        else:
            action, _, _ = agent.act(obs, deterministic=True); action = action * env.action_space.high
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    env.close()

if __name__ == "__main__": main()
