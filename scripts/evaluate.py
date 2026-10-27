from __future__ import annotations

import argparse
import json

import gymnasium as gym
import torch

from src.deep_rl.agents.dqn import DQNAgent, DQNConfig
from src.deep_rl.agents.ppo import PPOAgent, PPOConfig
from src.deep_rl.utils import choose_device
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--environment", required=True); parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    env = gym.make(args.environment)
    if args.algorithm == "dqn":
        cfg = DQNConfig(**checkpoint["config"]); agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, device); agent.online.load_state_dict(checkpoint["model"])
    else:
        cfg = PPOConfig(**checkpoint["config"]); agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg, device); agent.network.load_state_dict(checkpoint["model"])
    values = evaluate(agent, args.environment, args.seed, args.episodes, args.algorithm, device)
    print(json.dumps({"returns": values, "mean": sum(values)/len(values), "std": float(torch.tensor(values).std(unbiased=False))}))
    env.close()

if __name__ == "__main__": main()
