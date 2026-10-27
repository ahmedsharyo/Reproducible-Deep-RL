import numpy as np
import torch

from src.deep_rl.agents.dqn import DQNAgent, DQNConfig
from src.deep_rl.agents.ppo import PPOAgent, PPOConfig
from src.deep_rl.buffers.replay import ReplayBuffer
from src.deep_rl.buffers.rollout import Rollout
from src.deep_rl.networks.actor_critic import ActorCritic
from src.deep_rl.networks.q_network import QNetwork


def test_q_network_shape():
    assert QNetwork(4, 2)(torch.zeros(3, 4)).shape == (3, 2)


def test_replay_sample_and_terminal_flag():
    replay = ReplayBuffer(5)
    replay.add(np.zeros(4), 0, 1, np.ones(4), True)
    batch = replay.sample(1)
    assert batch["observations"].shape == (1, 4)
    assert batch["terminated"].item() == 1


def test_dqn_terminal_bootstrap_mask():
    agent = DQNAgent(4, 2, DQNConfig(batch_size=1), torch.device("cpu"))
    with torch.no_grad():
        for p in agent.target.parameters(): p.zero_()
        agent.target.model[-1].bias.fill_(2.0)
    vals = agent.bellman_targets([1, 1], np.zeros((2,4), dtype=np.float32), [1,0])
    assert torch.allclose(vals, torch.tensor([1.0, 1 + .99*2]))


def test_dqn_gradient_and_target_update():
    cfg = DQNConfig(batch_size=2, learning_starts=2, target_update_frequency=1)
    agent = DQNAgent(4, 2, cfg, torch.device("cpu"))
    agent.replay.add(np.zeros(4),0,1,np.ones(4),False); agent.replay.add(np.ones(4),1,0,np.zeros(4),True)
    assert agent.update() is not None
    assert all(torch.equal(a,b) for a,b in zip(agent.online.parameters(),agent.target.parameters()))


def test_actor_critic_actions_and_log_probs():
    net = ActorCritic(3, 1)
    actions, logp, value = net.sample(torch.zeros(5,3))
    assert actions.shape == (5,1) and value.shape == (5,) and torch.isfinite(logp).all()
    assert (actions.abs() <= 1).all()


def test_gae_terminal_and_truncation_bootstrap():
    rollout = Rollout(np.zeros((2,1)),np.zeros((2,1)),np.zeros(2),np.array([1.,1.]),np.array([0.,0.]),
                      np.array([0.,0.]),np.array([1.,0.]),np.array([5.,5.]))
    adv, _ = rollout.advantages_and_returns(1.0, 1.0)
    assert np.allclose(adv, [6.,6.])
    terminal = Rollout(np.zeros((1,1)),np.zeros((1,1)),np.zeros(1),np.array([1.]),np.array([0.]),np.array([1.]),np.array([0.]),np.array([5.]))
    assert np.allclose(terminal.advantages_and_returns(1.,1.)[0], [1.])


def test_ppo_optimization_step():
    agent = PPOAgent(3, 1, PPOConfig(update_epochs=1, minibatch_size=4), torch.device("cpu"))
    obs = np.zeros((4,3),dtype=np.float32); actions=np.zeros((4,1),dtype=np.float32)
    metrics=agent.optimize(obs,actions,np.zeros(4),np.array([1.,2.,3.,4.]),np.ones(4))
    assert set(metrics) == {"policy_loss","value_loss","entropy","approx_kl","clip_fraction"}
    assert all(np.isfinite(list(metrics.values())))
