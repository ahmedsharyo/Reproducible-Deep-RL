# PPO: problem, intuition, mathematics, algorithm, implementation

## Problem

An MDP is the tuple `(S, A, P, R, gamma)`: states, actions, transition probabilities, rewards, and a discount factor. At each step the agent observes `s_t`, chooses `a_t`, receives `r_t`, and reaches `s_{t+1}`. The goal is to maximize expected discounted return `E[sum_t gamma^t r_t]`.

Bellman learning (DQN) learns action values and picks the best discrete action. This works naturally for a small finite action set, as in CartPole's left/right choices. Continuous control has infinitely many possible torques, making enumeration unsuitable. Actor-critic methods learn a policy that directly emits a continuous action, alongside a critic estimating how good a state is.

## Intuition

PPO collects data under a current policy, then improves that policy in several minibatch passes. Reusing the rollout improves sample efficiency over a single policy-gradient pass. However, if the policy changes too far, the collected data no longer describes it well and learning can destabilize. PPO limits the incentive for large changes with a clipped probability ratio.

## Mathematics

The policy gradient theorem gives `grad J(theta) = E[grad log pi_theta(a|s) Q_pi(s,a)]`. Here `theta` are policy parameters, `pi` is the action distribution, and `Q` is expected future return. Increasing log probability of actions with positive return improves expected performance.

The critic estimates `V_phi(s)`, expected return from state `s`. Subtracting a baseline does not change the expected policy gradient but reduces variance. The advantage `A(s,a)=Q(s,a)-V(s)` measures whether an action did better than expected.

For a rollout, TD residual is `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`. GAE sums residuals with exponentially decaying weights: `A_t = sum_{l>=0}(gamma lambda)^l delta_{t+l}`. `gamma` discounts distant outcomes; `lambda` trades variance against bias. In code this is a reverse recursion in `Rollout.advantages_and_returns`.

PPO compares new and old policies using `r_t(theta)=pi_theta(a_t|s_t)/pi_old(a_t|s_t)=exp(logpi_new-logpi_old)`. Its clipped loss to minimize is `-E[min(r A, clip(r,1-epsilon,1+epsilon) A)]`. `A` is the old rollout's advantage and `epsilon` controls the allowed ratio range. Clipping removes additional benefit for moving farther in the favorable direction; it is a practical constraint, not a hard guarantee on KL divergence.

The critic loss is `0.5 E[(V_phi(s)-return)^2]`. Entropy encourages policy randomness. Total loss is policy loss plus value coefficient times value loss minus entropy coefficient times entropy. This implementation uses a tanh-squashed Gaussian; log probability includes the change-of-variables term `-sum(log(1-tanh(u)^2))`, where `u` is the pre-tanh Gaussian action. `ActorCritic.sample` and `evaluate_actions` implement the same transformation for action execution and PPO ratios.

## Algorithm

1. Reset the environment and sample actions from the policy.
2. Store observations, bounded normalized actions, old log probabilities, rewards, values, and both `terminated` and `truncated` flags.
3. Bootstrap from the final observation on truncation, but stop the advantage trace across reset boundaries. True termination disables bootstrapping.
4. Compute GAE and returns; normalize advantages.
5. Shuffle rollout indices and optimize policy and critic for multiple epochs using minibatches.
6. Monitor KL, clipping fraction, losses, entropy, evaluation return, and wall time. Stop epochs early if configured target KL is exceeded.

Pseudocode:

```text
for update:
    rollout = collect(policy, horizon)
    advantages, returns = GAE(rollout, gamma, lambda)
    normalize(advantages)
    for epoch:
        for minibatch in shuffled(rollout):
            ratio = exp(logp(policy, action) - old_logp)
            policy_loss = -mean(min(ratio * A, clip(ratio) * A))
            value_loss = mean_squared_error(value(state), returns)
            optimize(policy_loss + c_v * value_loss - c_e * entropy)
            stop epoch loop if approximate_KL > target_KL
```

PPO is on-policy: it discards old rollouts after updating. DQN is off-policy and can reuse replay, usually yielding better sample efficiency in suitable discrete problems. PPO is simpler and often stable but can require many environment steps. For example, Pendulum's torque is continuous, so the actor outputs a distribution over bounded torques rather than choosing from a table.

Gymnasium distinguishes termination (MDP terminal) from truncation (often a time limit). A terminal state has no future value; a truncated state should bootstrap from its final observation, though the GAE recursion must not continue into the reset episode. This distinction is implemented per transition in the rollout buffer.

## Debugging

Check action bounds and log-probability consistency first; then inspect reward scale, value prediction versus returns, advantage mean/std, approximate KL, clip fraction, and evaluation across multiple seeds. Entropy collapse can indicate too little exploration; high KL or clip fraction can indicate oversized updates; an unlearning critic makes the advantage signal noisy. Explained variance near one is useful but not sufficient evidence of a good policy.
