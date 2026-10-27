# Experiment and failure analysis plan

No empirical failures or benchmark outcomes are asserted until the corresponding runs have been executed and saved. Use at least seeds 42, 123, and 456 and preserve every run folder.

| Factor | Practical ablation | What to inspect |
|---|---|---|
| Learning rate | PPO 1e-4, 3e-4, 1e-3 | KL, value loss, return variance |
| Clip coefficient | 0.1, 0.2, 0.3 | clip fraction and policy improvement |
| Entropy coefficient | 0, 0.001, 0.01 | action dispersion and returns |
| Rollout length | 256, 1024, 2048 | update frequency, variance, wall time |
| GAE lambda | 0.90, 0.95, 0.99 | advantage noise and value bias |

DQN can oscillate because bootstrapped targets depend on learned estimates; maximization over noisy action values can overestimate values. Double DQN reduces this selection/evaluation coupling. PPO can collapse exploration if entropy is too low, or learn slowly if too high. A weak critic yields poor advantages; a large learning rate or excessive reuse of one rollout may cause destructive policy updates.

These are expected mechanisms, not observations about this project's runs. Individual seeds can fail due to stochastic transitions, initialization, exploration, and finite sample budgets even when update equations are correct. Report seed distributions, evaluation curves, AUC, and wall time; do not select only successful seeds. The current ablation entry point runs repeatable seed groups, while changing an ablation parameter is done explicitly in a copied YAML config to keep the source defaults and experiment provenance clear.
