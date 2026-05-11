# Comparison Results: Baseline vs RL Policy

| Metric | Fixed-Power Baseline | RL Policy (Q-Learning) | Delta (RL - Base) |
|--------|---------------------|------------------------|---------------|
| Avg Total Reward | 96.96 | 175.12 | +78.16 (+80.6%) |
| Std Reward | 10.98 | 24.01 | — |
| Min Reward | 72.00 | 103.50 | — |
| Max Reward | 126.00 | 261.00 | — |
| Avg Battery Remaining (%) | 0.00 | 0.00 | +0.00 |
| Strategy | Always medium power | State-adaptive Q-table | — |
| Adapts to demand | No | Yes | — |

**RL Policy improves over Baseline by 80.6% in average total reward.**