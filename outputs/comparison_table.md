# Comparison Results: Baseline vs RL Policy

| Metric | Fixed-Power Baseline | RL Policy (Q-Learning) | Delta (RL - Base) |
|--------|---------------------|------------------------|---------------|
| Avg Total Reward | 96.50 | 179.62 | +83.12 (+86.1%) |
| Std Reward | 12.01 | 25.13 | — |
| Min Reward | 74.00 | 135.00 | — |
| Max Reward | 124.00 | 247.50 | — |
| Avg Battery Remaining (%) | 0.00 | 0.00 | +0.00 |
| Strategy | Always medium power | State-adaptive Q-table | — |
| Adapts to demand | No | Yes | — |

**RL Policy improves over Baseline by 86.1% in average total reward.**