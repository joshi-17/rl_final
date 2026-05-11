"""
agent.py — Q-Learning Agent for Smart Energy Management

Algorithm: Q-Learning (off-policy TD control)

Why Q-Learning?
  The state space is discrete and small (3 battery levels × 3 task demands = 9 states)
  and the action space has only 3 choices. Q-learning converges reliably in such
  tabular settings without needing neural networks or function approximators.

Q-Table shape: [battery_levels=3, task_demands=3, actions=3]

Update rule:
  Q(s, a) <- Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]

Exploration: epsilon-greedy
  With probability ε  -> take a random action  (explore)
  With probability 1-ε -> take the greedy action (exploit)
  ε decays over time so the agent gradually shifts from exploration to exploitation.
"""

import numpy as np
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    n_battery  : int   Number of discrete battery levels  (default 3)
    n_tasks    : int   Number of discrete task demands    (default 3)
    n_actions  : int   Number of power-level choices      (default 3)
    lr         : float Learning rate α                    (default 0.1)
    gamma      : float Discount factor γ                  (default 0.95)
    epsilon    : float Initial exploration rate           (default 1.0)
    epsilon_min: float Minimum exploration rate           (default 0.01)
    epsilon_decay: float Multiplicative decay per episode (default 0.995)
    """

    def __init__(
        self,
        n_battery=3,
        n_tasks=3,
        n_actions=3,
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        # -- Hyper-parameters --------------------------------------------------
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions     = n_actions

        # -- Q-table initialised to zeros --------------------------------------
        # Dimensions: [battery_level, task_demand, action]
        self.q_table = np.zeros((n_battery, n_tasks, n_actions))

    # -- Core methods -----------------------------------------------------------

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : tuple(int, int)  (battery_level, task_demand)

        Returns
        -------
        action : int
        """
        if np.random.random() < self.epsilon:
            # Explore: pick a random power level
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: pick the action with the highest Q-value
            b, t = state
            return int(np.argmax(self.q_table[b, t]))

    def update(self, state, action, reward, next_state, done):
        """
        Apply the Q-learning update rule for one transition.

        Q(s,a) <- Q(s,a) + α · [target - Q(s,a)]
        target  = r                          if done
                = r + γ · max_a' Q(s', a')  otherwise
        """
        b,  t  = state
        nb, nt = next_state

        current_q = self.q_table[b, t, action]

        if done:
            # Terminal state: no future reward
            target_q = reward
        else:
            # Bellman target with greedy look-ahead
            target_q = reward + self.gamma * np.max(self.q_table[nb, nt])

        # Gradient step
        self.q_table[b, t, action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay ε after each episode (call once per episode end)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- Policy extraction -----------------------------------------------------

    def get_policy(self):
        """
        Extract the greedy policy from the current Q-table.

        Returns
        -------
        policy : dict  { (battery_level, task_demand): best_action }
        """
        policy = {}
        for b in range(self.q_table.shape[0]):
            for t in range(self.q_table.shape[1]):
                policy[(b, t)] = int(np.argmax(self.q_table[b, t]))
        return policy

    def get_q_table_summary(self):
        """Print Q-table values for debugging / inspection."""
        labels_b = ["low ", "med ", "high"]
        labels_t = ["low ", "med ", "high"]
        labels_a = ["low_pwr", "med_pwr", "hi_pwr "]
        print("\n-- Q-Table (battery × task -> Q per action) --")
        print(f"{'State':>20}  |  " + "  ".join(labels_a))
        print("-" * 60)
        for b in range(3):
            for t in range(3):
                q_vals = self.q_table[b, t]
                row = "  ".join(f"{v:7.3f}" for v in q_vals)
                print(f"  bat={labels_b[b]} task={labels_t[t]}  |  {row}")
        print()

    # -- Persistence -----------------------------------------------------------

    def save(self, filepath):
        """Serialise the agent (Q-table + hyper-params) to a pickle file."""
        payload = {
            "q_table"       : self.q_table,
            "lr"            : self.lr,
            "gamma"         : self.gamma,
            "epsilon"       : self.epsilon,
            "epsilon_min"   : self.epsilon_min,
            "epsilon_decay" : self.epsilon_decay,
        }
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"  [agent] Saved to {filepath}")

    def load(self, filepath):
        """Restore agent state from a pickle file."""
        with open(filepath, "rb") as fh:
            data = pickle.load(fh)
        self.q_table       = data["q_table"]
        self.lr            = data["lr"]
        self.gamma         = data["gamma"]
        self.epsilon       = data["epsilon"]
        self.epsilon_min   = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        print(f"  [agent] Loaded from {filepath}")
