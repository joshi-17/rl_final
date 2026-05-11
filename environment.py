"""
environment.py — Smart Energy Management RL Environment

Simulates a battery-powered device that receives tasks with varying load.
The environment manages:
  - Battery drain based on chosen power level
  - Random task generation (low / medium / high demand)
  - State transitions (battery_level x task_demand)

State  : (battery_level [0-2], task_demand [0-2])
Action : power level    0=low  1=medium  2=high
"""

import random


# -- Human-readable label maps -------------------------------------------------
BATTERY_LABELS = {0: "low", 1: "medium", 2: "high"}
TASK_LABELS    = {0: "low", 1: "medium", 2: "high"}
ACTION_LABELS  = {0: "low_power", 1: "medium_power", 2: "high_power"}


class EnergyEnvironment:
    """
    Discrete RL environment for smart energy management.

    Battery is tracked as a continuous percentage (0-100) and then
    discretised into three levels for the state representation:
        0 = low   ( 0 – 33 %)
        1 = medium (34 – 66 %)
        2 = high  (67 – 100%)

    Task demand is sampled uniformly at random each step.

    Reward = task_completion_score – battery_usage_penalty
    """

    # ------------------------------------------------------------------
    # Default battery drain per step (percentage points) for each action
    # ------------------------------------------------------------------
    DEFAULT_DRAIN = {
        0: 2,   # low power   -> small drain
        1: 5,   # medium power -> moderate drain
        2: 10,  # high power  -> heavy drain
    }

    # ------------------------------------------------------------------
    # Task completion score  key = (action, task_demand)
    # Matching or exceeding demand gives the best score.
    # ------------------------------------------------------------------
    DEFAULT_SCORES = {
        (0, 0): 10,  (0, 1): 4,  (0, 2): 1,   # low power
        (1, 0):  8,  (1, 1): 10, (1, 2): 4,   # medium power
        (2, 0):  6,  (2, 1):  8, (2, 2): 10,  # high power
    }

    def __init__(self, max_battery=100, drain_rates=None, completion_scores=None, seed=None):
        """
        Parameters
        ----------
        max_battery        : int   Full charge value (default 100)
        drain_rates        : dict  Override default drain per action
        completion_scores  : dict  Override default scores per (action, demand)
        seed               : int   Optional RNG seed for reproducibility
        """
        self.max_battery       = max_battery
        self.drain_rates       = drain_rates or self.DEFAULT_DRAIN
        self.completion_scores = completion_scores or self.DEFAULT_SCORES

        if seed is not None:
            random.seed(seed)

        # Will be initialised in reset()
        self.battery      = max_battery
        self.current_task = 0
        self.done         = False

    # -- Public API -------------------------------------------------------------

    def reset(self):
        """Reset environment to a fully-charged state. Returns initial state."""
        self.battery      = self.max_battery
        self.done         = False
        self.current_task = self._generate_task()
        return self._get_state()

    def step(self, action):
        """
        Apply an action and advance one timestep.

        Parameters
        ----------
        action : int  0=low_power  1=medium_power  2=high_power

        Returns
        -------
        next_state : tuple(int, int)
        reward     : float
        done       : bool
        info       : dict  (diagnostic information)
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset() before stepping.")

        # -- 1. Task completion score ------------------------------------------
        task_score = self.completion_scores.get((action, self.current_task), 0)

        # -- 2. Drain the battery ----------------------------------------------
        drain          = self.drain_rates[action]
        self.battery   = max(0.0, self.battery - drain)

        # -- 3. Battery-usage penalty (proportional to drain) -----------------
        battery_penalty = drain * 0.5

        # -- 4. Reward ---------------------------------------------------------
        reward = task_score - battery_penalty

        # -- 5. Terminal condition ---------------------------------------------
        if self.battery <= 0:
            self.done = True

        # -- 6. Next task (state transition) ----------------------------------
        self.current_task = self._generate_task()
        next_state = self._get_state()

        info = {
            "battery_pct"      : self.get_battery_percentage(),
            "task_score"       : task_score,
            "battery_penalty"  : battery_penalty,
            "action_label"     : ACTION_LABELS[action],
            "task_label"       : TASK_LABELS[self.current_task],
        }
        return next_state, reward, self.done, info

    # -- Utility helpers --------------------------------------------------------

    def get_battery_percentage(self):
        """Return battery as a 0-100 percentage."""
        return (self.battery / self.max_battery) * 100.0

    def _generate_task(self):
        """Sample a task demand uniformly: 0=low, 1=medium, 2=high."""
        return random.randint(0, 2)

    def _discretise_battery(self):
        """Map continuous battery % to a discrete level (0/1/2)."""
        pct = self.get_battery_percentage()
        if pct > 66:
            return 2   # high
        elif pct > 33:
            return 1   # medium
        else:
            return 0   # low

    def _get_state(self):
        """Return the current (battery_level, task_demand) state tuple."""
        return (self._discretise_battery(), self.current_task)

    def __repr__(self):
        b, t = self._get_state()
        return (f"EnergyEnvironment(battery={self.get_battery_percentage():.1f}% "
                f"[{BATTERY_LABELS[b]}], task={TASK_LABELS[t]}, done={self.done})")
