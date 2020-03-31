# Standard
import sys
from importlib import reload

# Third Party
import numpy as np
import pandas as pd

# Local
from AlphaSilico.src.printer import standard_treatment
from AlphaSilico.src.insilico import Environment
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src import config


# PRINTOUTS
# standard_treatment()

# Environment parameters
min_doses = 0
max_doses = 4
treatment_start = 0  # Treatment start offset in days
treatment_len = 5  # Treatment length in days
observation_len = 180  # Observation period length, including treatment

# Initialize the environment and agent
environment = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))


# Testing
# agent.brain.plot_model()
agent.build_MCTS_root(environment.state)

for x in range(500):
    print('Monte Carlo search #', x)
    agent.simulate()


X = 10
"""
Potential vulnerabilites:
    -Solver is deterministic but computer is imperfect. Simulating the same action twice on the same node might not result in states with exactly equal states.
     Must find unique state ID
        -patient params + t + treatment_history
"""


