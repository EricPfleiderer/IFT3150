# Standard
import sys
from importlib import reload

# Third Party
import numpy as np
import pandas as pd

# Local
from AlphaSilico.src.printer import standard_treatment, MCTS_params, MCTS_variance
from AlphaSilico.src.insilico import Environment
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src import config

# PRINTOUTS
# standard_treatment()
MCTS_params(simulations=200)
# MCTS_variance(simulations=200, runs=5)

# Environment parameters
min_doses = 0
max_doses = 4
treatment_start = 0  # Treatment start offset in days
treatment_len = 7  # Treatment length in days
observation_len = 180  # Observation period length, including treatment

# Initialize the environment and agent
environment = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))
agent.build_MCTS_root(environment.state)
