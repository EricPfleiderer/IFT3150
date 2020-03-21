# Standard
import sys
from importlib import reload

# Third Party
import numpy as np
import pandas as pd

# Local
from AlphaSilico.src.printer import standard_treatment
from AlphaSilico.src.insilico import Environment, State
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src.MCTS import MCTS, Node
from AlphaSilico.src import config


# PRINTOUTS
# standard_treatment()

# Environment parameters
min_doses = 0
max_doses = 4
treatment_start = 0  # Treatment start offset in days
treatment_len = 75  # Treatment length in days
observation_len = 90  # Observation period length, including treatment

# Initialize the environment
environment = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)

X = [[np.arange(10)], [np.arange(28)]]

agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))

agent.brain.plot_model()
agent.build_MCTS_root(environment.state)

for x in range(3):
    agent.simulate()




