# Standard
import sys
from importlib import reload

# Third Party
import numpy as np
import pandas as pd

# Local
# from AlphaSilico.src import printer # Uncomment to print graphs

from AlphaSilico.src.insilico import TumorModel
from AlphaSilico.src.learners import GenericLearner
from AlphaSilico.src.Agent import Agent
from AlphaSilico.src import config

# Initialize the environment
environment = TumorModel()

# Create players TO DO: implement model load/save
current_agent = Agent(name='AlphaSilico', action_size=(4, 4), mcts_simulations=config.MCTS_SIMS, cpuct=config.CPUCT, model=Learner())
best_agent = Agent(name='AlphaSilico', action_size=(4, 4), mcts_simulations=config.MCTS_SIMS, cpuct=config.CPUCT, model=Learner())

# Training loop
iteration = 0
while 1:

    iteration += 1
    reload(config)  # Reload config file in case it was modified
