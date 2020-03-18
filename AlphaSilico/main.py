# Standard
import sys
from importlib import reload

# Third Party
import numpy as np
import pandas as pd

# Local
from AlphaSilico.src import printer # Uncomment to print graphs

from AlphaSilico.src.insilico import Environment, State
from AlphaSilico.src.learners import GenericLearner
from AlphaSilico.src.Agent import Agent
from AlphaSilico.src import config

# # Initialize the environment
# environment = Environment()
#
# # Create players TO DO: implement model load/save
# current_agent = Agent(name='AlphaSilico', action_size=(4, 4), mcts_simulations=config.MCTS_SIMS, cpuct=config.CPUCT, model=GenericLearner())
# best_agent = Agent(name='AlphaSilico', action_size=(4, 4), mcts_simulations=config.MCTS_SIMS, cpuct=config.CPUCT, model=GenericLearner())
