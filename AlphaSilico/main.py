# Standard
import sys

# Third Party
import numpy as np
import pandas as pd

# Local
# from AlphaSilico.src import printer # Uncomment to print graphs

from AlphaSilico.src.learner import Learner
from AlphaSilico.src.Agent import Agent
from AlphaSilico.src import config

alphasilico = Agent(name='AlphaSilico', action_size=(4, 4), mcts_simulations=config.MCTS_SIMS, cpuct=config.CPUCT, model=Learner())



