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
from AlphaSilico.src.training import clinical_trial, train

# PRINTOUTS
# standard_treatment()
# MCTS_params(simulations=200)
# MCTS_variance(simulations=200, runs=5)

# agent = Agent(name='Brain', action_size=5, mcts_simulations=config.MCTS_SIMS, cpuct=config.CPUCT, learner=Learner(learning_rate=config.LEARNING_RATE))

train()

'''

DEBUG:
    -MCTS should not use done condition, only is_leaf()
        -expande/evaluate should deal with done condition (only expand if not done)

TO DO:
    - Implement dynamic input size for solution vector.  Learner currently initializes with fixed input size, cannot deal with patients with varying tau parameter
        -When done, remove cohort_size=1 from clinical_trial() calls.
        
    - Optimize MCTS to avoid recomputing previously visited solutions
    
    - Parallelisation
    
    - Establish victory, draw, defeat criteria:
        - Treatment length: 75 days
        - Total simulation length: 180 days (or  360?)
        - Max dose: 4
        
        
brainstorm:
    - Perform criteria analysis 
        -make it harder to win (lower doubling treshold, lower total doses treshold) and see if lerned behavior is different
    -change max dose to 3, 2 and compare learned behaviors
    
'''


