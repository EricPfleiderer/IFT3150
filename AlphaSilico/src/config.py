"""
Settings related to self-play and training schedules, learning, etc.
"""

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGENT CONFIG

# Self play
EPISODES = 10  # Number of clinical trials performed every iteration (fill memory).
MCTS_SIMS = 20  # Number of Monte Carlo simulations before acting on the environment.
MEMORY_SIZE = 300  # Number of states needed in memory before refitting.
TURNS_UNTIL_TAU0 = 10  # Turns until deterministic play.
TAU = 1  # Temperature variable for move selection.
CPUCT = 1  # Multiplicative constant to the exploration term. Higher values encourage stochastic play and plays suggested by the policy. (Inverse empirically)
EPSILON = 0.2  # Suggested 0 to 1. Multiplicative term to Dirichlet noise. Higher values encourage stochastic play.
ALPHA = 0.8    # Input to dirichlet noise added to prior probabilities. Lower values encourage exploration at root node.

# Training
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

# Evaluation
EVAL_EPISODES = 10  # Number of clinical trials performed every iteration (evaluate agents).
SCORING_THRESHOLD = 1.2  # Threshold to beat for current agent to become best agent.

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INSILICO CONFIG

# Environment parameters
TREATMENT_START = 0  # Treatment start offset in days
TREATMENT_LEN = 40  # Treatment length in days
OBSERVATION_LEN = 180  # Observation period length, including treatment
MAX_DOSES = 4  # Maximum dosages allowed (defines action space)
IMMUNOTHERAPY_OFFSET = 1  # Administer immunotherapy every x days
VIROTHERAPY_OFFSET = 7  # Administer virotherapy every x days

# Default patient
DEFAULT_PARAMS = (1.183658646441553, 1.758233712464858, 0, 0.539325116600707, 0.539325116600707, (33.7/24 - 1/1.758233712464858), 0.05, 10, 10, 4.6754)

