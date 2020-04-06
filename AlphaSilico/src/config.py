"""
Settings related to self-play and training schedules, learning, etc.
"""

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGENT CONFIG

# Self play
EPISODES = 25  # Number of matches played between current and best AlphaZero during model comparison.
MCTS_SIMS = 10  # Number of Monte Carlo simulations before acting on the environment.
MEMORY_SIZE = 5000  # Number of states needed in memory before refitting.
TURNS_UNTIL_TAU0 = 5  # Turns until deterministic play.
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
EVAL_EPISODES = 1  # 20  # TESTING
SCORING_THRESHOLD = 1.3


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INSILICO CONFIG

# Environment parameters
TREATMENT_START = 0  # Treatment start offset in days
TREATMENT_LEN = 10  # Treatment length in days
OBSERVATION_LEN = 180  # Observation period length, including treatment
MAX_DOSES = 4
IMMUNOTHERAPY_OFFSET = 1
VIROTHERAPY_OFFSET = 7

DEFAULT_PARAMS = (1.183658646441553, 1.758233712464858, 0, 0.539325116600707, 0.539325116600707, (33.7/24 - 1/1.758233712464858), 0.05, 10, 10, 4.6754)

