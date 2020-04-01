"""
Settings related to self-play and training schedules, learning, etc.
"""


# SELF PLAY
EPISODES = 25  # Number of matches played between current and best AlphaZero during model comparison.
MCTS_SIMS = 50  # Number of Monte Carlo simulations before acting on the environment.
MEMORY_SIZE = 5000  # Number of states needed in memory before refitting.
TURNS_UNTIL_TAU0 = 5  # Turns until deterministic play.
CPUCT = 1  # Multiplicative constant to the exploration term. Higher values encourage stochastic play and plays suggested by the policy. (Inverse empirically)
EPSILON = 0.2  # Suggested 0 to 1. Multiplicative term to Dirichlet noise. Higher values encourage stochastic play.
ALPHA = 0.8    # Input to dirichlet noise added to prior probabilities. Lower values encourage exploration at root node.

# RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [{'filters': 64, 'kernel_size': (2, 2)}] * 6

# EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3
