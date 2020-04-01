import pickle
from shutil import copyfile
from importlib import reload

from AlphaSilico.src import settings, config
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src.memory import Memory


def compete(agent1, agent2, episodes, turns_until_tau0, memory=None):
    return 0, Memory(memory_size=config.MEMORY_SIZE), 0, 0


def train():

    # Load memory if necessary
    if settings.INITIAL_MEMORY_VERSION is None:
        memory = Memory(config.MEMORY_SIZE)
    else:
        print('LOADING MEMORY VERSION ' + str(settings.INITIAL_MEMORY_VERSION) + '...')
        memory = pickle.load(open(settings.archive_folder + 'Model_' + str(settings.INITIAL_RUN_NUMBER) + "/memory" + str(settings.INITIAL_MEMORY_VERSION).zfill(4) +
                                  ".p", "rb"))

    # Environment parameters
    treatment_start = 0  # Treatment start offset in days
    treatment_len = 75  # Treatment length in days
    observation_len = 180  # Observation period length, including treatment

    # Initialize the environment and agent
    environment = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)

    current_agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))
    best_agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))

    iteration = 0

    while 1:

        iteration += 1
        reload(config)

        # SELF PLAY (filling memory)
        _, memory, _, _ = compete(best_agent, best_agent, config.EPISODES, turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)

        if len(memory.ltmemory) >= config.MEMORY_SIZE:

            # Fit the current agent on the long term memory
            current_agent.replay(memory.ltmemory)

            # Save memory every now and then (more recent memory is higher quality)
            if iteration % 5 == 0:
                pickle.dump(memory, open(settings.run_folder + 'Model_' + str(settings.INITIAL_RUN_NUMBER) + "/memory" + str(iteration).zfill(4) + ".p", "wb"))

            # Score the agents in a competition
            scores, _, points, sp_scores = compete(best_agent, current_agent, config.EVAL_EPISODES, turns_until_tau0=0, memory=None)

            # Update best agent if needed
            # if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            #     best_player_version = best_player_version + 1
            #     best_NN.model.set_weights(current_NN.model.get_weights())
            #     best_NN.write(initialise.INITIAL_RUN_NUMBER, best_player_version)
