import pickle
from shutil import copyfile
from importlib import reload
import numpy as np

from AlphaSilico.src import settings, config
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src.memory import Memory
from AlphaSilico.src.insilico import Environment


def clinical_trial(agent, memory=None, turns_until_tau0=config.TURNS_UNTIL_TAU0, cohort_size=config.EPISODES):
    """
    Perform a clinical trial using an agent. Returns trial statistics.
    :param agent: Agent instance. Self learning agent to be evaluated.
    :param memory: Memory instance.
    :param turns_until_tau0: Int. Turn at which agent starts playing deterministically.
    :param cohort_size: Integer. Size of cohort during trial.
    :return: Clinical trial statistics.
    """

    # Load default parameters
    cohort_params = np.array([config.DEFAULT_PARAMS])

    # Patient params are within 10% of default values 99.7% of the time according to normal distribution.
    if cohort_size != 1:
        cohort_params = np.random.normal(config.DEFAULT_PARAMS, config.DEFAULT_PARAMS/30, size=cohort_size)

    stats = {'z': [],
             'v': [],
             }

    for params in cohort_params:

        trial = Environment(params=params, treatment_start=config.TREATMENT_START, treatment_len=config.TREATMENT_LEN, observation_len=config.OBSERVATION_LEN,
                            max_doses=config.MAX_DOSES, immunotherapy_offset=config.IMMUNOTHERAPY_OFFSET, virotherapy_offset=config.VIROTHERAPY_OFFSET)

        done = False
        while not done:

            # Select an action
            after_tau0 = int(trial.t) > config.TURNS_UNTIL_TAU0  # Check if agent should play randomly or deterministically
            action, pi, MCTS_value, NN_value = agent.act(trial.state, after_tau0=after_tau0, tau=config.TAU)

            # Play the chosen action
            next_state, value, done = trial.step(action)

            # Get endgame statistics
            if done:
                stats['z'].append(MCTS_value)
                stats['v'].append(NN_value)

        trial.reset()

    return stats


def compete(first_agent, second_agent, episodes, turns_until_tau0, memory=None):

    """
    Put two agents into competition. Each agent must complete a clinical trial.
    :param first_agent:
    :param second_agent:
    :param episodes:
    :param turns_until_tau0:
    :param memory:
    :return:
    """

    return 0, Memory(memory_size=config.MEMORY_SIZE), 0, 0


def train():

    # Load memory if necessary
    if settings.INITIAL_MEMORY_VERSION is None:
        memory = Memory(config.MEMORY_SIZE)
    else:
        print('LOADING MEMORY VERSION ' + str(settings.INITIAL_MEMORY_VERSION) + '...')
        memory = pickle.load(open(settings.archive_folder + 'Model_' + str(settings.INITIAL_RUN_NUMBER) + "/memory" + str(settings.INITIAL_MEMORY_VERSION).zfill(4) +
                                  ".p", "rb"))

    # Agents
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
