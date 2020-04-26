import pickle
from copy import deepcopy
import sys
from shutil import copyfile
from importlib import reload
import numpy as np

from AlphaSilico.src import settings, config
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src.memory import Memory
from AlphaSilico.src.insilico import Environment


def clinical_trial(agent, turns_until_tau0, memory=None, episodes=config.EPISODES):
    """
    Perform a clinical trial using an agent. Returns trial statistics.
    :param agent: Agent instance. Self learning agent to be evaluated.
    :param memory: Memory instance.
    :param turns_until_tau0: Int. Turn at which agent starts playing deterministically.
    :param episodes: Integer. Number of trials performed.
    :return: Clinical trial statistics.
    """

    # # Load default parameters
    # cohort_params = np.array([config.DEFAULT_PARAMS])

    # # Patient params are within 10% of default values 99.7% of the time according to normal distribution. NOT YET FUNCTIONAL, NEED DYNAMIC NEURAL NET INPUT
    # if cohort_size != 1:
    #     cohort_params = np.random.multivariate_normal(cohort_params[0], np.diag(cohort_params[0]/30), size=cohort_size)

    stats = {'score': [],
             'v': [],
             }

    params = config.DEFAULT_PARAMS

    for episode in range(episodes):

        trial = Environment(params=params, treatment_start=config.TREATMENT_START, treatment_len=config.TREATMENT_LEN, observation_len=config.OBSERVATION_LEN,
                            max_doses=config.MAX_DOSES, immunotherapy_offset=config.IMMUNOTHERAPY_OFFSET, virotherapy_offset=config.VIROTHERAPY_OFFSET)

        print('EPISODE:', episode)

        done = False
        turn = 0
        while not done:

            if turn % 10 == 0:
                print('Day...', turn)

            # Select an action
            after_tau0 = turns_until_tau0 > 0  # Check if agent should play randomly or deterministically
            action, pi, MCTS_value, NN_value = agent.act(trial.state, after_tau0=after_tau0, tau=config.TAU)

            # Commit the move to memory
            if memory is not None:
                memory.commit_stmemory(params=params, y=trial.state.y, pi=pi)

            # Play the chosen action
            next_state, value, done = trial.step(action)
            turn += 1

            # Get endgame statistics
            if done:
                stats['score'].append(value)  # Save score (-1, 1)

                # Propagate z through path taken in MC tree
                if memory is not None:
                    for move in memory.stmemory:
                        move['value'] = value

                    memory.commit_ltmemory()  # Commit to long term and clear short term for next trial

        trial.reset()

    return stats, memory


def train():

    """
    Main training loop for AlphaSilico reinforcment agent.
    :return: Void.
    """

    # Training memory
    memory = Memory(config.MEMORY_SIZE)

    # Agents
    current_agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=config.LEARNING_RATE))
    best_agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=config.LEARNING_RATE))

    iteration = 0
    best_player_version = 0

    while 1:

        print('---------------------------------------------------------------------------------------------------------')
        print('ITERATION: ', iteration)

        iteration += 1
        reload(config)

        print('FILLING MEMORY...')

        # Self play (filling memory, semi-random play)
        _, memory = clinical_trial(best_agent, turns_until_tau0=config.TURNS_UNTIL_TAU0-iteration, memory=memory, episodes=config.EPISODES)

        # Once the memory is full...
        if len(memory.ltmemory) >= config.MEMORY_SIZE:

            print('FITTING AGENT...')

            # Fit the current agent on the long term memory
            current_agent.replay(memory.ltmemory)

            # Save memory every now and then (recent memory is higher quality)
            if iteration % 5 == 0:
                pickle.dump(memory, open(settings.run_folder + 'Model_' + str(settings.INITIAL_RUN_NUMBER) + "/memory" + str(iteration).zfill(4) + ".p", "wb"))

            # Score the agents in a competition (deterministic play)
            print('SCORING CURRENT AGENT...')
            current_stats, _ = clinical_trial(agent=current_agent, turns_until_tau0=0, memory=None, episodes=config.EVAL_EPISODES)
            print('SCORE:', current_stats['score'])
            print('SCORING BEST AGENT...')
            best_stats, _ = clinical_trial(agent=best_agent, turns_until_tau0=0, memory=None, episodes=config.EVAL_EPISODES)
            print('SCORE:', best_stats['score'])

            # Update best agent if needed
            if np.sum(current_stats['score']) > np.sum(best_stats['score']) * config.SCORING_THRESHOLD:
                print('BETTER AGENT FOUND! SAVING...')
                best_player_version = best_player_version + 1
                best_agent.brain = deepcopy(current_agent.brain)  # Deep copy the current brain since it scored higher
                pickle.dump(best_agent, open(settings.run_folder + 'Model_' + str(settings.INITIAL_RUN_NUMBER) + "/model" + str(iteration).zfill(4) + ".p", "wb"))

        else:
            print('MEMORY SIZE:', len(memory.ltmemory))


