# Third Party
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

# Local
from AlphaSilico.src.insilico import Environment
from AlphaSilico.src.learners import Learner, Agent
from AlphaSilico.src import config


def MCTS_variance(simulations=250, runs=5):
    # Environment parameters
    treatment_start = 0  # Treatment start offset in days
    treatment_len = 10  # Treatment length in days
    observation_len = 180  # Observation period length, including treatment

    MC_tree_sizes = []

    for i in range(runs):

        environment = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
        agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))
        agent.build_MCTS_root(environment.state)

        MC_tree_size = []

        for j in range(simulations):
            agent.simulate()
            MC_tree_size.append(len(agent.mcts.tree))

        MC_tree_sizes.append(MC_tree_size)

    MC_tree_sizes = np.array(MC_tree_sizes)

    plt.figure()
    plt.errorbar(x=np.arange(0, MC_tree_sizes.shape[1]), y=np.sum(MC_tree_sizes), yerr=np.std(MC_tree_sizes))
    plt.xlabel('Number of simulations')
    plt.ylabel('Tree size')
    plt.savefig('outputs/MCTS_analysis_variance.png')


def MCTS_params(simulations=200):
    # Environment parameters
    treatment_start = 0  # Treatment start offset in days
    treatment_len = 10  # Treatment length in days
    observation_len = 180  # Observation period length, including treatment

    # Avg MC tree size vs simulations, varying CPUCT, EPSILON, ALPHA

    test_params = {'CPUCT': (0.1, 1, 10),
                   'EPSILON': (0.01, 0.1, 0.8),
                   'ALPHA': (0.1, 0.8, 10)
                   }

    colors = ((0.8, 0, 0), (0, 0.8, 0), (0, 0, 0.8))

    for param_name, param_values in test_params.items():
        print('Analysing ', param_name, '...')

        MC_tree_sizes = []

        for param_value in param_values:

            setattr(config, param_name, param_value)

            environment = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
            agent = Agent('Singularity', 4, config.MCTS_SIMS, config.CPUCT, Learner(learning_rate=1e-3))
            agent.build_MCTS_root(environment.state)

            MC_tree_size = []

            for x in range(simulations):
                if x % 50 == 0:
                    print('Simulation #', x)
                agent.simulate()
                MC_tree_size.append(len(agent.mcts.tree))

            MC_tree_sizes.append(MC_tree_size)
            reload(config)

        plt.figure()
        for idx, MC_tree_size in enumerate(MC_tree_sizes):
            plt.plot(np.arange(len(MC_tree_size)), MC_tree_size, c=colors[idx], label=param_name+' = '+str(param_values[idx]))
        plt.xlabel('Number of simulations')
        plt.ylabel('Tree size')
        plt.legend()
        plt.savefig('outputs/MCTS_analysis_' + param_name + '.png')


def standard_treatment():

    # Simulation and treatment parameters
    min_doses = 0
    max_doses = 4
    treatment_start = 0
    treatment_len = 75  # Treatment length in days
    observation_len = 90  # Observation period length, including treatment

    # Random treatment
    treatment = np.transpose(np.array([np.random.randint(min_doses, max_doses+1, size=treatment_len), np.random.randint(min_doses, max_doses+1, size=treatment_len)]))

    # Models
    # Treated tumor
    env = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
    for day in range(observation_len):
        actions = (0, 0)
        if day < treatment_len:
            actions = (treatment[day][0], treatment[day][1])
        env.step(actions)

    tumor_size, cumulative_tumor_burden = env.evaluate_obective()  # Compute the objective function from the history

    # Untreated tumor, control group
    control = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
    for _ in range(observation_len):
        actions = (0, 0)
        control.step(actions)
    control_size, control_ctb = control.evaluate_obective()

    print('Plotting...')
    titles = ['Quiescent cells', 'G1 cells', 'Infected cells', 'Virions'] + \
             ['Transit compartment ' + str(n+1) for n in range(env.state.j)] + \
             ['Cytokines', 'Phagocytes', 'Total number of cells in cycle', 'Resistant quiescent cells', 'Resistant G1 cells'] + \
             ['Resistant transit compartment ' + str(n+1) for n in range(env.state.j)] + \
             ['Total number of resistant cells in cycle']

    # Print tracked quantities
    for idx, quantity in enumerate(env.history['y'].transpose()):
        plt.figure()
        plt.plot(np.arange(0, observation_len + env.dt, env.dt), quantity)
        plt.title(titles[idx])
        plt.savefig('outputs/' + str(idx) + '_' + titles[idx] + '.png')

    # Print main metric (tumor size)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Tumor size')
    line1 = ax1.plot(np.arange(0, observation_len + env.dt, env.dt), tumor_size, '--', c=(0, 0.6, 0), label='Tumor size, test')
    line2 = ax1.plot(np.arange(0, observation_len + env.dt, control.dt), control_size, '-', c=(0.2, 0, 0), label='Tumor size, control')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative burden')
    line3 = ax2.plot(np.arange(0, observation_len, env.dt), cumulative_tumor_burden, '--', c=(0, 0.8, 0), label='Cumulative burden, test')
    line4 = ax2.plot(np.arange(0, observation_len, control.dt), control_ctb, '-', c=(0.8, 0, 0), label='Cumulative burden, control')
    ax2.tick_params(axis='y')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    lines = line1+line2+line3+line4
    plt.legend(lines, [l.get_label() for l in lines])
    plt.title('Tumor growth')
    plt.savefig('outputs/Objective.png')

    # Print dosages vs time
    plt.figure()
    plt.plot(env.state.dose_history['immunotherapy']['t'], env.state.dose_history['immunotherapy']['y'])
    plt.xlabel('Temps (jours)')
    plt.ylabel('Cytokines')
    plt.savefig('outputs/Immunotherapy_doses.png')

    plt.figure()
    plt.plot(env.state.dose_history['virotherapy']['t'], env.state.dose_history['virotherapy']['y'])
    plt.xlabel('Temps (jours)')
    plt.ylabel('Virus')
    plt.savefig('outputs/Virotherapy_doses.png')
