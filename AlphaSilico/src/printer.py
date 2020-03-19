# Third Party
import numpy as np
import matplotlib.pyplot as plt

# Local
from AlphaSilico.src.insilico import Environment


def standard_treatment():

    # Simulation and treatment parameters
    min_doses = 1
    max_doses = 2
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
    plt.plot(env.state.dose_history['immunotherapy']['t'][0:3000], env.state.dose_history['immunotherapy']['y'][0:3000])
    plt.xlabel('Temps (jours)')
    plt.ylabel('Cytokines')
    plt.savefig('outputs/Immunotherapy_doses.png')

    plt.figure()
    plt.plot(env.state.dose_history['virotherapy']['t'][0:3000], env.state.dose_history['virotherapy']['y'][0:3000])
    plt.xlabel('Temps (jours)')
    plt.ylabel('Virus')
    plt.savefig('outputs/Virotherapy_doses.png')
