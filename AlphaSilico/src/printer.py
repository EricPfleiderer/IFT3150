# Third Party
import numpy as np
import matplotlib.pyplot as plt

# Local
from AlphaSilico.src.insilico import Environment

# Number of doses per day of treatment.

min_dose = 0
max_dose = 4
treatment_start_time = 0
treatment_len = 2.5
observation_len = 3

# Random treatment
treatment = np.transpose(np.array([np.random.randint(min_dose, max_dose+1, size=int(treatment_len*30)), np.random.randint(min_dose, max_dose+1, size=int(treatment_len*30))]))

# Models
# Treated tumor
env = Environment()
for day in range(int(observation_len*30)):
    actions = (0, 0)
    if day < int(treatment_len*30):
        actions = (treatment[day][0], treatment[day][1])
    env.step(actions)

tumor_size, cumulative_tumor_burden = env.evaluate_obective()  # Compute the objective function from the history

# Untreated tumor, control group
control = Environment()
for _ in range(int(observation_len*30)):
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
for idx, quantity in enumerate(env.history.transpose()):
    plt.figure()
    plt.plot(np.arange(0, quantity.size), quantity)
    plt.title(titles[idx])
    plt.savefig('outputs/' + str(idx) + '_' + titles[idx] + '.png')

# Print main metric (tumor size)
plt.figure()
plt.plot(np.arange(0, tumor_size.size), tumor_size, '--', label='Tumor size, test')
plt.plot(np.arange(0, cumulative_tumor_burden.size), cumulative_tumor_burden, '--', label='Cumulative burden, test')
plt.plot(np.arange(0, control_size.size), control_size, '-', label='Tumor size, control')
plt.plot(np.arange(0, control_ctb.size), control_ctb, '-', label='Cumulative burden, control')
plt.title('Tumor growth')
plt.legend()
plt.savefig('outputs/Tumor size.png')

# Print dosages vs time
plt.figure()
plt.plot(env.state.dose_history['immunotherapy']['t'][0:2500], env.state.dose_history['immunotherapy']['y'][0:2500])
plt.title('Immunotherapy vs time')
plt.savefig('outputs/Immunotherapy_doses.png')

plt.figure()
plt.plot(env.state.dose_history['virotherapy']['t'], env.state.dose_history['virotherapy']['y'])
plt.title('Virotherapy vs time')
plt.savefig('outputs/Virotherapy_doses.png')
