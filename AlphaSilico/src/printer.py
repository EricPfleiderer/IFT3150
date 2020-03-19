# Third Party
import numpy as np
import matplotlib.pyplot as plt

# Local
from AlphaSilico.src.insilico import Environment

# Number of doses per day of treatment.
min_doses = 0
max_doses = 4
treatment_start = 0
treatment_len = 75  # Treatment length in months
observation_len = 90  # Observation period length, including treatment

# Random treatment
treatment = np.transpose(np.array([np.random.randint(min_doses, max_doses+1, size=int(treatment_len)), np.random.randint(min_doses, max_doses+1, size=int(treatment_len))]))

# Models
# Treated tumor
env = Environment(treatment_len=treatment_len, observation_len=observation_len, treatment_start=treatment_start)
for day in range(observation_len):
    actions = (0, 0)
    if day < int(treatment_len):
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
    plt.plot(np.arange(0, observation_len, env.dt), quantity)
    plt.title(titles[idx])
    plt.savefig('outputs/' + str(idx) + '_' + titles[idx] + '.png')

# Print main metric (tumor size)
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Tumor size (number of cells)')
ax1.plot(np.arange(0, observation_len, env.dt), tumor_size, '--', c=(0, 0.6, 0), label='Tumor size, test')
ax1.plot(np.arange(0, observation_len, control.dt), control_size, '-', c=(0.2, 0, 0), label='Tumor size, control')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.set_ylabel('Cumulative burden')
ax2.plot(np.arange(0, observation_len - env.dt, env.dt), cumulative_tumor_burden, '--', c=(0, 0.8, 0), label='Cumulative burden, test')
ax2.plot(np.arange(0, observation_len - control.dt, control.dt), control_ctb, '-', c=(0.8, 0, 0), label='Cumulative burden, control')
ax2.tick_params(axis='y')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # otherwise the right y-label is slightly clipped
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
