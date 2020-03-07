
# Third Party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local
from AlphaSilico.src.insilico import ClinicalTrial, TumorModel
from AlphaSilico.src.learner import Learner

# virtual_cohort = pd.read_csv('data/virtual_cohort.csv')
# trial = ClinicalTrial(patients=virtual_cohort.to_numpy(), viral_treatment=viral_treatment, immune_treatment=immune_treatment)
# learner = Learner()

# Number of doses per day of treatment.
immunotherapy = np.ones(75) * np.random.randint(0, 5, size=75)
virotherapy = np.ones(10) * np.random.randint(1, 5, size=10)

# Total length of treatment
treatment_start_time = 0
treatment_total_time = 3.5

# Models
# Treated tumor
tumor = TumorModel(immunotherapy, virotherapy, treatment_start_time=treatment_start_time)
history = tumor.simulate(t_start=0, t_end=treatment_total_time)  # Run through a simulation and get the history
tumor_size, cumulative_tumor_burden = tumor.evaluate_obective(history)  # Compute the objective function from the history
# Untreated tumor, control group
control = TumorModel(immunotherapy * 0, virotherapy * 0, treatment_start_time=treatment_start_time)
control_history = control.simulate(t_start=0, t_end=treatment_total_time)
control_size, control_ctb = control.evaluate_obective(control_history)

print('Plotting...')
titles = ['Quiescent cells', 'G1 cells', 'Infected cells', 'Virions'] + \
         ['Transit compartment ' + str(n+1) for n in range(tumor.j)] + \
         ['Cytokines', 'Phagocytes', 'Total number of cells in cycle', 'Resistant quiescent cells', 'Resistant G1 cells'] + \
         ['Resistant transit compartment ' + str(n+1) for n in range(tumor.j)] + \
         ['Total number of resistant cells in cycle']

# Print tracked quantities
for idx, quantity in enumerate(history['y'].transpose()):
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
plt.plot(tumor.dose_history['immunotherapy']['t'], tumor.dose_history['immunotherapy']['h'])
plt.title('Immunotherapy vs time')
plt.savefig('outputs/Immunotherapy_doses.png')

plt.figure()
plt.plot(tumor.dose_history['virotherapy']['t'], tumor.dose_history['virotherapy']['h'])
plt.title('Virotherapy vs time')
plt.savefig('outputs/Virotherapy_doses.png')
