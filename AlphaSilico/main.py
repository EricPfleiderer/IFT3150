
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

# Number of doses per day of treatment
immunotherapy = np.ones(75)
# immunotherapy = np.zeros(75)
# immunotherapy[np.arange(immunotherapy.size) % 7 == 0] = 1
virotherapy = np.ones(10)

# Model
tumor = TumorModel(immunotherapy, virotherapy)
history = tumor.simulate(t_start=0, t_end=3, nsteps=1000)  # Run through a simulation and get the history
tumor_size, cumulative_tumor_burden = tumor.evaluate_obective(history)  # Compute the objective function from the history

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
plt.plot(np.arange(0, tumor_size.size), tumor_size)
plt.title('Tumor size')
plt.savefig('outputs/Tumor size.png')



