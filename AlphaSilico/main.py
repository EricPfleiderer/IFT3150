import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from AlphaSilico.src.insilico import ClinicalTrial, TumorModel
from AlphaSilico.src.learner import Learner
import tensorflow as tf

# virtual_patient_parameter_input = np.transpose(sio.loadmat('references/CodeEssaiClinique/16052019VirtualPopulation300PatientParameters.mat')['VirtualPatientParameters'])  # shape
# 254x302.

# immune_treatment = np.random.randint(0, 5, size=90)
# viral_treatment = np.random.randint(0, 5, size=90//7)
# virtual_cohort = pd.read_csv('data/virtual_cohort.csv')
# trial = ClinicalTrial(patients=virtual_cohort.to_numpy(), viral_treatment=viral_treatment, immune_treatment=immune_treatment)
# learner = Learner()

# Number of doses per day of treatment
immunotherapy = [1] * 100
virotherapy = [1] * 100

tumor = TumorModel(immunotherapy, virotherapy)

titles = ['Quiescent cells', 'G1 cells', 'Infected cells', 'Virions'] + \
         ['Transit compartment ' + str(n+1) for n in range(tumor.N)] + \
         ['Cytokines', 'Phagocytes', 'Total number of cells in cycle', 'Resistant quiescent cells', 'Resistant G1 cells'] + \
         ['Resistant transit compartment ' + str(n+1) for n in range(tumor.N)] + \
         ['Total number of cells in cyle']

history = tumor.simulate(t_start=0, t_end=1/2, dt=1/30, nsteps=1000)

print('Plotting...')
for idx, quantity in enumerate(history['y'].transpose()):
    plt.figure()
    plt.plot(np.arange(0, quantity.size), quantity)
    plt.title(titles[idx])
    plt.savefig('outputs/' + str(idx) + '_' + titles[idx] + '.png')



