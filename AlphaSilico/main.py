import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from AlphaSilico.src.insilico import ClinicalTrial, TumorModel
from AlphaSilico.src.learner import Model
import tensorflow as tf

# virtual_patient_parameter_input = np.transpose(sio.loadmat('references/CodeEssaiClinique/16052019VirtualPopulation300PatientParameters.mat')['VirtualPatientParameters'])  # shape
# 254x302.

virtual_cohort = pd.read_csv('data/virtual_cohort.csv')

immune_treatment = np.random.randint(0, 5, size=90)
viral_treatment = np.random.randint(0, 5, size=90//7)

trial = ClinicalTrial(patients=virtual_cohort.to_numpy(), viral_treatment=viral_treatment, immune_treatment=immune_treatment)

learner = Model()

