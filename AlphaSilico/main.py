import numpy as np
import matplotlib.pyplot as plt
from AlphaSilico.src.insilico import ClinicalTrial, TumorModel
import scipy.io as sio
import pandas as pd

# virtual_patient_parameter_input = np.transpose(sio.loadmat('references/CodeEssaiClinique/16052019VirtualPopulation300PatientParameters.mat')['VirtualPatientParameters'])  # shape
# 254x302.

virtual_cohort = pd.read_csv('data/virtual_cohort.csv')

trial = ClinicalTrial(patients=virtual_cohort.to_numpy())

x=10


