# Standard
import sys

# Third Party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local
from AlphaSilico.src.insilico import ClinicalTrial, TumorModel
from AlphaSilico.src.learner import Learner

immunotherapy = np.ones(75)
virotherapy = np.ones(7)

virtual_cohort = pd.read_csv('data/virtual_cohort.csv')
trial = ClinicalTrial(patients=virtual_cohort.to_numpy(), immunotherapy=immunotherapy, virotherapy=virotherapy)
learner = Learner()


