import numpy as np
import matplotlib.pyplot as plt
from AlphaSilico.src.insilico import TumorModel

model = TumorModel()

t0 = 0
t1 = 10
dt = 0.1

time = np.arange(t0, t1 + dt, dt)

history = model.simulate(t0, t1, dt)

for col in range(history.shape[1]):
    plt.figure()
    plt.plot(time, history[:, col])
    plt.savefig('outputs/history'+str(col)+'.png')
