import AlphaSilico.viral_oncology_fit as vof
from AlphaSilico.insilico import Params 
from scipy.integrate import ode
import numpy as np

def test_fit():
    P = Params()
    y0 = np.array(P.initial_conditions)
    print(y0)
    t0 = 0 # Time in months
    args = P.model_input()
    integrator = ode(vof.f)
    integrator.set_initial_value(y0, t0).set_f_params(args)
    tf = 3 # 3 month at the end
    dt = 0.1 #
    while r.succesful() and r.t < tf:
        print(r.t + dt, r.integrate(r.t + dt))
    assert 0 == 1
