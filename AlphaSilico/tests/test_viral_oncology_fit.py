import AlphaSilico.viral_oncology_fit as vof
from AlphaSilico.insilico import Params as P
from scipy.integrate import ode

def test_fit():
    y0 = 
    t0 = 0 # Time in months
    args = P.model_input()
    integrator = ode(vof.f)
    integrator.set_initial_value(y0, t0).set_f_params(args)
    assert 0 == 1
