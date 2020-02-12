import numpy as np
from scipy.integrate import ode
from AlphaSilico.src.insilico import TumorModel


def f(t, y, model):

    """


    :param t: Float. Current time value.
    :param y: 1D list. Previous solution.
    :param model: TumorModel instance.
    :return: 1D list. Derivative of each quantity in y at time step t.
    """

    infection = 0
    if y[3] > 1e-10:
        infection = y[3] / (model.eta12 + y[3])

    eta = model.kappa * infection
    psi_Q = y[model.N+5] * model.kp / (1 + model.kq * y[0])
    psi_S = y[model.N+5] * model.kp / (1 + model.ks * y[1])
    a = psi_S * y[1] + model.delta * y[2] + psi_Q * y[0]
    C_prod = model.C_prod_homeo + (model.C_prod_max - model.C_prod_homeo) * (a / model.C12 + a)

    # Quiescent cells, y[0]
    dQ_dt = 2 * (1 - model.nu) * model.transit_rate * y[model.N+3] - (model.a1 + model.d1 + psi_Q) * y[0]

    # G1 cells, y[1]
    dG1_dt = model.a1 * y[0] - (model.a2 + model.d2 + psi_S + eta) * y[1]

    # Infected cells, y[2]
    dI_dt = -model.delta * y[2] + eta * (y[1] + y[model.N+6] + y[model.N+8] + y[2*model.N+9])

    # Virions, y[3]
    dV_dt = model.alpha * model.delta * y[2] - model.omega * y[3] - eta * (y[1] + y[model.N+6] + y[model.N+8] + y[2*model.N+9])  # + ViralDose(PA,t)

    # First compartment, y[4]
    dA1_dt = model.a2 * y[1] - model.transit_rate * y[4] - (model.d3_hat + eta + psi_S) * y[4]

    # ODE for first compartment, y[5], ..., y[N+3]
    dAi_dt = []
    for j in range(5, model.N+4):
        dAj_dt = model.transit_rate * (y[j-1] - y[j]) - (model.d3_hat + eta + psi_S * y[j])
        dAi_dt.append(dAj_dt)

    # Immune cytokine, y[N+4]
    dC_dt = C_prod - model.k_elim * y[model.N+4]  # + Dose(PA, t)

    # Phagocytes, y[N+5]
    dP_dt = model.Kcp * y[model.N+4] / (model.P12 + y[model.N + 4]) - model.gamma_P * y[model.N+5]

    # ODE for total number of cells in cell cycle, y[N+6]
    dT_dt = model.a2 * y[1] - (model.d3_hat + eta + psi_S) * y[model.N + 6] - (model.transit_rate / model.a2) * y[model.N + 3]

    # Resistant quiescent cells, y[N+7]
    dQR_dt = 2 * model.nu * model.transit_rate * y[model.N+3] + 2 * model.transit_rate * y[2*model.N+8] - (model.a1_R + model.d1_R) * y[model.N+7]

    # Resistant G1 cells, y[N+8]
    dG1R_dt = model.a1_R * y[model.N+7] - (model.a2_R + model.d2_R + eta) * y[model.N+8]

    # Resistant first transit, y[N+9]
    dA1R_dt = model.a2_R * y[model.N+8] - model.transit_rate * y[model.N+9] - (model.d3_hat + eta * y[model.N+9])

    # DE for resistant first transit, y[N+10], ..., y[2*N+8]
    dAiR_dt = []
    for j in range(model.N+10, 2*model.N+9):
        dAjR_dt = model.transit_rate * (y[j - 1] - y[j]) - (model.d3_hat + eta * y[j])
        dAiR_dt.append(dAjR_dt)

    # DE for total resistant cells, y[2*N+9]
    dTR_dt = model.a2 * y[model.N+8] - (model.d3_hat + eta) * y[2*model.N+9] - (model.transit_rate / model.a2) * y[2*model.N+9]

    return [dQ_dt, dG1_dt, dI_dt, dV_dt, dA1_dt] + dAi_dt + [dC_dt, dP_dt, dT_dt, dQR_dt, dG1R_dt, dA1R_dt] + dAiR_dt + [dTR_dt]


def simulate(t_start, t_end, dt):

    """

    :param t_start: Float. Time at the start of the simulation.
    :param t_end:  Float. Time at the end of the simulation.
    :param dt: Float. Time step.
    :return: Simulation history.
    """

    r = ode(f).set_integrator('zvode')  # Initialize the integrator
    r.set_initial_value(TumorModel.initial_conditions, t_start).set_f_params(TumorModel)  # Set initial conditions and model args

    history = np.array(TumorModel.initial_conditions)

    while r.successful() and r.t < t_end:

        r.integrate(r.t+dt)
        history = np.vstack((history, np.real(r.y)))

    return history



