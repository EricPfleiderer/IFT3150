import numpy as np
from scipy.integrate import ode
from AlphaSilico.insilico import Params


def f(t, y, a1, a1_R, a2, a2_R, alpha, C12, C_prod_homeo, C_prod_max, d1, d1_R, d2, d2_R, d3_hat, delta, eta12, gamma_P, kappa, Kcp, k_elim, kp, kq, ks, N, nu, omega, P12,
      transit_rate):

    """
    :param t: Time step.
    :param y: [Q, G1, I, V, A1, Ai, C, P, T, QR, G1R, AiR, TR], where Ai and AiR are multiple sequential entries
    :return: Derivative of each quantity in y at time step t.
    """

    infection = 0
    if y[3] > 1e-10:
        infection = y[3] / (eta12 + y[3])

    eta = kappa * infection
    psi_Q = y[N+5] * kp / (1 + kq * y[0])
    psi_S = y[N+5] * kp / (1 + ks * y[1])
    a = psi_S * y[1] + delta * y[2] + psi_Q * y[0]
    C_prod = C_prod_homeo + (C_prod_max - C_prod_homeo) * (a / C12 + a)

    # Quiescent cells, y[0]
    dQ_dt = 2 * (1 - nu) * transit_rate * y[N+3] - (a1 + d1 + psi_Q) * y[0]

    # G1 cells, y[1]
    dG1_dt = a1 * y[0] - (a2 + d2 + psi_S + eta) * y[1]

    # Infected cells, y[2]
    dI_dt = -delta * y[2] + eta * (y[1] + y[N+6] + y[N+8] + y[2*N+9])

    # Virions, y[3]
    dV_dt = alpha * delta * y[2] - omega * y[3] - eta * (y[1] + y[N+6] + y[N+8] + y[2*N+9])  # + ViralDose(PA,t)

    # First compartment, y[4]
    dA1_dt = a2 * y[1] - transit_rate * y[4] - (d3_hat + eta + psi_S) * y[4]

    # ODE for first compartment, y[5], ..., y[N+3]
    dAi_dt = []
    for j in range(5, N+4):
        dAj_dt = transit_rate * (y[j-1] - y[j]) - (d3_hat + eta + psi_S * y[j])
        dAi_dt.append(dAj_dt)

    # Immune cytokine, y[N+4]
    dC_dt = C_prod - k_elim * y[N+4]  # + Dose(PA, t)

    # Phagocytes, y[N+5]
    dP_dt = Kcp * y[N+4] / (P12 + y[N + 4]) - gamma_P * y[N+5]

    # ODE for total number of cells in cell cycle, y[N+6]
    dT_dt = a2 * y[1] - (d3_hat + eta + psi_S) * y[N + 6] - (transit_rate / a2) * y[N + 3]

    # Resistant quiescent cells, y[N+7]
    dQR_dt = 2 * nu * transit_rate * y[N+3] + 2 * transit_rate * y[2*N+8] - (a1_R + d1_R) * y[N+7]

    # Resistant G1 cells, y[N+8]
    dG1R_dt = a1_R * y[N+7] - (a2_R + d2_R + eta) * y[N+8]

    # Resistant first transit, y[N+9]
    dA1R_dt = a2_R * y[N+8] - transit_rate * y[N+9] - (d3_hat + eta * y[N+9])

    # DE for resistant first transit, y[N+10], ..., y[2*N+8]
    dAiR_dt = []
    for j in range(N+10, 2*N+9):
        dAjR_dt = transit_rate * (y[j - 1] - y[j]) - (d3_hat + eta * y[j])
        dAiR_dt.append(dAjR_dt)

    # DE for total resistant cells, y[2*N+9]
    dTR_dt = a2 * y[N+8] - (d3_hat + eta) * y[2*N+9] - (transit_rate / a2) * y[2*N+9]

    return [dQ_dt, dG1_dt, dI_dt, dV_dt, dA1_dt] + dAi_dt + [dC_dt, dP_dt, dT_dt, dQR_dt, dG1R_dt, dA1R_dt] + dAiR_dt + [dTR_dt]


