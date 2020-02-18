import math
import numpy as np
import scipy.io as sio
from scipy.integrate import ode, cumtrapz


class ClinicalTrial:

    def __init__(self, patients, treatment_duration=3, viral_offset=7/30, immune_offset=1/30):
        """

        :param patients: (n, 7) ndarray. Rows correspond to patients and columns correspond to variable parameters.
        :param treatment_duration: Float. Length of treatment in months (3 months by default).
        :param viral_offset:  Float. Virotherapy offset (every 7 days by default).
        :param immune_offset: Float. Immunotherapy off set (every day by default).
        """

        self.treatment_duration = treatment_duration
        self.viral_offset = viral_offset
        self.immune_offset = immune_offset
        self.patients = patients
        self.tumors = [TumorModel(*patient) for patient in self.patients]


class TumorModel:

    # Constants
    intermitotic_SD = 6.7 / 24 / 30
    kappa = 3.534412642851458 * 30  # Virion contact rate
    delta = 4.962123414821151 * 30  # Lysis rate
    alpha = 0.008289097649957  # Lytic virion release rate
    omega = 9.686308020782763 * 30  # Virion death rate
    eta12 = 0.510538277701167  # Virion half effect concentration
    C_prod_homeo = 0.00039863 * 30  # Homeostatic cytokine production rate
    C_prod_max = 1.429574637713578 * 30  # Maximal cytokine production rate
    C12 = 0.739376299393775 * 30  # Maximal cytokine production rate
    k_elim = 0.16139 * 30  # Cytokine elimination rate
    P12 = 5  # Cytokine production half effect
    gamma_P = 0.35 * 30  # From Barrish 2017 PNAS elimination rate of phagocyte
    C_star = C_prod_homeo / k_elim

    def __init__(self, a1=1.183658646441553*30, a2=1.758233712464858*30, d1=0, d2=0.539325116600707*30, kp=0.05*30, kq=10, k_cp=4.6754*30):

        # Variable patient parameters
        self.a1 = a1
        self.a2 = a2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d2
        self.tau = (33.7/24 - 30/a2)/30
        self.kp = kp
        self.kq = kq
        self.ks = kq
        self.k_cp = k_cp  # Maximal phagocyte production rate

        # Distribution specific parameters
        self.N = int(self.tau**2 / self.intermitotic_SD**2)
        self.transit_rate = self.N / self.tau  # Transit rate across compartments
        self.d3_hat = self.N / self.tau * (math.exp(self.d3 * self.tau / (self.N + 1)) - 1)
        self.d3_hat_R = self.d3_hat
        self.P_star = (1 / self.gamma_P) * (self.k_cp * self.C_star / (self.P12 + self.C_star))

        # Resistant parameters
        self.nu = 1e-10  # Mutation percentage
        self.a1_R = self.a1
        self.a2_R = self.a2
        self.d1_R = self.d1
        self.d2_R = self.d2
        self.d3_R = self.d3
        self.kappa_R = self.kappa

        # Cell cycle duration
        self.total_time = 1 / self.a1 + 1 / (self.a2 + self.d2) + self.tau
        self.total_cells = 200

        # Initial conditions
        QIC = (1 / a1 / self.total_time) * self.total_cells * (1 - self.nu)
        SIC = (1 / (a2 + d2) / self.total_time) * self.total_cells * (1 - self.nu)
        TCIC = (self.tau / self.total_time) * self.total_cells * (1 - self.nu) * np.ones(shape=self.N) / self.N
        NCIC = (self.tau / self.total_time) * self.total_cells * (1 - self.nu)
        IIC = 0
        VIC = 0
        CIC = self.C_prod_homeo / self.k_elim
        PIC = k_cp * CIC / ((self.P12 + CIC) * self.gamma_P)
        RIC = (1 / a1 / self.total_time) * self.total_cells * self.nu
        RSIC = (1 / (a2 + d2) / self.total_time) * self.total_cells * self.nu
        resistant_TCIC = (self.tau / self.total_time) * self.total_cells * self.nu * np.ones(shape=self.N) / self.N
        resistant_total_cells_IC = (self.tau / self.total_time) * self.total_cells * self.nu
        self.initial_conditions = [QIC, SIC, IIC, VIC] + TCIC.tolist() + [CIC, PIC, NCIC, RIC, RSIC] + resistant_TCIC.tolist() + [resistant_total_cells_IC]

    def f_evaluate(self, t, y):

        """
        :param t: Float. Current time value.
        :param y: 1D list. Previous solution.
        :return: 1D list. Derivative of each quantity in y at time step t.
        """

        infection = 0
        if y[3] > 1e-10:
            infection = y[3] / (self.eta12 + y[3])

        eta = self.kappa * infection
        psi_Q = y[self.N+5] * self.kp / (1 + self.kq * y[0])
        psi_S = y[self.N+5] * self.kp / (1 + self.ks * y[1])
        a = psi_S * y[1] + self.delta * y[2] + psi_Q * y[0]
        C_prod = self.C_prod_homeo + (self.C_prod_max - self.C_prod_homeo) * (a / self.C12 + a)

        # Quiescent cells, y[0]
        dQ_dt = 2 * (1 - self.nu) * self.transit_rate * y[self.N+3] - (self.a1 + self.d1 + psi_Q) * y[0]

        # G1 cells, y[1]
        dG1_dt = self.a1 * y[0] - (self.a2 + self.d2 + psi_S + eta) * y[1]

        # Infected cells, y[2]
        dI_dt = -self.delta * y[2] + eta * (y[1] + y[self.N+6] + y[self.N+8] + y[2*self.N+9])

        # Virions, y[3]
        dV_dt = self.alpha * self.delta * y[2] - self.omega * y[3] - eta * (y[1] + y[self.N+6] + y[self.N+8] + y[2*self.N+9])  # + ViralDose(PA,t)

        # First compartment, y[4]
        dA1_dt = self.a2 * y[1] - self.transit_rate * y[4] - (self.d3_hat + eta + psi_S) * y[4]

        # ODE for transit compartments, y[5], ..., y[N+3]
        dAi_dt = []
        for j in range(5, self.N+4):
            dAj_dt = self.transit_rate * (y[j-1] - y[j]) - (self.d3_hat + eta + psi_S * y[j])
            dAi_dt.append(dAj_dt)

        # Immune cytokine, y[N+4]
        dC_dt = C_prod - self.k_elim * y[self.N+4]  # + Dose(PA, t)

        # Phagocytes, y[N+5]
        dP_dt = self.k_cp * y[self.N+4] / (self.P12 + y[self.N+4]) - self.gamma_P * y[self.N+5]

        # Total number of cells in cell cycle, y[N+6]
        dN_dt = self.a2 * y[1] - (self.d3_hat + eta + psi_S) * y[self.N + 6] - (self.transit_rate / self.a2) * y[self.N+3]

        # Resistant quiescent cells, y[N+7]
        dQR_dt = 2 * self.nu * self.transit_rate * y[self.N+3] + 2 * self.transit_rate * y[2*self.N+8] - (self.a1_R + self.d1_R) * y[self.N+7]

        # Resistant G1 cells, y[N+8]
        dG1R_dt = self.a1_R * y[self.N+7] - (self.a2_R + self.d2_R + eta) * y[self.N+8]

        # Resistant first transit, y[N+9]
        dA1R_dt = self.a2_R * y[self.N+8] - self.transit_rate * y[self.N+9] - (self.d3_hat + eta * y[self.N+9])

        # DE for resistant transit compartments, y[N+10], ..., y[2*N+8]
        dAiR_dt = []
        for j in range(self.N+10, 2*self.N+9):
            dAjR_dt = self.transit_rate * (y[j - 1] - y[j]) - (self.d3_hat + eta * y[j])
            dAiR_dt.append(dAjR_dt)

        # Total number of resistant cells in cell cyle, y[2*N+9]
        dNR_dt = self.a2 * y[self.N+8] - (self.d3_hat + eta) * y[2*self.N+9] - (self.transit_rate / self.a2) * y[2*self.N+9]

        return [dQ_dt, dG1_dt, dI_dt, dV_dt, dA1_dt] + dAi_dt + [dC_dt, dP_dt, dN_dt, dQR_dt, dG1R_dt, dA1R_dt] + dAiR_dt + [dNR_dt]

    def simulate(self, t_start=0, t_end=3, dt=1/30):

        """

        :param t_start: Float. Time at the start of the simulation. Included.
        :param t_end:  Float. Time at the end of the simulation. Excluded.
        :param dt: Float. Time step.
        :return: Simulation history. Includes initial conditions as first entry.
        """

        r = ode(self.f_evaluate).set_integrator('zvode')  # Initialize the integrator
        r.set_initial_value(self.initial_conditions, t_start)  # Set initial conditions

        y = np.array(self.initial_conditions)  # Solution

        while r.successful() and r.t < t_end - dt:
            r.integrate(r.t + dt)
            y = np.vstack((y, np.real(r.y)))

        history = {'t_start': t_start,
                   't_end': t_end,
                   'dt': dt,
                   'y': y,
                   }

        return history

    def f_obective(self, history):

        # F(Dose) = Cumulative Tumor Burden + Cumulative Dose

        non_resistant_cycle = history['y'][:, 0] + history['y'][:, 1] + history['y'][:, self.N+6]  # Q, G1 and T
        resistant_cycle = history['y'][:, self.N+7] + history['y'][:, self.N+8] + history['y'][:, 2*self.N+9]  # QR, G1R and TR
        tumor_burden = non_resistant_cycle + resistant_cycle + history['y'][:, 2]

        cumulative_tumor_burden = cumtrapz(y=tumor_burden, x=None, dx=history['dt'])  # Cumulative integral of tumor size

        return cumulative_tumor_burden[-1]
