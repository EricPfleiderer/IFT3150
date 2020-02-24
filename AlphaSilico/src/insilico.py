import math
import numpy as np
import scipy.io as sio
from scipy.integrate import ode, cumtrapz


class ClinicalTrial:

    def __init__(self, patients, viral_treatment, immune_treatment, viral_offset=7/30, immune_offset=1/30):
        """
        :param patients: (n, 7) ndarray. Rows correspond to patients and columns correspond to variable parameters.
        :param viral_offset:  Float. Virotherapy offset (every 7 days by default).
        :param immune_offset: Float. Immunotherapy off set (every day by default).
        """

        self.viral_offset = viral_offset
        self.immune_offset = immune_offset
        self.patients = patients
        self.tumors = [TumorModel(*patient) for patient in self.patients]


class TumorModel:

    vol = 7

    # Immunotherapy
    immune_offset = 1/30  # Offset
    immune_admin = 1 * 125000  # Cytokine per unit volume
    immune_kabs = 6.6311 * 30  # Absorbtion rate
    immune_availability = 0.85  # Bioavailability

    # Virotherapy
    viral_offset = 7 / 30
    viral_admin = 250
    viral_kabs = 20 * 30  # Absorbtion rate
    viral_availability = 1

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

    def __init__(self, immunotherapy_doses, virotherapy_doses, a1=1.183658646441553*30, a2=1.758233712464858*30, d1=0, d2=0.539325116600707*30,
                 kp=0.05*30, kq=10, k_cp=4.6754*30):

        self.immunotherapy_doses = immunotherapy_doses
        self.virotherapy_doses = virotherapy_doses

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
        self.k_tr = self.N / self.tau  # Transit rate across compartments
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
        QIC = (1 / a1 / self.total_time) * self.total_cells * (1 - self.nu)  # Quiescent
        SIC = (1 / (a2 + d2) / self.total_time) * self.total_cells * (1 - self.nu)  # G1
        TCIC = (self.tau / self.total_time) * self.total_cells * (1 - self.nu) * np.ones(shape=self.N) / self.N  # Transit
        NCIC = (self.tau / self.total_time) * self.total_cells * (1 - self.nu)  # Total number of cells in cycle
        IIC = 0  # Infected cells
        VIC = 0  # Virions
        CIC = self.C_prod_homeo / self.k_elim  # Cytokines
        PIC = k_cp * CIC / ((self.P12 + CIC) * self.gamma_P)  # Phagocytes
        RIC = (1 / a1 / self.total_time) * self.total_cells * self.nu  # Resistant quiescent
        RSIC = (1 / (a2 + d2) / self.total_time) * self.total_cells * self.nu  # Resistant G1
        resistant_TCIC = (self.tau / self.total_time) * self.total_cells * self.nu * np.ones(shape=self.N) / self.N  # Resistant transit
        resistant_total_cells_IC = (self.tau / self.total_time) * self.total_cells * self.nu  # Total number resistant cells in cyle
        self.initial_conditions = [QIC, SIC, IIC, VIC] + TCIC.tolist() + [CIC, PIC, NCIC, RIC, RSIC] + resistant_TCIC.tolist() + [resistant_total_cells_IC]

    # [QIC, SIC, IIC, VIC, TCIC, CIC, PIC, NCIC, RIC, RSIC, ResistantTCIC, ResistantTotalCellsIC]

    def immune_dose(self, t):
        if int(t*30) % int(self.immune_offset*30) == 0:
            dose = self.immune_admin * self.immunotherapy_doses[int(t / self.immune_offset)]
            return self.immune_availability * self.immune_kabs * dose / self.vol * math.exp(self.immune_kabs * t)
        return 0

    def viral_dose(self, t):
        if int(t*30) % int(self.viral_offset*30) == 0:
            dose = self.viral_admin * self.virotherapy_doses[int(t / self.viral_offset)]
            return self.viral_kabs * dose * self.viral_availability / self.vol * math.exp(self.viral_kabs*t)
        return 0

    def evaluate_derivatives(self, t, y):
        
        # N ambiguity (total cells in cycle)

        """
        :param t: Float. Current time value.
        :param y: 1D list. Previous solution.
        :return: 1D list. Derivative of each quantity in y at time step t.
        """
        Q, G1, I, V, A1, Ai, C, P, N, QR, G1R, A1R, AiR, NR = y[0], y[1], y[2], y[3], y[4], y[5:self.N+3], y[self.N+4],\
                                                              y[self.N+5], y[self.N+6], y[self.N+7], y[self.N+8], \
                                                              y[self.N+9], y[self.N+10:2*self.N+8], y[2*self.N+9]

        infection = 0
        if y[3] > 1e-10:
            infection = y[3] / (self.eta12 + y[3])

        eta = self.kappa * infection
        psi_Q = P * self.kp / (1 + self.kq * Q)
        psi_G = P * self.kp / (1 + self.ks * G1)
        a = psi_G * G1 + self.delta * I + psi_Q * Q
        C_prod = self.C_prod_homeo + (self.C_prod_max - self.C_prod_homeo) * (a / self.C12 + a)

        # Quiescent cells, y[0]
        dQ_dt = 2 * (1 - self.nu) * self.k_tr * Ai[-1] - (self.a1 + self.d1 + psi_Q) * Q

        # G1 cells, y[1]
        dG1_dt = self.a1 * Q - (self.a2 + self.d2 + psi_G + eta) * G1

        # Infected cells, y[2]
        dI_dt = -self.delta * I + eta * (G1 + N)

        # Virions, y[3]
        dV_dt = self.alpha * self.delta * I - self.omega * V - eta * (G1 + N)  # + self.viral_dose(t)

        # First compartment, y[4]
        dA1_dt = self.a2 * G1 - self.k_tr * A1 - (self.d3_hat + eta + psi_G) * A1

        # ODE for transit compartments, y[5], ..., y[N+3]
        dAi_dt = []
        for j in range(5, self.N+4):
            dAj_dt = self.k_tr * (y[j-1] - y[j]) - (self.d3_hat + eta + psi_G) * y[j]
            dAi_dt.append(dAj_dt)

        # Immune cytokine, y[N+4]
        dC_dt = C_prod - self.k_elim * C  # + self.immune_dose(t)

        # Phagocytes, y[N+5]
        dP_dt = self.k_cp * C / (self.C12 + C) - self.gamma_P * P  # DIFFERENCE BETWEEN MATLAB (P12) CODE AND S.I (C12).

        # Total number of cells in cycle, y[N+6]
        dN_dt = self.a2 * G1 - (self.d3_hat + eta + psi_G) * N - (self.k_tr / self.a2) * Ai[-1]  # Doesnot appear in SI

        # Resistant quiescent cells, y[N+7]
        dQR_dt = 2 * self.nu * self.k_tr * Ai[-1] + 2 * self.k_tr * AiR[-1] - (self.a1_R + self.d1_R) * QR

        # Resistant G1 cells, y[N+8]
        dG1R_dt = self.a1_R * QR - (self.a2_R + self.d2_R + eta) * G1R

        # Resistant first transit, y[N+9]
        dA1R_dt = self.a2_R * G1R - self.k_tr * A1R - (self.d3_hat + eta + psi_G) * A1R  # DIFFERENCE

        # DE for resistant transit compartments, y[N+10], ..., y[2*N+8]
        dAiR_dt = []
        for j in range(self.N+10, 2*self.N+9):
            dAjR_dt = self.k_tr * (y[j - 1] - y[j]) - (self.d3_hat + eta + psi_G) * y[j]  # DIFFERENCE
            dAiR_dt.append(dAjR_dt)

        # Total number of resistant cells in cyle, y[2*N+9]
        dNR_dt = self.a2 * G1R - (self.d3_hat + eta) * NR - (self.k_tr / self.a2) * y[2*self.N+9]  # DOES NOT APPEAR IN SI

        return [dQ_dt, dG1_dt, dI_dt, dV_dt, dA1_dt] + dAi_dt + [dC_dt, dP_dt, dN_dt, dQR_dt, dG1R_dt, dA1R_dt] + dAiR_dt + [dNR_dt]

    def simulate(self, t_start=0, t_end=3, dt=1/30, nsteps=10000):

        """
        :param t_start: Float. Time at the start of the simulation. Included.
        :param t_end:  Float. Time at the end of the simulation. Excluded.
        :param dt: Float. Time step.
        :param nsteps: Int. Max number of steps for a single call of the ode solver.
        :return: Simulation history. Includes initial conditions as first entry.
        """

        r = ode(self.evaluate_derivatives).set_integrator('vode', method='bdf', nsteps=nsteps)
        # r = ode(self.evaluate_derivatives).set_integrator('Isoda', nsteps=nsteps) #  Initialize the integrator
        r.set_initial_value(self.initial_conditions, t_start)  # Set initial conditions

        y = np.empty(shape=(0, len(self.initial_conditions)))  # Solution

        while r.successful() and r.t < t_end:
            print('Simulating... day ' + str(int(r.t*30)))
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
