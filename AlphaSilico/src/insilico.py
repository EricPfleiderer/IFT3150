import math
import numpy as np
from scipy.integrate import ode, cumtrapz


class ClinicalTrial:

    def __init__(self, patients, viral_treatment, immune_treatment, viral_offset=7/30, immune_offset=1/30):
        """
        :param patients: (n, 7) ndarray. Rows correspond to patients and columns correspond to variable parameters.
        :param viral_offset:  Float. Virotherapy offset (every 7 days by default).
        :param immune_offset: Float. Immunotherapy off set (every day by default).
        """

        self.immune_treatment = immune_treatment
        self.viral_treatment = viral_treatment

        self.viral_offset = viral_offset
        self.immune_offset = immune_offset
        self.patients = patients
        self.tumors = [TumorModel(*patient) for patient in self.patients]


class TumorModel:

    vol = 7  # DOSE VOLUME ??

    # Immunotherapy
    immune_offset = 1/30  # Offset
    immune_admin = 125000  # Cytokine per unit volume
    immune_k_abs = 6.6311 * 30  # Absorbtion rate
    immune_availability = 0.85  # Bioavailability

    # Virotherapy
    viral_offset = 7 / 30
    viral_admin = 250  # Viral load
    viral_k_abs = 20 * 30  # Absorbtion rate
    viral_availability = 1  # Bioavailability
    kappa = 3.534412642851458 * 30  # Virion contact rate
    delta = 4.962123414821151 * 30  # Lysis rate
    alpha = 0.008289097649957  # Lytic virion release rate
    omega = 9.686308020782763 * 30  # Virion death rate
    eta12 = 0.510538277701167  # Virion half effect concentration

    # Cytokine parameters
    C_prod_homeo = 0.00039863 * 30  # Homeostatic cytokine production rate
    C_prod_max = 1.429574637713578 * 30  # Maximal cytokine production rate
    C12 = 0.739376299393775 * 30  # Maximal cytokine production rate
    k_elim = 0.16139 * 30  # Cytokine elimination rate

    # Constants
    intermitotic_SD = 6.7 / 24 / 30
    PSI12 = 5 * 30  # Cytokine production half effect  # MISSING * 30 IN MATLAB CODE ???
    gamma_P = 0.35 * 30  # From Barrish 2017 PNAS elimination rate of phagocyte  # CLEARANCE RATE > 1 ??? P STRONGLY LIMITED

    def __init__(self, immunotherapy, virotherapy, a1=1.183658646441553*30, a2=1.758233712464858*30, d1=0, d2=0.539325116600707*30,
                 kp=0.05*30, kq=10, k_cp=4.6754*30):

        """
        Initializes a system of ordinary differential equations to model melanoma tumor growth. Model by Craig & Cassidy.

        :param immunotherapy: 1D Numpy array of floats. Each entry corresponds to the number of immunotherapy doses at a given day.
        :param virotherapy: 1D Numpy array of floats. Each entry corresponds to the number of virotherapy doses at a given day.
        :param a1: Float. Quiescent to interphase rate. (1/month)
        :param a2: Float. Interphase to active phase rate. (1/month)
        :param d1: Float. Quiescent death rate. (1/month)
        :param d2: Float. Interphase death rate. (1/month)
        :param kp: Float. Phagocyte-tumor cell contact rate. (1/month)
        :param kq: Float. Phagocyte cell digestion constant.
        :param k_cp: Float. Maximal phagocyte production rate (10**10 cells/month)
        """

        self.t = 0  # Current time in months
        self.dt = 1/30  # Time step (1 day)

        # Treatment plan
        self.immunotherapy = immunotherapy
        self.virotherapy = virotherapy
        self.t_immune_admin = np.arange(immunotherapy.size) * self.immune_offset
        self.t_viral_admin = np.arange(virotherapy.size) * self.viral_offset

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
        self.k_cp = k_cp

        # Distribution specific parameters
        self.j = round(self.tau**2 / self.intermitotic_SD**2)  # Number of transit compartments
        self.k_tr = self.j / self.tau  # Transit rate across compartments
        self.dg_hat = self.j / self.tau * (math.exp(self.d3 * self.tau / (self.j + 1)) - 1)  # 14.8948 VS SUGGESTED 0.167 * 30 = 5.01 in S.I. GREATLY AFFECTS SCALE
        # self.dg_hat = 0.167 * 30  # Suggested by S.I.
        self.dg_hat_R = self.dg_hat

        # Immune steady state (not used?)
        self.C_star = self.C_prod_homeo / self.k_elim
        self.P_star = (1 / self.gamma_P) * (self.k_cp * self.C_star / (self.PSI12 + self.C_star))

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
        Q = (1 / a1 / self.total_time) * self.total_cells * (1 - self.nu)  # Quiescent
        G1 = (1 / (a2 + d2) / self.total_time) * self.total_cells * (1 - self.nu)  # G1
        I = 0  # Infected cells
        V = 0  # Virions
        A = (self.tau / self.total_time) * self.total_cells * (1 - self.nu) * np.ones(self.j) / self.j  # Transit compartments (1 to N)
        N = (self.tau / self.total_time) * self.total_cells * (1 - self.nu)  # Total number of cells in cycle
        C = self.C_prod_homeo / self.k_elim  # Cytokines
        P = self.k_cp * C / ((self.PSI12 + C) * self.gamma_P)  # Phagocytes
        QR = (1 / a1 / self.total_time) * self.total_cells * self.nu  # Resistant quiescent
        G1R = (1 / (a2 + d2) / self.total_time) * self.total_cells * self.nu  # Resistant G1
        AR = (self.tau / self.total_time) * self.total_cells * self.nu * np.ones(self.j) / self.j  # Resistant transitcompartments (1 to N)
        NR = (self.tau / self.total_time) * self.total_cells * self.nu  # Total number resistant cells in cycle
        self.initial_conditions = [Q, G1, I, V] + A.tolist() + [C, P, N, QR, G1R] + AR.tolist() + [NR]  # Length 28 with N = 9

        self.dose_history = {'immunotherapy': {'t': [],
                                               'h': [],
                                               },
                             'virotherapy': {'t': [],
                                             'h': [],
                                             }
                             }

        self.test_counter = 0

    def immune_dose(self, t):

        """
        TO DO:
            - Bug when tracking immunotherapy through time
        :param t: Float. Current time in months.
        :return: Float. Current instantaneous change in immunotherapy.
        """

        time_mask = np.where(t >= self.t_immune_admin)  # Mask doses that have not yet been applied

        immunotherapy = self.immunotherapy[time_mask]
        t_admin = self.t_immune_admin[time_mask]

        doses = self.immune_k_abs * self.immune_availability * self.immune_admin * immunotherapy  # Convert doses to available cytokines
        decay = np.exp(-self.immune_k_abs * (t - t_admin))  # Exponential decay term

        if (len(self.dose_history['immunotherapy']['t']) == 0 and len(self.dose_history['immunotherapy']['h']) == 0) or self.dose_history['immunotherapy']['t'][-1] != t:
            self.dose_history['immunotherapy']['t'].append(t)
            self.dose_history['immunotherapy']['h'].append(np.sum(doses*decay)/self.vol)

        return np.sum(doses*decay)/self.vol

    def viral_dose(self, t):
        time_mask = np.where(t >= self.t_viral_admin)  # Mask doses that have not yet been applied

        virotherapy = self.virotherapy[time_mask]
        t_admin = self.t_viral_admin[time_mask]

        doses = self.viral_k_abs * self.viral_availability * self.viral_admin * virotherapy  # Convert doses to available cytokines
        decay = np.exp(-self.viral_k_abs * (t - t_admin))  # Exponential decay term

        if (len(self.dose_history['virotherapy']['t']) == 0 and len(self.dose_history['virotherapy']['h']) == 0) or self.dose_history['virotherapy']['t'][-1] != t:
            self.dose_history['virotherapy']['t'].append(t)
            self.dose_history['virotherapy']['h'].append(np.sum(doses*decay)/self.vol)

        return np.sum(doses * decay) / self.vol

    def evaluate_derivatives(self, t, y):

        """
        :param t: Float. Current time value.
        :param y: 1D list. Previous solution.
        :return: 1D list. Derivative of each quantity in y at time step t.
        """

        Q, G1, I, V, A, C, P, N, QR, G1R, AR, NR = y[0], y[1], y[2], y[3], y[4:self.j+4], y[self.j+4], y[self.j+5], y[self.j+6], y[self.j+7], y[self.j+8], \
                                                   y[self.j+9:2*self.j+9], y[2*self.j+9]

        # Auxiliary function
        infection = 0
        if V > 1e-10:
            infection = V / (self.eta12 + V)
        eta = self.kappa * infection
        phi = self.k_cp * C / (self.C12 + C)  # DIFFERENCE BETWEEN MATLAB (PSI12) CODE AND S.I (PSI12 VS C12).
        psi_Q = self.kp * P / (1 + self.kq * Q)
        psi_G = self.kp * P / (1 + self.ks * G1)
        a = self.delta * I + psi_G * G1 + psi_Q * Q  # CAPITAL PSI == PSI_G * G + PSI_Q * Q ???
        C_prod = self.C_prod_homeo + (self.C_prod_max - self.C_prod_homeo) * (a / self.C12 + a)

        # Quiescent cells, y[0]
        dQ_dt = 2 * (1 - self.nu) * self.k_tr * A[-1] - (self.a1 + self.d1 + psi_Q) * Q

        # G1 cells, y[1]
        dG1_dt = self.a1 * Q - (self.a2 + self.d2 + psi_G + eta) * G1

        # Infected cells, y[2]
        dI_dt = -self.delta * I + eta * (G1 + N + G1R + NR)

        # Virions, y[3]
        dV_dt = self.alpha * self.delta * I - self.omega * V - eta * (G1 + N + G1R + NR) + self.viral_dose(t)

        # Transit compartments, y[4], ..., y[j+3]
        dA1_dt = self.a2 * G1 - self.k_tr * A[1] - (self.dg_hat + eta + psi_G) * A[1]

        dA_dt = [dA1_dt]
        for i in range(1, self.j):
            dAi_dt = self.k_tr * (A[i-1] - A[i]) - (self.dg_hat + eta + psi_G) * A[i]
            dA_dt.append(dAi_dt)

        # Immune cytokine, y[j+4]
        dC_dt = C_prod - self.k_elim * C + self.immune_dose(t)

        # Phagocytes, y[j+5]
        dP_dt = phi - self.gamma_P * P

        # Total number of cells in cycle, y[j+6]
        dN_dt = self.a2 * G1 - (self.dg_hat + eta + psi_G) * N - (self.k_tr / self.a2) * A[-1]

        # Resistant quiescent cells, y[j+7]
        dQR_dt = 2 * self.nu * self.k_tr * A[-1] + 2 * self.k_tr * AR[-1] - (self.a1_R + self.d1_R) * QR

        # Resistant G1 cells, y[j+8]
        dG1R_dt = self.a1_R * QR - (self.a2_R + self.d2_R + eta) * G1R

        # Resistant transit compartments, y[j+9], ..., y[2*j+8]
        dA1R_dt = self.a2 * G1R - self.k_tr * AR[1] - (self.dg_hat + eta + psi_G) * AR[1]

        dAR_dt = [dA1R_dt]
        for i in range(1, self.j):
            dAiR_dt = self.k_tr * (AR[i - 1] - AR[i]) - (self.dg_hat + eta + psi_G) * AR[i]
            dAR_dt.append(dAiR_dt)

        # Total number of resistant cells in cyle, y[2*j+9]
        dNR_dt = self.a2 * G1R - (self.dg_hat + eta) * NR - (self.k_tr / self.a2) * AR[-1]

        # Test
        # self.test_counter += 1
        # print(self.test_counter)
        # test_return = [0] * 28
        # test_return[3] = dV_dt
        # test_return[13] = dC_dt
        # test_return[14] = dP_dt
        # return test_return

        return [dQ_dt, dG1_dt, dI_dt, dV_dt] + dA_dt + [dC_dt, dP_dt, dN_dt, dQR_dt, dG1R_dt] + dAR_dt + [dNR_dt]

    def simulate(self, t_start=0, t_end=3, nsteps=100000):

        """
        :param t_start: Float. Time at the start of the simulation. Included.
        :param t_end:  Float. Time at the end of the simulation. Excluded.
        :param dt: Float. Time step.
        :param nsteps: Int. Max number of steps for a single call of the ode solver.
        :return: Simulation history. Includes initial conditions as first entry.
        """

        # r = ode(self.evaluate_derivatives).set_integrator('vode', method='bdf', atol=1e-8, rtol=1e-8, nsteps=nsteps, max_step=1/30/20)
        r = ode(self.evaluate_derivatives).set_integrator('lsoda', nsteps=nsteps, atol=1e-8, rtol=1e-8, max_step=1e-2)  # Small numerical fluctuations
        r.set_initial_value(self.initial_conditions, t_start)  # Set initial conditions

        y = np.empty(shape=(0, len(self.initial_conditions)))  # Solution

        while r.successful() and r.t + self.dt < t_end:
            print('Simulating... day ' + str(int(r.t*30)))
            r.integrate(r.t + self.dt)
            self.t = r.t
            y = np.vstack((y, np.real(r.y)))

        history = {'t_start': t_start,
                   't_end': t_end,
                   'y': y,
                   }

        return history

    def evaluate_obective(self, history):

        # TO DO: Add cumulative dose burden

        non_resistant_cycle = history['y'][:, 0] + history['y'][:, 1] + history['y'][:, self.j+6]  # Q, G1 and N
        resistant_cycle = history['y'][:, self.j+7] + history['y'][:, self.j+8] + history['y'][:, 2*self.j+9]  # QR, G1R and NR
        tumor_size = non_resistant_cycle + resistant_cycle + history['y'][:, 2]

        cumulative_tumor_burden = cumtrapz(y=tumor_size, x=None, dx=self.dt)  # Cumulative integral of tumor size

        return tumor_size, cumulative_tumor_burden
