import math
import numpy as np
import scipy.io as sio
from scipy.integrate import ode


class TumorModel:

    # Baseline model parameters
    a1 = 1.183658646441553*30
    a2 = 1.758233712464858*30
    d1 = 0
    d2 = 0.539325116600707*30
    d3 = d2
    tau = (33.7/24 - 30/a2)/30  # Mean intermitotic time (lower bound on a2)
    intermitotic_SD = 6.7/24/30

    # Distribution specific parameters
    N = round(tau**2 / intermitotic_SD**2)
    transit_rate = N / tau  # Transit rate across compartments
    d3_hat = N / tau * (math.exp(d3 * tau / (N + 1)) - 1)
    d3_hat_R = d3_hat

    # Viral therapy
    kappa = 3.534412642851458 * 30
    delta = 4.962123414821151 * 30
    alpha = 0.008289097649957
    omega = 9.686308020782763 * 30
    eta12 = 0.510538277701167  # virion half effect contact rate

    # Cytokine parameters
    C_prod_homeo = 0.00039863 * 30  # Homeostatic cytokine production
    C_prod_max = 1.429574637713578 * 30  # Maximal cytokine production rate
    C12 = 0.739376299393775 * 30  # Half effect in cytokine production
    k_elim = 0.16139 * 30  # Elimination rate of cytokine

    # Immune parameters
    kp = 0.050 * 30  # Contact rate with phagocytes
    kq = 10  # Factor in denominator of Q phagocytosis term
    ks = kq  # Factor in denominator of S phagocytosis term
    P12 = 5  # Half effect in cytokine driven phagocyte production
    gamma_P = 0.35 * 30  # From Barrish 2017 PNAS elimination rate of phagocyte
    Kcp = 4.6754 * 30  # From Barrish 2017 PNAS conversion of cytokine into phagocyte

    # Immune Steady State
    C_star = C_prod_homeo / k_elim
    P_star = (1/gamma_P) * (Kcp * C_star / (P12 + C_star))

    # Resistant parameters
    nu = 1e-10  # Mutation percentage
    a1_R = a1
    a2_R = a2
    d1_R = d1
    d2_R = d2
    d3_R = d3
    kappa_R = kappa

    # Cell cycle duration
    total_time = 1 / a1 + 1 / (a2 + d2) + tau
    total_cells = 200

    # Initial conditions
    QIC = (1 / a1 / total_time) * total_cells * (1 - nu)
    SIC = (1 / (a2 + d2) / total_time) * total_cells * (1 - nu)
    TCIC = (tau / total_time) * total_cells * (1 - nu) * np.ones(shape=N) / N  # Transit compartment ICs (DOUBLE CHECK)
    NCIC = (tau / total_time) * total_cells * (1 - nu)
    IIC = 0
    VIC = 0
    CIC = C_prod_homeo / k_elim
    PIC = Kcp * CIC / ((P12 + CIC) * gamma_P)
    RIC = (1 / a1 / total_time) * total_cells * nu
    RSIC = (1 / (a2 + d2) / total_time) * total_cells * nu
    resistant_TCIC = (tau / total_time) * total_cells * nu * np.ones(shape=N) / N
    resistant_total_cells_IC = (tau / total_time) * total_cells * nu
    initial_conditions = [QIC, SIC, IIC, VIC] + TCIC.tolist() + [CIC, PIC, NCIC, RIC, RSIC] + resistant_TCIC.tolist() + [resistant_total_cells_IC]

    # OPTIMIZATION
    # admin_number = 75  # Number of immunotherapy doses - possible to dose everyday for 2.5 months
    # viral_admin_number = 10  # Number of viral therapy doses - possible to dose every week for 2.5 months
    # n_vars = admin_number + viral_admin_number  # Number of doses (variables to optimize over)
    # lower_bound = np.zeros(shape=(1, n_vars))  # Lower bound for the optimizer
    # upper_bound = 4 * np.ones(shape=(1, n_vars))  # Upper bound for the optimizer
    # int_con = np.arange(1, n_vars+1)  # The condition that ensures that the optimial solution enforces integer multiple doses of the baseline dose
    #
    # virtual_patient_parameter_input = sio.loadmat('16052019VirtualPopulation300PatientParameters.mat')['VirtualPatientParameters']  # shape 254x302. absolute path, better way ?

    def f_eval(self, t, y):

        """
        :param t: Float. Current time value.
        :param y: 1D list. Previous solution.
        :param model: TumorModel instance.
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

        # ODE for first compartment, y[5], ..., y[N+3]
        dAi_dt = []
        for j in range(5, self.N+4):
            dAj_dt = self.transit_rate * (y[j-1] - y[j]) - (self.d3_hat + eta + psi_S * y[j])
            dAi_dt.append(dAj_dt)

        # Immune cytokine, y[N+4]
        dC_dt = C_prod - self.k_elim * y[self.N+4]  # + Dose(PA, t)

        # Phagocytes, y[N+5]
        dP_dt = self.Kcp * y[self.N+4] / (self.P12 + y[self.N + 4]) - self.gamma_P * y[self.N+5]

        # ODE for total number of cells in cell cycle, y[N+6]
        dT_dt = self.a2 * y[1] - (self.d3_hat + eta + psi_S) * y[self.N + 6] - (self.transit_rate / self.a2) * y[self.N + 3]

        # Resistant quiescent cells, y[N+7]
        dQR_dt = 2 * self.nu * self.transit_rate * y[self.N+3] + 2 * self.transit_rate * y[2*self.N+8] - (self.a1_R + self.d1_R) * y[self.N+7]

        # Resistant G1 cells, y[N+8]
        dG1R_dt = self.a1_R * y[self.N+7] - (self.a2_R + self.d2_R + eta) * y[self.N+8]

        # Resistant first transit, y[N+9]
        dA1R_dt = self.a2_R * y[self.N+8] - self.transit_rate * y[self.N+9] - (self.d3_hat + eta * y[self.N+9])

        # DE for resistant first transit, y[N+10], ..., y[2*N+8]
        dAiR_dt = []
        for j in range(self.N+10, 2*self.N+9):
            dAjR_dt = self.transit_rate * (y[j - 1] - y[j]) - (self.d3_hat + eta * y[j])
            dAiR_dt.append(dAjR_dt)

        # DE for total resistant cells, y[2*N+9]
        dTR_dt = self.a2 * y[self.N+8] - (self.d3_hat + eta) * y[2*self.N+9] - (self.transit_rate / self.a2) * y[2*self.N+9]

        return [dQ_dt, dG1_dt, dI_dt, dV_dt, dA1_dt] + dAi_dt + [dC_dt, dP_dt, dT_dt, dQR_dt, dG1R_dt, dA1R_dt] + dAiR_dt + [dTR_dt]

    def simulate(self, t_start, t_end, dt):

        """

        :param t_start: Float. Time at the start of the simulation. Included.
        :param t_end:  Float. Time at the end of the simulation. Excluded.
        :param dt: Float. Time step.
        :return: Simulation history. Includes initial conditions as first entry.
        """

        r = ode(self.f_eval).set_integrator('zvode')  # Initialize the integrator
        r.set_initial_value(self.initial_conditions, t_start)  # Set initial conditions and model args

        history = np.array(self.initial_conditions)

        while r.successful() and r.t < t_end - dt:
            r.integrate(r.t + dt)
            history = np.vstack((history, np.real(r.y)))

        return history
