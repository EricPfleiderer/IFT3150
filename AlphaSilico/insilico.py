import math
import numpy as np
import scipy.io as sio


class Params:

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

    # Time interval
    tf = 3
    tstep = math.floor(tf / 20)
    totaltime = [0, tf]

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
    k_elim = 0.16139 * 30  # elimination rate of cytokine

    # Immune parameters
    kp = 0.050 * 30  # contact rate with phagocytes
    kq = 10  # factor in denominator of Q phagocytosis term
    ks = kq  # factor in denominator of S phagocytosis term
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

    # INITIAL CONDITIONS
    QIC = (1 / a1 / total_time) * total_cells * (1 - nu)
    SIC = (1 / (a2 + d2) / total_time) * total_cells * (1 - nu)
    TCIC = (tau / total_time) * total_cells * (1 - nu) * np.ones(shape=(1, N)) / N  # Transit compartment ICs (DOUBLE CHECK)
    NCIC = (tau / total_time) * total_cells * (1 - nu)
    IIC = 0
    VIC = 0
    CIC = C_prod_homeo / k_elim
    PIC = Kcp * CIC / ((P12 + CIC) * gamma_P)
    RIC = (1 / a1 / total_time) * total_cells * nu
    RSIC = (1 / (a2 + d2) / total_time) * total_cells * nu
    resistant_TCIC = (tau / total_time) * total_cells * nu * np.ones(shape=(1, N)) / N
    resistant_total_cells_IC = (tau / total_time) * total_cells * nu
    initial_conditions = [QIC, SIC, IIC, VIC, TCIC, CIC, PIC, NCIC, RIC, RSIC, resistant_TCIC, resistant_total_cells_IC]

    # OPTIMIZATION
    admin_number = 75  # Number of immunotherapy doses - possible to dose everyday for 2.5 months
    viral_admin_number = 10  # Number of viral therapy doses - possible to dose every week for 2.5 months
    n_vars = admin_number + viral_admin_number  # Number of doses (variables to optimize over)
    lower_bound = np.zeros(shape=(1, n_vars))  # Lower bound for the optimizer
    upper_bound = 4 * np.ones(shape=(1, n_vars))  # Upper bound for the optimizer
    int_con = np.arange(1, n_vars+1)  # The condition that ensures that the optimial solution enforces integer multiple doses of the baseline dose

    # virtual_patient_parameter_input = sio.loadmat('16052019VirtualPopulation300PatientParameters.mat')['VirtualPatientParameters']  # shape 254x302. absolute path, better way ?

    def model_input(self):
        return [self.a1,
                self.a1_R,
                self.a2,
                self.a2_R,
                self.alpha,
                self.C12,
                self.C_prod_homeo,
                self.C_prod_max,
                self.d1,
                self.d1_R,
                self.d2,
                self.d2_R,
                self.d3_hat,
                self.delta,
                self.eta12,
                self.gamma_P,
                self.kappa,
                self.Kcp,
                self.k_elim,
                self.kp,
                self.kq,
                self.ks,
                self.N,
                self.nu,
                self.omega,
                self.P12,
                self.transit_rate
                ]
