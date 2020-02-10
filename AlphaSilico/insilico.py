import math
import numpy as np
import scipy.io as sio


class Params:
    def __init__(self):
        # Baseline model parameters
        self.a1 = 1.183658646441553*30
        self.a2 = 1.758233712464858*30
        self.d1 = 0
        self.d2 = 0.539325116600707*30
        self.d3 = self.d2
        self.tau = (33.7/24 - 30/ self.a2)/30  # Mean intermitotic time (lower bound on a2)
        self.intermitotic_SD = 6.7/24/30

        # Distribution specific parameters
        self.N = round(self.tau**2 / self.intermitotic_SD**2)
        self.transit_rate = self.N / self.tau  # Transit rate across compartments
        self.d3_hat = self.N / self.tau * (math.exp(self.d3 * self.tau / (self.N + 1)) - 1)
        self.d3_hat_R = self.d3_hat

        # Time interval
        self.tf = 3
        self.tstep = math.floor(self.tf / 20)

        # Viral therapy
        self.kappa = 3.534412642851458 * 30
        self.delta = 4.962123414821151 * 30
        self.alpha = 0.008289097649957
        self.omega = 9.686308020782763 * 30
        self.eta12 = 0.510538277701167  # virion half effect contact rate

        # Cytokine parameters
        self.C_prod_homeo = 0.00039863 * 30  # Homeostatic cytokine production
        self.C_prod_max = 1.429574637713578 * 30  # Maximal cytokine production rate
        self.C12 = 0.739376299393775 * 30  # Half effect in cytokine production
        self.k_elim = 0.16139 * 30  # elimination rate of cytokine

        # Immune parameters
        self.kp = 0.050 * 30  # contact rate with phagocytes
        self.kq = 10  # factor in denominator of Q phagocytosis term
        self.ks = self.kq  # factor in denominator of S phagocytosis term
        self.P12 = 5  # Half effect in cytokine driven phagocyte production
        self.gamma_P = 0.35 * 30  # From Barrish 2017 PNAS elimination rate of phagocyte
        self.Kcp = 4.6754 * 30  # From Barrish 2017 PNAS conversion of cytokine into phagocyte

        # Immune Steady State
        self.C_star = self.C_prod_homeo / self.k_elim
        self.P_star = (1 / self.gamma_P) * (self.Kcp * self.C_star / (self.P12 + self.C_star))

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

        # INITIAL CONDITIONS
        self.QIC = (1 / self.a1 / self.total_time) * self.total_cells * (1 - self.nu)
        self.SIC = (1 / (self.a2 + self.d2) / self.total_time) * self.total_cells * (1 - self.nu)
        self.TCIC = (self.tau / self.total_time) * self.total_cells * (1 - self.nu) * np.ones(shape=(1, self.N)) / self.N  # Transit compartment ICs (DOUBLE CHECK)
        self.NCIC = (self.tau / self.total_time) * self.total_cells * (1 - self.nu)
        self.IIC = 0
        self.VIC = 0
        self.CIC = self.C_prod_homeo / self.k_elim
        self.PIC = self.Kcp * self.CIC / ((self.P12 + self.CIC) * self.gamma_P)
        self.RIC = (1 / self.a1 / self.total_time) * self.total_cells * self.nu
        self.RSIC = (1 / (self.a2 + self.d2) / self.total_time) * self.total_cells * self.nu
        self.resistant_TCIC = (self.tau / self.total_time) * self.total_cells * self.nu * np.ones(shape=(1, self.N)) / self.N
        self.resistant_total_cells_IC = (self.tau / self.total_time) * self.total_cells * self.nu
        self.initial_conditions = [self.QIC, self.SIC, self.IIC, self.VIC, self.TCIC, self.CIC, self.PIC
                                   , self.NCIC, self.RIC, self.RSIC, self.resistant_TCIC, self.resistant_total_cells_IC]

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
