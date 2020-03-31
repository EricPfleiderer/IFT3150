import math
import sys
import numpy as np
from scipy.integrate import ode, cumtrapz


class Environment:  # Equivalent to Game

    def __init__(self, treatment_start=0, treatment_len=75, observation_len=90, max_doses=4, immunotherapy_offset=1, virotherapy_offset=7):

        """
        This class serves as an interface between the State class and the Agent class.
        :param treatment_len: Float. Total length of treatment in days. 2.5 by default.
        :param observation_len: Float. Total length of observation period in days. 6 by default.
        :param max_doses: Int. Upper limit on how many doses may be administered at once. 4 by default.
        :param immunotherapy_offset: Float. Immunotherapy offset in days. Daily by default.
        :param virotherapy_offset: Float. Virotherapy offset in days. Weekly by default.
        """

        # Parameters
        self.treatment_start = treatment_start
        self.treatment_len = treatment_len
        self.observation_len = observation_len
        self.immunotherapy_offset = immunotherapy_offset
        self.virotherapy_offset = virotherapy_offset
        self.max_doses = max_doses

        # Initializations
        self.state = State(treatment_start=treatment_start, treatment_len=treatment_len, observation_len=observation_len, immunotherapy_offset=immunotherapy_offset,
                           virotherapy_offset=virotherapy_offset)
        self.t = 0  # Current time in days
        self.dt = 1  # Step size in days
        self.y = np.array(self.state.initial_conditions)  # Current solution
        self.history = {'t': np.array([self.t]),
                        'y': np.array([self.state.initial_conditions])
                        }
        self.dose_history = {'immunotherapy': {'t': [],
                                               'y': [],
                                               },
                             'virotherapy': {'t': [],
                                             'y': [],
                                             }
                             }

    def evaluate_obective(self):

        non_resistant_cycle = self.history['y'][:, 0] + self.history['y'][:, 1] + self.history['y'][:, self.state.j + 6]  # Q, G1 and N
        resistant_cycle = self.history['y'][:, self.state.j + 7] + self.history['y'][:, self.state.j + 8] + self.history['y'][:, 2 * self.state.j + 9]  # QR, G1R and NR
        tumor_size = non_resistant_cycle + resistant_cycle + self.history['y'][:, 2]

        cumulative_tumor_burden = cumtrapz(y=tumor_size, x=self.history['t'])  # Cumulative integral of tumor size

        # cumulative_dose_burden = cumtrapz(y=self.dose_history['immunotherapy']['y'], x=self.dose_history['immunotherapy']['t']) + \
        #                          cumtrapz(y=self.dose_history['virotherapy']['y'], x=self.dose_history['virotherapy']['t'])

        return tumor_size, cumulative_tumor_burden  # , cumulative_dose_burden

    def reset(self):
        """
        Resets the environment for new simulations.
        :return: Void.
        """
        self.state = State(treatment_start=0, treatment_len=self.treatment_len, immunotherapy_offset=self.immunotherapy_offset, virotherapy_offset=self.virotherapy_offset)
        self.t = 0
        self.y = np.array(self.state.initial_conditions)  # Current solution
        self.history = {'t': np.array([self.t]),
                        'y': np.array([self.state.initial_conditions])
                        }

    def step(self, action=(0, 0), verbose=True):

        """
        A single call to this method will advance the state by dt and preserve history.
        :param action: Tuple of two ints. Dosage to be applied during the call. First entry corresponds to immunotherapy, second to virotherapy.
        :param verbose: Boolean. Prints simulation progress if True.
        :return: State instance, Boolean. New state after call to step, boolean flag for endgame.
        """

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write('Simulating... ' + str(round(self.t/self.observation_len*100, 1)) + '%')
            sys.stdout.flush()

        # Simulate a time step
        self.state, value, done = self.state.take_action(action)

        # Updates
        self.history['y'] = np.vstack((self.history['y'], self.state.y))
        self.history['t'] = np.append(self.history['t'], self.t)
        self.t, self.y = self.state.t, self.state.y

        if done and verbose:
            print('\n Simulating... done!')

        return self.state, value, done


class State:  # Equivalent to GameState

    # Immunotherapy
    immunotherapy_admin = 125000  # Cytokine per unit volume
    immunotherapy_k_abs = 6.6311  # Absorbtion rate
    immunotherapy_availability = 0.85  # Bioavailability
    immunotherapy_vol = 7

    # Virotherapy
    virotherapy_admin = 250  # Viral load
    virotherapy_k_abs = 20  # Absorbtion rate
    virotherapy_availability = 1  # Bioavailability
    virotherapy_vol = 7
    kappa = 3.534412642851458  # Virion contact rate
    delta = 4.962123414821151  # Lysis rate
    alpha = 0.008289097649957  # Lytic virion release rate
    omega = 9.686308020782763  # Virion death rate
    eta12 = 0.510538277701167  # Virion half effect concentration

    # Cytokine parameters
    C_prod_homeo = 0.00039863  # Homeostatic cytokine production rate
    C_prod_max = 1.429574637713578  # Maximal cytokine production rate
    C12 = 0.739376299393775  # Maximal cytokine production rate
    k_elim = 0.16139  # Cytokine elimination rate

    # Constants
    intermitotic_SD = 6.7 / 24
    PSI12 = 5  # Cytokine production half effect  # MISSING IN MATLAB CODE ???
    gamma_P = 0.35  # From Barrish 2017 PNAS elimination rate of phagocyte

    def __init__(self, initial_conditions=None, dose_history=None, immunotherapy=None, virotherapy=None, treatment_start=0, treatment_len=75, observation_len=90,
                 immunotherapy_offset=1, virotherapy_offset=7, a1=1.183658646441553, a2=1.758233712464858, d1=0, d2=0.539325116600707, kp=0.05, kq=10, k_cp=4.6754):

        """
        Initializes a system of ordinary differential equations and an integrator to simulate and solve a melanoma tumor growth. Model by Craig & Cassidy.
        :param treatment_start: Int. First day of treatment.
        :param treatment_len: Int. Treatment length in days.
        :param a1: Float. Quiescent to interphase rate. (1/day)
        :param a2: Float. Interphase to active phase rate. (1/day)
        :param d1: Float. Quiescent death rate. (1/day)
        :param d2: Float. Interphase death rate. (1/day)
        :param kp: Float. Phagocyte-tumor cell contact rate. (1/day)
        :param kq: Float. Phagocyte cell digestion constant.
        :param k_cp: Float. Maximal phagocyte production rate (10**10 cells/day)
        """

        self.treatment_start = treatment_start  # Treatment start time in days
        self.treatment_len = treatment_len  # Treatment length in days
        self.observation_len = observation_len  # Total expected simulation length in days

        self.immunotherapy = immunotherapy
        self.virotherapy = virotherapy

        # Treatment plan (no treatment by default)
        if immunotherapy is None:
            self.immunotherapy = np.array([])
        if virotherapy is None:
            self.virotherapy = np.array([])

        # Treatment offsets
        self.immunotherapy_offset = immunotherapy_offset
        self.virotherapy_offset = virotherapy_offset

        # Administration times
        self.t_immunotherapy_admin = np.arange(self.immunotherapy.size) * self.immunotherapy_offset + treatment_start
        self.t_virotherapy_admin = np.arange(self.virotherapy.size) * self.virotherapy_offset + treatment_start

        # Variable patient parameters
        self.a1 = a1
        self.a2 = a2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d2
        self.tau = (33.7/24 - 1/a2)
        self.kp = kp
        self.kq = kq
        self.ks = kq
        self.k_cp = k_cp
        self.variable_params = (self.a1, self.a2, self.d1, self.d2, self.d3, self.tau, self.kp, self.kq, self.ks, self.k_cp)

        # Distribution specific parameters
        self.j = int(round(self.tau**2 / self.intermitotic_SD**2))  # Number of transit compartments
        self.k_tr = self.j / self.tau  # Transit rate across compartments
        self.dg_hat = self.j / self.tau * (math.exp(self.d3 * self.tau / (self.j + 1)) - 1)  # 14.8948 VS SUGGESTED 0.167 = 5.01 in S.I. GREATLY AFFECTS SCALE
        # self.dg_hat = 0.167  # Suggested by S.I.
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
        if initial_conditions is None:
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
            self.t = 0
        else:
            self.initial_conditions = initial_conditions['y']
            self.t = initial_conditions['t']

        # Simulation statistics
        self.y = np.array(self.initial_conditions)
        self.id = self.get_id()

        self.dose_history = dose_history
        if self.dose_history is None:
            self.dose_history = {'immunotherapy': {'t': [],
                                                   'y': [],
                                                   },
                                 'virotherapy': {'t': [],
                                                 'y': [],
                                                 }
                                 }

        self.integrator = None

        # Set integrator
        self.integrator = ode(self._evaluate_derivatives).set_integrator('lsoda', nsteps=100000, atol=1e-8, rtol=1e-8, max_step=5e-3)

        self.end_game_value = None
        self.end_game_state = None

    def get_id(self):
        """
        Convert current solution to string format to be used as keys in MCTS dictionnaries.
        :return: String. Representation of current solution.
        """
        return 'p:' + ''.join(str(self.variable_params)) + 'i:' + np.array2string(self.immunotherapy) + 'v:' + np.array2string(self.virotherapy)

    def get_tumor_stats(self):
        non_resistant_cycle = self.y[0] + self.y[1] + self.y[self.j+6]  # Q, G1 and N
        resistant_cycle = self.y[self.j+7] + self.y[self.j+8] + self.y[2*self.j+9]  # QR, G1R and NR
        tumor_size = non_resistant_cycle + resistant_cycle + self.y[2]  # Infected cells

        doublings = math.log2(tumor_size) - math.log2(self.total_cells)  # Number of doublings since beginning of treatment

        return tumor_size, doublings

    def reset_integrator(self, nsteps=100000, max_steps=5e-3, atol=1e-8, rtol=1e-8):
        """

        :param nsteps: Int. Maximum number of calls before the solver quits.
        :param max_steps: Float. Max step size the solver can take.
        :param atol: Float. Absolute tolerance.
        :param rtol: Float. Relative tolerance.
        :return: Void.
        """

        self.integrator = ode(self._evaluate_derivatives).set_integrator('lsoda', nsteps=nsteps, atol=atol, rtol=rtol, max_step=max_steps)
        self.integrator.set_initial_value(self.y, self.t)  # Set initial conditions

    def _add_to_treatment(self, dosage, treatment_type):

        """
        Used to add a dosage to immunotherapy or virotherapy treatments.
        :param dosage: Int. Dosage to add.
        :param treatment_type: String. 'immunotherapy' or 'virotherapy'.
        :return:
        """

        treatment = getattr(self, treatment_type)  # Select appropriate treatment
        offset = getattr(self, treatment_type + '_offset')  # Select appropriate offset
        setattr(self, treatment_type, np.append(treatment, dosage))  # Add dose to treatment plan
        setattr(self, 't_' + treatment_type + '_admin', np.arange(treatment.size+1) * offset + self.treatment_start)  # Update admninistration times to cover new treatmant plan

    def dose(self, t, treatment_type):

        """
        Returns dose of a defined treatment to be administered at time t.

        VULNERABILITY: Decay terms at t in t_admin are unsteady due to finite solver steps. Solver must use small steps around t_admin. Suggested 5e-3 and smaller.

        TO DO:
            - Implement detection event (first passes through t_admin) to dynamically rescale step size.

        :param t: Float. Current time in days.
        :param treatment_type: String. Either 'immunotherapy' or 'virotherapy'
        :return: Float. Dose administered at time t.
        """

        # Only medicate if a treatment is defined and ongoing
        if t < self.treatment_len + self.treatment_start and getattr(self, treatment_type).size is not 0:

            # Masking
            time_mask = np.where(t >= getattr(self, 't_' + treatment_type + '_admin'))  # Create a mask to ignore future doses
            treatment = getattr(self, treatment_type)[time_mask]  # Apply mask to treatment
            t_admin = getattr(self, 't_' + treatment_type + '_admin')[time_mask]  # Apply mask to administration times

            # Get relevant parameters
            k_abs = getattr(self, treatment_type + '_k_abs')
            availability = getattr(self, treatment_type + '_availability')
            admin = getattr(self, treatment_type + '_admin')

            # Compute initial dose and decay terms
            doses = k_abs * availability * admin * treatment
            decay = np.exp(-k_abs*(t-t_admin))  # UNSTABLE / NOISY, REQUIRES SMALL STEPS AROUND t_admin

            if (len(self.dose_history[treatment_type]['t']) == 0 and len(self.dose_history[treatment_type]['y']) == 0) or self.dose_history[treatment_type]['t'][-1] != t:
                self.dose_history[treatment_type]['t'].append(t)
                self.dose_history[treatment_type]['y'].append(np.sum(doses*decay)/getattr(self, treatment_type + '_vol'))

            return np.sum(doses * decay) / getattr(self, treatment_type + '_vol')

        return 0

    def _evaluate_derivatives(self, t, y):

        """
        Returns the right hand side of our system of ODEs at time t, given previous solution y.
        :param t: Float. Current time value.
        :param y: 1D list. Previous solution.
        :return: 1D list. Derivative of each quantity in y at time step t.
        """

        Q, G1, I, V, A, C, P, N, QR, G1R, AR, NR = y[0], y[1], y[2], y[3], y[4:self.j+4], y[self.j+4], y[self.j+5], y[self.j+6], y[self.j+7], y[self.j+8], \
                                                   y[self.j+9:2*self.j+9], y[2*self.j+9]

        # Auxiliary functions
        infection = 0
        if V > 1e-10:
            infection = V / (self.eta12 + V)
        eta = self.kappa * infection
        phi = self.k_cp * C / (self.C12 + C)  # DIFFERENCE BETWEEN MATLAB (PSI12) CODE AND S.I (PSI12 VS C12).
        psi_Q = self.kp * P / (1 + self.kq * Q)
        psi_G = self.kp * P / (1 + self.ks * G1)
        a = self.delta * I + psi_G * G1 + psi_Q * Q
        C_prod = self.C_prod_homeo + (self.C_prod_max - self.C_prod_homeo) * (a / self.C12 + a)

        # Quiescent cells, y[0]
        dQ_dt = 2 * (1 - self.nu) * self.k_tr * A[-1] - (self.a1 + self.d1 + psi_Q) * Q

        # G1 cells, y[1]
        dG1_dt = self.a1 * Q - (self.a2 + self.d2 + psi_G + eta) * G1

        # Infected cells, y[2]
        dI_dt = -self.delta * I + eta * (G1 + N + G1R + NR)

        # Virions, y[3]
        dV_dt = self.alpha * self.delta * I - self.omega * V - eta * (G1 + N + G1R + NR) + self.dose(t, treatment_type='virotherapy')

        # Transit compartments, y[4], ..., y[j+3]
        dA1_dt = self.a2 * G1 - self.k_tr * A[1] - (self.dg_hat + eta + psi_G) * A[1]

        dA_dt = [dA1_dt]
        for i in range(1, self.j):
            dAi_dt = self.k_tr * (A[i-1] - A[i]) - (self.dg_hat + eta + psi_G) * A[i]
            dA_dt.append(dAi_dt)

        # Immune cytokine, y[j+4]
        dC_dt = C_prod - self.k_elim * C + self.dose(t, treatment_type='immunotherapy')

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

        return [dQ_dt, dG1_dt, dI_dt, dV_dt] + dA_dt + [dC_dt, dP_dt, dN_dt, dQR_dt, dG1R_dt] + dAR_dt + [dNR_dt]

    def _forward(self, step_size=1):

        """
        Simulate the model through time.
        :param step_size: Float. Time step in days.
        :return: Numpy array. New solution.
        """

        # Must reinitialize at each call (Isoda cannot integrate multiple problems simultaneously)
        self.integrator.set_initial_value(self.y, self.t)  # Set initial conditions

        # Integrate the solution
        self.integrator.integrate(self.integrator.t + step_size)
        self.t, self.y = self.integrator.t, self.integrator.y

        self.id = self.get_id()

    def get_available_actions(self, max_dose=4):
        """
        Check if allowed to medicate at current time (immunotherapy, virotherapy respectively).
        :return: Tuple of Booleans.
        """

        # 0 doses allowed by default
        possible_doses = [np.arange(1), np.arange(1)]

        # Extend action space on administration days
        if round(self.t) % self.immunotherapy_offset == 0:
            possible_doses[0] = np.arange(max_dose+1)
        if round(self.t) % self.virotherapy_offset == 0:
            possible_doses[1] = np.arange(max_dose+1)

        meshgrid = np.array(np.meshgrid(possible_doses[0], possible_doses[1]))
        combinations = meshgrid.T.reshape(-1, 2)

        return meshgrid, combinations

    def get_end_game(self):
        next_state = State(initial_conditions={'t': self.t, 'y': self.y}, treatment_start=self.treatment_start, treatment_len=self.treatment_len,
                           observation_len=self.observation_len, immunotherapy=self.immunotherapy, virotherapy=self.virotherapy, dose_history=self.dose_history)
        next_state._forward(step_size=self.observation_len - self.treatment_len)
        tumor_size, doublings = next_state.get_tumor_stats()
        self.end_game_value = doublings
        self.end_game_state = next_state
        return self.end_game_value, self.end_game_state

    def take_action(self, action):

        # Check for endgame
        value = 0  # If not terminal, return agnostic value ? HEURISTIC HERE
        next_state = State(initial_conditions={'t': self.t, 'y': self.y}, treatment_start=self.treatment_start, treatment_len=self.treatment_len,
                           observation_len=self.observation_len, immunotherapy=self.immunotherapy, virotherapy=self.virotherapy, dose_history=self.dose_history)

        done = self.t >= self.treatment_len  # check if next state is terminal

        # Run the rest of the simulation to get endgame statistics
        if done:
            if self.end_game_value is None and self.end_game_state is None:  # Save endgame statistics on first visit
                value, next_state = self.get_end_game()
            else:
                value, next_state = self.end_game_value, self.end_game_state

        else:
            # Modify treatment
            if self.t < self.treatment_len:

                # Add immunotherapy if needed
                if round(self.t) % self.immunotherapy_offset == 0:
                    next_state._add_to_treatment(action[0], 'immunotherapy')

                # Add virotherapy if needed
                if round(self.t) % self.virotherapy_offset == 0:
                    next_state._add_to_treatment(action[1], 'virotherapy')

            # Simulate a step in time
            next_state._forward(step_size=1)

        return next_state, value, done
