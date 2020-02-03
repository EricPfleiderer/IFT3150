import numpy as np


def psi_q(P, Q, args):
    """
    Cleareance of quiescent cells by immune system
    :param P:
    :param Q:
    :param args:
    """
    return P * args.kp / (1 + args.kp * Q)


def psi_s(P, S, args):
    """
    Clearance of susceptible cells by immune system
    :param P:
    :param S:
    :param args:
    """
    return P * args.kp / (1 + args.ks * S)


def viral_oncology_fit(t, y, **args): 
    dydt = np.array(12)
    dydt[0] = 2*(1 - args.nu)*args.transit_rate*y[args.N + 4] - (args.a1 + args.d1 + psi_q(y[args.N + 6], y[1], args) + psi_s(y[args.N+6], y[2], args))*y[2]
    dydt[1] = args[a1]*y[0] - args[a2]

