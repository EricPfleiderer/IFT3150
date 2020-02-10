import AlphaSilico.viral_oncology_fit as vof
from AlphaSilico.insilico import Params as P

def test_fit():
    args = [P.a1, P.a1_R, P.a2, P.a2_R, P.alpha,
            P.C12, P.C_prod_homeo, P.C_prod_max, 
            P.d1, P.d1_R, P.d2, P.d2_R, P.d3_hat, 
            P.delta, P.eta12, P.gamma_P, P.kappa, 
            P.Kcp, P.k_elim, P.kp, P.kq, P.ks, P.N, 
            P.nu, P.omega, P.P12, P.transit_rate]
    print(args)
    assert 0 == 1
