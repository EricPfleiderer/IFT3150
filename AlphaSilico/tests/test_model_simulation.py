from AlphaSilico.src.insilico import TumorModel


def test_fit():

    t0 = 0
    t1 = 10
    dt = 0.1
    model = TumorModel()
    model.simulate(t0, t1, dt)

    assert 0 == 1
