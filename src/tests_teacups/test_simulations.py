import teacups.classes as cl
import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("./..")





class TestTeacups:
    def setup(self):
        pass

    def test_triplet_2D_hilbert(self):
        Sys = cl.Sys()
        Exp = cl.Exp()
        SimOpt = cl.SimOpt()

        # choose a triplet spin system
        Sys.spin_system = "trip"

        # set up g-tensors of the triplet
        Sys.g_tri = [2.003, 2.003, 2.003]

        # define triplet ZFS
        Sys.D_tri = -700
        Sys.E_tri = 100

        # define initial state
        Sys.precursor = "zf"
        Sys.population = [0, 0, 1]

        # define decay time and line width
        Sys.decay = 1e-6
        Sys.width_gauss = 1

        # experimental setup
        Exp.B_z = np.linspace(300, 400, 1024)
        Exp.t_scale = [0, 2e-6]
        Exp.t_points = 2
        Exp.B_mw = 0.001
        Exp.freq_mw = 9.75e9

        # simulation options
        SimOpt.grid_points = 10
        SimOpt.grid = "sophe"
        SimOpt.sym = "D2h"

        # do the simulation
        spec_sim = sim.teacups(Sys, Exp, SimOpt)
        spec_sim = spec_sim.real[1]/max(abs(spec_sim.real[1]))

        desired = np.load("./simulations/triplet_2D_hilbert.npy")

        np.testing.assert_allclose(spec_sim, desired, atol=2e-6)


    def test_tdp_2D_hilbert(self):
        Sys = cl.Sys()
        Exp = cl.Exp()
        SimOpt = cl.SimOpt()

        # choose a triplet doublet pair spin system
        Sys.spin_system = "tdp"

        # set up g-tensors of the radical and the triplet
        Sys.g = [2.0059, 2.0059, 2.0059]
        Sys.g_tri = [2.003, 2.003, 2.003]

        # define triplet ZFS
        Sys.D_tri = 500
        Sys.E_tri = -50

        # define couplings of the radical and the triplet
        Sys.J_ex = -400000

        Sys.precursor = "triplet-zf"
        Sys.population = [0.2, 0.205, 0.1, 0.1, 0.2]

        # define decay time and line width
        Sys.decay = 1e-6
        Sys.width_gauss = 1

        # experimental setup
        Exp.B_z = np.linspace(333, 363, 550)
        Exp.t_scale = [0, 2e-6]
        Exp.t_points = 2
        Exp.freq_mw = 9.75e9

        # simulation options
        SimOpt.grid = "sophe"
        SimOpt.sym = "D2h"
        SimOpt.grid_points = 8
        SimOpt.CUPY = False

        # do the simulation
        cal = sim.teacups(Sys, Exp, SimOpt, development=True)

        spec_sim = cal.spec_sim
        spec_sim = spec_sim.real[1]/max(abs(spec_sim.real[1]))

        desired = np.load("./simulations/tdp_2D_hilbert.npy")
        np.testing.assert_allclose(spec_sim, desired, atol=2e-6)
