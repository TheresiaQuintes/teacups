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
        spec = sim.teacups(Sys, Exp, SimOpt)
        spec = spec.real[1]/max(abs(spec.real[1]))

        desired = np.load("./simulations/triplet_2D_hilbert.npy")

        np.testing.assert_allclose(spec, desired, atol=2e-6)


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
        spec = sim.teacups(Sys, Exp, SimOpt)
        spec = spec.real[1]/max(abs(spec.real[1]))

        desired = np.load("./simulations/tdp_2D_hilbert.npy")
        np.testing.assert_allclose(spec, desired, atol=2e-6)


    def test_doublet_transient_nutations(self):
        # initialize classes with default parameters
        Sys = cl.Sys()
        Exp = cl.Exp()
        SimOpt = cl.SimOpt()

        # set up spin System parameters
        Sys.g = [2, 2, 2]
        Sys.width_gauss = 3
        Sys.decay = 5e-6


        Sys.spin_system = 'doub'
        Sys.precursor = 'eigen'
        Sys.population = [0, 1]

        # set up Experimental parameters
        Exp.B_z = np.linspace(320, 380, 3000)
        Exp.t_scale = [0, 10e-6]
        Exp.t_points = 60
        Exp.B_mw = 0.2

        # set up simulation SimOption parameters
        SimOpt.grid_points = 1
        SimOpt.space = 'hilbert'

        # do simulation
        spec = sim.teacups(Sys, Exp, SimOpt)
        spec = spec/abs(spec).real.max()

        desired = np.load("./simulations/doublet_transient_nutations.npy")
        np.testing.assert_allclose(spec, desired, atol=2e-6)


    def test_rp_quantum_beats(self):
        "including triplet precursor"
        # initialize classes with default parameters
        Sys = cl.Sys()
        Exp = cl.Exp()
        SimOpt = cl.SimOpt()


        Sys.spin_system = 'rp'
        Sys.g1 = [2.00304, 2.00262, 2.00232]
        Sys.g1_frame = [-0.262, 0.489, 0.471]
        Sys.g2 = [2.00564, 2.00494, 2.00217]

        # Sys.precursor = 'singlet'
        # Triplet precursor
        Sys.precursor = 'triplet-zf'
        Sys.g_tri = [2.00370, 2.00285, 2.00246]
        Sys.population = [1, 0, 0]
        Sys.D_tri = -1.9217e+03
        Sys.E_tri = +525.4678

        Sys.decay = 1e-6
        Sys.T_relax_1 = 1e-6
        Sys.T_relax_2 = 1e-6

        Sys.D = 3.3630
        Sys.D_frame = [0, 1.012, -0.017]
        Sys.J_ex = 0

        Sys.width_gauss = 0.08

        Exp.B_z = np.linspace(350.5, 352.3, 200)
        Exp.t_scale = [0, 2e-6]
        Exp.t_points = 75
        Exp.B_mw = 0.03
        Exp.freq_mw = 9.8562*1e9

        SimOpt.grid_points = 10
        SimOpt.space = 'hilbert'

        # do simulation
        spec = sim.teacups(Sys, Exp, SimOpt)
        spec = spec/abs(spec).real.max()

        desired = np.load("./simulations/rp_quantum_beats.npy")
        np.testing.assert_allclose(spec, desired, atol=2e-6)


    def test_tdp_rqm(self):
        Sys = cl.Sys()
        Exp = cl.Exp()
        SimOpt = cl.SimOpt()

        # choose a triplet doublet pair spin system
        Sys.spin_system = "tdp"

        # set up g-tensors of the radical and the triplet
        Sys.g = [2.0059, 2.0059, 2.0059]
        Sys.g_tri = [2.003, 2.003, 2.003]

        # define triplet ZFS
        Sys.D_tri = 700

        # define couplings of the radical and the triplet
        Sys.J_ex = -20000

        # define initial state
        Sys.precursor = "eigen"
        Sys.population = [0.3, 0.225, 0.2, 0.3, 0.5, 0.48]

        # define line width
        Sys.width_gauss = 1

        # experimental setup
        Exp.B_z = np.linspace(320, 380, 350)
        Exp.t_scale = [0, 2e-6]
        Exp.t_points = 60
        Exp.B_mw = 0.001
        Exp.freq_mw = 9.75e9

        # simulation options
        SimOpt.grid_points = 3
        SimOpt.space = "liouville"
        SimOpt.pop_evolution = True

        ### set up dynamics-matrix

        # determine eigenvalues
        SimOpt.eigval_mode = True
        eigval = sim.teacups(Sys, Exp, SimOpt)
        e = np.mean(eigval[175], axis=0)

        # build delta E between trip-doublet and trip-quartet states
        de_53 = e[5]-e[3]
        de_51 = e[5]-e[1]
        de_50 = e[5]-e[0]
        de_43 = e[4]-e[3]
        de_42 = e[4]-e[2]
        de_40 = e[4]-e[0]

        # Dipolar coupling squared
        D = (Sys.D_tri*1e6)**2

        # set isc and doublet decay rates
        k_isc = 0.3/1e-11
        k_d = 0.25/1e-6

        # set up dynamics matrix
        R = np.zeros((6, 6))
        R[5, 5] = -k_d
        R[4, 4] = -k_d

        R[5, 3] = k_isc/45*(D/(de_53)**2)
        R[5, 1] = k_isc/135*(D/(de_51)**2)
        R[5, 0] = k_isc/45*(D/(de_50)**2)
        R[3, 5] = k_isc/45*(D/(de_53)**2)
        R[1, 5] = k_isc/135*(D/(de_51)**2)
        R[0, 5] = k_isc/45*(D/(de_50)**2)

        R[4, 3] = k_isc/45*(D/(de_42)**2)
        R[4, 2] = k_isc/135*(D/(de_42)**2)
        R[4, 0] = k_isc/45*(D/(de_40)**2)
        R[3, 4] = k_isc/45*(D/(de_43)**2)
        R[2, 4] = k_isc/135*(D/(de_42)**2)
        R[0, 4] = k_isc/45*(D/(de_40)**2)

        Sys.dynamics = R

        SimOpt.eigval_mode = False

        # do simulation
        spec, pop_evolution = sim.teacups(Sys, Exp, SimOpt)
        spec = spec/abs(spec).real.max()

        desired = np.load("./simulations/tdp_rqm.npy")
        np.testing.assert_allclose(spec, desired, atol=2e-6)


    def test_znp_triplet_asymmetric(self):
        Sys = cl.Sys()
        Exp = cl.Exp()
        SimOpt = cl.SimOpt()

        # set up spin System parameters
        Sys.g_tri = [2.008, 2.008, 2.008]
        Sys.D_tri = -898
        Sys.E_tri = 161

        Sys.spin_system = 'trip'
        Sys.precursor = 'zf'
        Sys.population = [0.05, 0, 0.95]

        Sys.decay = 1e-6
        Sys.dynamics = np.array([[0, 0.01e6, 0],
                                 [0.01e6, 0, 0.25e6],
                                 [0, 0.25e6, 0]])

        Sys.width_gauss = 2

        # set up Experimental parameters
        Exp.B_z = np.linspace(295, 395, 256*4)
        Exp.t_scale = [1.9e-6, 3.8e-6]
        Exp.t_points = 476
        Exp.B_mw = 0.0001

        # set up simulation SimOption parameters
        SimOpt.grid_points = 3
        SimOpt.grid = 'sophe'
        SimOpt.sym = "D2h"
        SimOpt.space = 'liouville'
        SimOpt.pop_evolution = True



        # do simulation
        spec, pop = sim.teacups(Sys, Exp, SimOpt)

        desired_spec = np.load("./simulations/znp_triplet_asymmetric_spec.npy")
        desired_pop = np.load("./simulations/znp_triplet_asymmetric_pop.npy")
        np.testing.assert_allclose(spec.real, desired_spec, atol=2e-6)
        np.testing.assert_allclose(pop.real, desired_pop, atol=2e-6)


# %%
