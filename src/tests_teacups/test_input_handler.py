import sys
sys.path.append("./..")

import teacups.input_handler as inp
import teacups.grid as gri
import numpy as np


class Sys:
    def __init__(self):
        self.J_ex = 2
        self.D = 3
        self.E = 4
        self.D_tri = 5
        self.E_tri = 6
        self.width_gauss = 7
        return


class Exp:
    def __init__(self):
        self.B_z = np.linspace(300, 400, 3)
        self.B_mw = 1
        self.t_scale = [10, 14]
        self.t_points = 5
        return



class Opt:
    def __init__(self):
        self.grid_points = 5


class Cal:
    def __init__(self):
        return


def initialize_classes(self):
    self.sys = Sys()
    self.opt = Opt()
    self.cal = Cal()
    self.exp = Exp()
    return


class TestScaleInputs:
    def setup(self):
        initialize_classes(self)
        inp.scale_inputs(self.sys, self.exp, self.opt)

    def test_j(self):
        assert self.sys.J_ex == 2e6

    def test_d(self):
        assert self.sys.D == 3e6

    def test_e(self):
        assert self.sys.E == 4e6

    def test_d_tri(self):
        assert self.sys.D_tri == 5e6

    def test_e_tri(self):
        assert self.sys.E_tri == 6e6

    def test_B_z(self):
        np.testing.assert_array_equal(self.exp.B_z, np.linspace(0.3, 0.4, 3))

    def test_B_mw(self):
        assert self.exp.B_mw == 0.001

    def test_width_gauss(self):
        std = 0.21/(2*np.sqrt(2*np.log(2)))
        assert round(self.sys.width_gauss, 8) == round(std, 8)


class TestPredefinitions:
    def setup(self):
        initialize_classes(self)
        inp.predefinitions(self.sys, self.exp, self.cal)

    def test_t(self):
        t = np.array([10, 11, 12, 13, 14])
        np.testing.assert_array_equal(t, self.cal.t)

    def test_spec(self):
        spec = np.zeros((5, 3))
        np.testing.assert_array_equal(spec, self.cal.spec_sim)

    def test_dtype_t(self):
        assert self.cal.t.dtype == "float32"

    def test_dtype_spec(self):
        assert self.cal.spec_sim.dtype == "complex64"


class TestInitalizeSpinSystem:
    def setup(self):
        initialize_classes(self)

    def test_rp(self):
        self.sys.spin_system = "rp"
        inp.initialize_spin_system(self.sys)
        assert self.sys.s == [1/2, 1/2]

    def test_doub(self):
        self.sys.spin_system = "doub"
        inp.initialize_spin_system(self.sys)
        assert self.sys.s == [1/2]

    def test_trip(self):
        self.sys.spin_system = "trip"
        inp.initialize_spin_system(self.sys)
        assert self.sys.s == [1]

    def test_tdp(self):
        self.sys.spin_system = "tdp"
        inp.initialize_spin_system(self.sys)
        assert self.sys.s == [1/2, 1]


class TestCreateGridSophe:
    def setup(self):
        initialize_classes(self)
        self.opt.grid = "sophe"
        self.opt.sym = "D2h"
        inp.create_grid(self.opt, self.cal)

    def test_shape(self):
        assert self.cal.theta.shape == (21, )
        assert self.cal.phi.shape == (21, )
        assert self.cal.weights.shape == (21, )
        assert self.opt.grid_points == 21

    def test_arrays(self):
        theta, phi, weights = gri.sophe_grid(5, "D2h")
        np.testing.assert_allclose(self.cal.theta, theta)
        np.testing.assert_allclose(self.cal.phi, phi)
        np.testing.assert_allclose(self.cal.weights, weights)

    def test_type(self):
        assert self.cal.theta.dtype == "float32"
        assert self.cal.phi.dtype == "float32"
        assert self.cal.weights.dtype == "float32"

class TestCreateGridFibonacci:
    def setup(self):
        initialize_classes(self)
        self.opt.grid = "fibonacci"
        inp.create_grid(self.opt, self.cal)

    def test_shape(self):
        assert self.cal.theta.shape == (45, )
        assert self.cal.phi.shape == (45, )
        assert self.opt.grid_points == 45

    def test_arrays(self):
        theta, phi = gri.fibonacci_grid(45)
        np.testing.assert_allclose(self.cal.theta, theta)
        np.testing.assert_allclose(self.cal.phi, phi)

    def test_type(self):
        assert self.cal.theta.dtype == "float32"
        assert self.cal.phi.dtype == "float32"

class TestCreateGridSingle:
    def setup(self):
        initialize_classes(self)
        self.opt.grid = "single"
        self.opt.theta = [2, 5]
        self.opt.phi = [1, 2]
        inp.create_grid(self.opt, self.cal)

    def test_shape(self):
        assert self.cal.theta.shape == (2, )
        assert self.cal.phi.shape == (2, )
        assert self.opt.grid_points == 2

    def test_arrays(self):
        theta = np.array([2, 5])
        phi = np.array([1, 2])
        np.testing.assert_array_equal(self.cal.theta, theta)
        np.testing.assert_array_equal(self.cal.phi, phi)

    def test_type(self):
        assert self.cal.theta.dtype == "float32"
        assert self.cal.phi.dtype == "float32"


class TestSplitGrid:
    def setup(self):
        initialize_classes(self)
        self.sys.spin_system = "bla"
        self.sys.precursor = "blub"
        self.opt.grid = "piep"
        self.cal.theta = np.array([1, 2, 3])
        self.cal.phi = np.array([4, 5, 6])
        self.cal.weights = np.array([7, 8, 9])
        inp.split_grid(self.sys, self.exp, self.opt, self.cal)

    def test_type(self):
        assert type(self.cal.phi_split) == list
        assert type(self.cal.theta_split) == list

    def test_len(self):
        assert len(self.cal.phi_split) == 1
        assert len(self.cal.theta_split) == 1

    def test_phi(self):
        np.testing.assert_array_equal(self.cal.phi, self.cal.phi_split[0])

    def test_theta(self):
        np.testing.assert_array_equal(self.cal.theta, self.cal.theta_split[0])

    def test_weights(self):
        self.opt.grid = "sophe"
        inp.split_grid(self.sys, self.exp, self.opt, self.cal)
        np.testing.assert_array_equal(self.cal.weights,
                                      self.cal.weights_split[0])


class Test_hyperfine_converter:
    def setup(self):
        self.sys = Sys()

    def test_no_hyperfines(self):
        inp.hyperfine_converter(self.sys)
        try:
            self.sys.A
            assert False
        except AttributeError:
            assert True

    def test_hyperfine_in_list(self):
        self.sys.I = [[1/2]]
        self.sys.A = [[[1, 1, 1]]]
        inp.hyperfine_converter(self.sys)
        assert self.sys.A == [[[1, 1, 1]]]

    def test_hyperfine_doublet(self):
        self.sys.spin_system = 'doub'
        self.sys.I1 = 1/2
        self.sys.n1 = 2
        self.sys.A1 = [2, 2, 2]
        inp.hyperfine_converter(self.sys)

        assert self.sys.A == [[[2, 2, 2], [2, 2, 2]]]
        assert self.sys.I == [[1/2, 1/2]]
        assert self.sys.A_frame == [[[0, 0, 0], [0, 0, 0]]]

    def test_hyperfine_rp(self):
        self.sys.spin_system = 'rp'
        self.sys.I1 = 1/2
        self.sys.n1 = 1
        self.sys.A1 = [1, 2, 3]
        self.sys.I2 = 1
        self.sys.n2 = 1
        self.sys.A2 = [4, 5, 6]
        self.sys.A2_frame = [7, 7, 7]
        self.sys.acceptor_list = [1]
        self.sys.donor_list = [2]
        inp.hyperfine_converter(self.sys)
        assert self.sys.I == [[1/2], [1]]
        assert self.sys.A == [[[1, 2, 3]], [[4, 5, 6]]]
        assert self.sys.A_frame == [[[0, 0, 0]], [[7, 7, 7]]]
