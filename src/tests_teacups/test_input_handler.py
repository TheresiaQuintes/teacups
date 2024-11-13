import sys
sys.path.append("./..")

import teacups.input_handler as inp
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
        return


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
