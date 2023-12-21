import sys
sys.path.append("./..")

import teacups.input_handler as inp
import numpy as np


class Sys:
    def __init__(self):
        return


class Exp:
    def __init__(self):
        return


class Opt:
    def __init__(self):
        return


class Cal:
    def __init__(self):
        return


class Test_predefinitions:

    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.cal = Cal()

        self.exp.t_scale = [100, 200]
        self.exp.t_points = 50
        self.exp.B_z = np.linspace(1, 2, 3)
        inp.predefinitions(self.sys, self.exp, self.cal)

    def test_t(self):
        t = np.linspace(100, 200, 50, dtype=np.float32)
        assert np.array_equal(t, self.cal.t)


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
