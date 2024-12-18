import sys
sys.path.append("./..")

import numpy as np
import scipy.linalg as la
from unittest.mock import Mock
import teacups.signals_and_processing as sap
import teacups.relaxation as rlx
from copy import deepcopy
import teacups.matrix_tools as mt

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
        self.t = [1, 2, 3]
        ham = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        ham = np.array([ham, ham, ham+1])
        self.ham = np.array([ham, ham+2, ham, ham])

        ham_s = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        ham_s = np.array([ham_s, ham_s, ham_s+1])
        self.ham_superop = np.array([ham_s, ham_s+2, ham_s, ham_s])

        t1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                      dtype=np.complex64)
        t2 = t1/2
        self.signal = np.array([t1, t2])
        return


def initialize_classes(self):
    self.sys = Sys()
    self.opt = Opt()
    self.cal = Cal()
    self.exp = Exp()
    return


class TestPropagationHilbert:
    def setup(self):
        initialize_classes(self)
        self.opt.space = 'hilbert'
        sap.propagation(self.sys, self.opt, self.cal)

    def test_shape(self):
        assert self.cal.propagation.shape == (4, 3, 2, 2)

    def test_dtype(self):
        assert self.cal.propagation.dtype == "complex64"

    def test_value(self):
        comp1 = la.expm(-1j*np.array([[0, 1], [1, 0]], dtype=np.complex64))
        comp2 = la.expm(-1j*np.array([[1, 2], [2, 1]], dtype=np.complex64))
        comp3 = la.expm(-1j*np.array([[2, 3], [3, 2]], dtype=np.complex64))
        np.testing.assert_allclose(comp1, self.cal.propagation[0, 0], atol=2e-6)
        np.testing.assert_allclose(comp2, self.cal.propagation[0, 2], atol=2e-6)
        np.testing.assert_allclose(comp3, self.cal.propagation[1, 1], atol=2e-6)


class TestPropagationLiouville:
    def setup(self):
        initialize_classes(self)
        self.opt.space = 'liouville'
        self.cal.ham_superop = self.cal.ham
        sap.propagation(self.sys, self.opt, self.cal)

    def test_shape(self):
        assert self.cal.propagation.shape == (4, 3, 2, 2)

    def test_dtype(self):
        assert self.cal.propagation.dtype == "complex64"

    def test_value(self):
        comp1 = la.expm(-1j*np.array([[0, 1], [1, 0]], dtype=np.complex64))
        comp2 = la.expm(-1j*np.array([[1, 2], [2, 1]], dtype=np.complex64))
        comp3 = la.expm(-1j*np.array([[2, 3], [3, 2]], dtype=np.complex64))
        np.testing.assert_allclose(comp1, self.cal.propagation[0, 0], atol=2e-6)
        np.testing.assert_allclose(comp2, self.cal.propagation[0, 2], atol=2e-6)
        np.testing.assert_allclose(comp3, self.cal.propagation[1, 1], atol=2e-6)


class TestPowderAverageSophe:
    def setup(self):
        initialize_classes(self)
        self.cal.spec_sim = np.zeros((2, 4), dtype=np.complex64)
        self.cal.weights = [0.1, 0.2, 0.3]
        self.opt.grid = "sophe"

        sap.powder_average(self.opt, self.cal)

    def test_shape(self):
        assert self.cal.spec_sim.shape == self.cal.signal.shape[:2]

    def test_dtype(self):
        assert self.cal.spec_sim.dtype == "complex64"

    def test_value(self):
        w = np.array([[0.1, 0.4, 0.9], [0.4, 1.0, 1.8],
                            [0.7, 1.6, 2.7], [1, 2.2, 3.6]],
                      dtype=np.complex64)

        comp_t1 = np.array([w[0].sum(), w[1].sum(), w[2].sum(), w[3].sum()])
        comp_t2 = comp_t1/2
        comp = np.array([comp_t1, comp_t2])

        np.testing.assert_allclose(self.cal.spec_sim, comp)


class TestPowderAverageFibonacci:
    def setup(self):
        initialize_classes(self)
        self.cal.spec_sim = np.zeros((2, 4), dtype=np.complex64)
        self.opt.grid = "fibonacci"

        sap.powder_average(self.opt, self.cal)

    def test_shape(self):
        assert self.cal.spec_sim.shape == self.cal.signal.shape[:2]

    def test_dtype(self):
        assert self.cal.spec_sim.dtype == "complex64"

    def test_value(self):
        w = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                      dtype=np.complex64)

        comp_t1 = np.array([w[0].sum(), w[1].sum(), w[2].sum(), w[3].sum()])
        comp_t2 = comp_t1/2
        comp = np.array([comp_t1, comp_t2])

        np.testing.assert_allclose(self.cal.spec_sim, comp)


class TestPowderAverageSingle:
    def setup(self):
        initialize_classes(self)
        self.cal.spec_sim = np.zeros((2, 4), dtype=np.complex64)
        self.opt.grid = "single"
        sap.powder_average(self.opt, self.cal)

    def test_shape(self):
        assert self.cal.spec_sim.shape == self.cal.signal.shape[:2]

    def test_dtype(self):
        assert self.cal.spec_sim.dtype == "complex64"

    def test_value(self):
        w = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                      dtype=np.complex64)

        comp_t1 = np.array([w[0].sum(), w[1].sum(), w[2].sum(), w[3].sum()])
        comp_t2 = comp_t1/2
        comp = np.array([comp_t1, comp_t2])

        np.testing.assert_allclose(self.cal.spec_sim, comp)


class TestSignalHilbertDecay:
    def setup(self):
        initialize_classes(self)
        self.cal.spec_sim = np.array([[1, 2, 3, 4], [1, 2, 3, 4],
                                      [1, 2, 3, 4]], dtype=np.complex64)
        self.sys.decay = 2
        self.cal.t = np.log([1, 4, 9])

        sap.signal_hilbert_decay(self.sys, self.cal)

    def test_shape(self):
        assert self.cal.spec_sim.shape == (3, 4)

    def test_dtype(self):
        assert self.cal.spec_sim.dtype == "complex64"

    def test_value(self):
        comp = np.array([[1, 2, 3, 4], [1/2, 2/2, 3/2, 4/2],
                         [1/3, 2/3, 3/3, 4/3]])
        np.testing.assert_allclose(self.cal.spec_sim, comp)


class TestTimeEvolutionHilbert:
    def setup(self):
        initialize_classes(self)
        p1 = np.array([[0, -1], [1, 0]])
        p2 = np.array([[-1, 0], [0, 1]])
        p = np.array([[p1, p2], [p2, p1]], dtype=np.complex64)
        self.cal.propagation = p
        self.propagation_invers = np.transpose(p, (0, 1, 3, 2))
        rho = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        self.cal.rho = np.array([[rho, rho], [rho, rho]])
        self.cal.observable = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        self.cal.signal = np.zeros((3, 2, 2), dtype=np.complex64)
        sap.time_evolution_hilbert(self.cal)

    def test_shape(self):
        assert self.cal.signal.shape == (3, 2, 2)

    def test_dtype(self):
        assert self.cal.signal.dtype == "complex64"

    def test_value(self):
        comp = np.array([[[5, 5], [5, 5]],
                        [[-5, -5], [-5, -5]],
                        [[5, 5], [5, 5]]])
        np.testing.assert_allclose(self.cal.signal, comp)



class TestTimeEvolutionLiouville:
    def setup(self):
        initialize_classes(self)
        p1 = np.array([[0, -1], [1, 0]])
        p2 = np.array([[-1, 0], [0, 1]])
        p = np.array([[p1, p2], [p2, p1]], dtype=np.complex64)
        self.cal.propagation = p
        rho = np.array([2, 3], dtype=np.complex64)
        self.cal.rho = np.array([[rho, rho], [rho, rho]])
        self.cal.observable = np.array([1, 1], dtype=np.complex64)
        self.cal.signal = np.zeros((3, 2, 2), dtype=np.complex64)
        sap.time_evolution_liouville(self.cal)

    def test_shape(self):
        assert self.cal.signal.shape == (3, 2, 2)

    def test_dtype(self):
        assert self.cal.signal.dtype == "complex64"

    def test_value(self):
        comp = np.array([[[5, 5], [5, 5]],
                        [[-1, 1], [1, -1]],
                        [[-5, 5], [5, -5]]])
        np.testing.assert_allclose(self.cal.signal, comp)


class TestMakeSignalSignals:
    def setup(self):
        initialize_classes(self)
        self.exp.B_z = np.array([2, 3])
        self.opt.grid_points = 2
        self.opt.cpu_cores = 1
        self.opt.pop_evolution = False
        self.opt.space = "liouville"

    def test_hilbert(self):
        p1 = np.array([[0, -1], [1, 0]])
        p2 = np.array([[-1, 0], [0, 1]])
        p = np.array([[p1, p2], [p2, p1]], dtype=np.complex64)
        self.cal.propagation = p
        self.propagation_invers = np.transpose(p, (0, 1, 3, 2))
        rho = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        self.cal.rho = np.array([[rho, rho], [rho, rho]])
        self.cal.observable = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        self.cal.signal = np.zeros((3, 2, 2), dtype=np.complex64)

        cal_comp = deepcopy(self.cal)
        sap.time_evolution_hilbert(cal_comp)

        self.opt.space = "hilbert"
        sap.make_signal(self.exp, self.opt, self.cal)

        assert self.cal.signal.dtype == "complex64"
        np.testing.assert_allclose(self.cal.signal, cal_comp.signal)

    def test_liouville(self):
        p1 = np.array([[0, -1], [1, 0]])
        p2 = np.array([[-1, 0], [0, 1]])
        p = np.array([[p1, p2], [p2, p1]], dtype=np.complex64)
        self.cal.propagation = p
        rho = np.array([2, 3], dtype=np.complex64)
        self.cal.rho = np.array([[rho, rho], [rho, rho]])
        self.cal.observable = np.array([1, 1], dtype=np.complex64)
        self.cal.signal = np.zeros((3, 2, 2), dtype=np.complex64)

        cal_comp = deepcopy(self.cal)
        sap.time_evolution_liouville(cal_comp)

        sap.make_signal(self.exp, self.opt, self.cal)

        assert self.cal.signal.dtype == "complex64"
        np.testing.assert_allclose(self.cal.signal, cal_comp.signal)

    def test_hilbert_multiple_cores(self):
        self.opt.cpu_cores = 0

        p1 = np.array([[0, -1], [1, 0]])
        p2 = np.array([[-1, 0], [0, 1]])
        p = np.array([[p1, p2], [p2, p1]], dtype=np.complex64)
        self.cal.propagation = p
        self.propagation_invers = np.transpose(p, (0, 1, 3, 2))
        rho = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        self.cal.rho = np.array([[rho, rho], [rho, rho]])
        self.cal.observable = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        self.cal.signal = np.zeros((3, 2, 2), dtype=np.complex64)

        cal_comp = deepcopy(self.cal)
        sap.time_evolution_hilbert(cal_comp)

        self.opt.space = "hilbert"
        sap.make_signal(self.exp, self.opt, self.cal)

        assert self.cal.signal.dtype == "complex64"
        np.testing.assert_allclose(self.cal.signal, cal_comp.signal)

    def test_liouville_multiple_cores(self):
        self.opt.cpu_cores = 0

        p1 = np.array([[0, -1], [1, 0]])
        p2 = np.array([[-1, 0], [0, 1]])
        p = np.array([[p1, p2], [p2, p1]], dtype=np.complex64)
        self.cal.propagation = p
        rho = np.array([2, 3], dtype=np.complex64)
        self.cal.rho = np.array([[rho, rho], [rho, rho]])
        self.cal.observable = np.array([1, 1], dtype=np.complex64)
        self.cal.signal = np.zeros((3, 2, 2), dtype=np.complex64)

        cal_comp = deepcopy(self.cal)
        sap.time_evolution_liouville(cal_comp)

        sap.make_signal(self.exp, self.opt, self.cal)

        assert self.cal.signal.dtype == "complex64"
        np.testing.assert_allclose(self.cal.signal, cal_comp.signal)


class TestMakeSignalPopEvolution:
    def setup(self):
        initialize_classes(self)
        self.exp.B_z = np.array([2, 3, 4])
        self.opt.grid_points = 2
        self.opt.cpu_cores = 1
        self.opt.pop_evolution = True
        self.opt.space = "liouville"

        self.sup = np.arange(1, 17, dtype=np.complex64).reshape(4, 4)
        self.sup_b_angle_matrix = np.array([[self.sup, self.sup],
                                            [self.sup, self.sup],
                                            [self.sup, self.sup]])
        self.vec = np.array([1, 2, 3, 4], dtype=np.complex64)
        self.vec_b_angle_matrix = np.array([[self.vec, self.vec],
                                            [self.vec, self.vec],
                                            [self.vec, self.vec]])
        self.hil = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        self.hil_b_angle_matrix = np.array([[self.hil, self.hil],
                                            [self.hil, self.hil],
                                            [self.hil, self.hil]])

        self.cal.propagation = self.sup_b_angle_matrix
        self.cal.rho = self.vec_b_angle_matrix
        self.cal.observable = self.vec

        self.cal.eigvec = self.hil_b_angle_matrix
        self.cal.s = mt.Spinoperator(1/2)
        sap.make_signal(self.exp, self.opt, self.cal)


    def test_shape(self):
        assert self.cal.pop_evolution.shape == (3, 2)

    def test_dtype(self):
        assert self.cal.pop_evolution.dtype == "complex64"

    def test_value(self):
        r = self.vec
        t1 = np.conj(self.hil.T) @ r.reshape(2, 2) @ self.hil
        r = self.sup @ r
        t2 = np.conj(self.hil.T) @ r.reshape(2, 2) @ self.hil
        r = self.sup @ r
        t3 = np.conj(self.hil.T) @ r.reshape(2, 2) @ self.hil

        comp = np.array([np.diag(t1), np.diag(t2), np.diag(t3)])
        np.testing.assert_allclose(comp, self.cal.pop_evolution)
