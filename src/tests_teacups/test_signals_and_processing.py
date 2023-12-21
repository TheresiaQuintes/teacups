import sys
sys.path.append("./..")

import numpy as np
import scipy.linalg as la
from unittest.mock import Mock
import teacups.signals_and_processing as sap
import teacups.relaxation as rlx


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


class Test_propagation:
    def setup(self):
        self.sys = Mock()
        self.exp = Mock()
        self.opt = Mock()
        self.cal = Mock()

        self.cal.t = [2, 1]
        ham = np.array([[0, 1], [1, 0]])
        self.cal.ham = np.array([[ham, ham], [ham, ham]])
        self.cal.eigval, self.cal.eigvec = np.linalg.eigh(self.cal.ham)
        self.cal.s.dimension = 3

        ham_adj = np.transpose(np.conjugate(self.cal.ham), [0, 1, 3, 2])
        self.cal.ham_superop = np.kron(
            np.eye(
                self.cal.ham.shape[-1], dtype=np.complex64), self.cal.ham[:, :])\
            - np.kron(ham_adj[:, :], np.eye(
                self.cal.ham.shape[-1], dtype=np.complex64))

        self.sys.dynamics = np.zeros((2, 2))

    def test_shape_hilbert(self):
        self.opt.space = 'hilbert'
        sap.propagation(self.sys, self.opt, self.cal)
        assert self.cal.propagation.shape == (2, 2, 2, 2)

    def test_shape_liouville(self):
        self.opt.space = 'liouville'
        sap.propagation(self.sys, self.opt, self.cal)
        assert self.cal.propagation.shape == (2, 2, 4, 4)

    def test_value_hilbert(self):
        comp = la.expm(-1j*self.cal.ham[0, 0])
        self.opt.space = 'hilbert'
        sap.propagation(self.sys, self.opt, self.cal)
        np.testing.assert_allclose(comp, self.cal.propagation[0, 0])

    def test_value_liouville(self):
        relax = rlx.create_relaxation_superoperator(self.sys, self.cal)
        superop = self.cal.ham_superop + 1j*relax

        comp = la.expm(1j*superop[0, 0])
        self.opt.space = 'liouville'
        sap.propagation(self.sys, self.opt, self.cal)
        np.testing.assert_allclose(comp, self.cal.propagation[0, 0])


class Test_make_signal:
    def setup(self):
        self.sys = Mock()
        self.exp = Mock()
        self.opt = Mock()
        self.cal = Mock()
