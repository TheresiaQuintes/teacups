import sys
sys.path.append("./..")

import numpy as np
import teacups.relaxation as relax
import teacups.matrix_tools as mt
import pytest as pt


class Sys:
    def setup(self):
        return

class Cal:
    def setup(self):
        return


class TestSuperoperatorPopulations:
    def setup(self):
        self.T = np.array([[-3, 2], [2, 0]])
        self.R = relax.superoperator_population_relaxation(self.T)

    def test_shape(self):
        assert self.R.shape == (4, 4)

    def test_dtype(self):
        assert self.R.dtype == "float32"

    def test_2_2(self):
        T = np.array([[0, 1],
                      [1, 0]])

        R = np.array([[-1, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, -1]])

        R_calc = relax.superoperator_population_relaxation(T)
        np.testing.assert_array_equal(R, R_calc)

    def test_3_3(self):
        T = np.array([[0, 2, 1],
                      [2, 0, 3],
                      [1, 3, 0]])

        R = np.array([[-3, 0, 0, 0, 2, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [2, 0, 0, 0, -5, 0, 0, 0, 3],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 3, 0, 0, 0, -4]])

        R_calc = relax.superoperator_population_relaxation(T)
        np.testing.assert_array_equal(R, R_calc)

    def test_4_4(self):
        T = np.array([[0, 2, 1, 6],
                      [2, 0, 5, 3],
                      [1, 5, 0, 4],
                      [6, 3, 4, 0]])

        R = np.array([[-9, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [2, 0, 0, 0, 0, -10, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 5, 0, 0, 0, 0, -10, 0, 0, 0, 0, 4],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [6, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, -13]])

        R_calc = relax.superoperator_population_relaxation(T)
        np.testing.assert_array_equal(R, R_calc)

    def test_diagonal_relaxation_to_no_equilibrium(self):
        R = np.array([[-5, 0, 0, 2],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [2, 0, 0, -2]])

        np.testing.assert_array_equal(R, self.R)


class TestSuperoperatorCoherenceRelaxation:
    def setup(self):
        self.T = 3
        self.R = relax.superoperator_coherence_relaxation(self.T, 3)

    def test_shape(self):
        assert self.R.shape == (9, 9)

    def test_dtype(self):
        assert self.R.dtype == "float32"

    def test_2_2(self):
        dimension = 2
        R = np.eye(4)
        R *= -self.T

        R[0, 0] = 0
        R[3, 3] = 0

        R_calc = relax.superoperator_coherence_relaxation(self.T, dimension)
        np.testing.assert_array_equal(R, R_calc)

    def test_3_3(self):
        R = np.eye(9)
        R *= -self.T

        R[0, 0] = 0
        R[4, 4] = 0
        R[8, 8] = 0

        np.testing.assert_array_equal(R, self.R)

    def test_4_4(self):
        dimension = 4
        R = np.eye(16)
        R *= -self.T

        R[0, 0] = 0
        R[5, 5] = 0
        R[10, 10] = 0
        R[15, 15] = 0

        R_calc = relax.superoperator_coherence_relaxation(self.T, dimension)
        np.testing.assert_array_equal(R, R_calc)


class TestPhenomenologicalSuperoperator:
    def setup(self):
        self.T_relax_1 = 5
        self.T_relax_2 = 4
        self.R = relax.phenomenological_relaxation_superoperator(
            self.T_relax_1, self.T_relax_2, 4)

    def test_shape(self):
        assert self.R.shape == (16, 16)

    def test_dtype(self):
        assert self.R.dtype == "float32"

    def test_rp_superoperator(self):
        a = 1/self.T_relax_1
        b = 1/self.T_relax_2

        # upper triangle of the matrix
        relax_up = np.zeros((16, 16), dtype=np.complex64)
        relax_up[0, 5] = a
        relax_up[0, 10] = a
        relax_up[0, 15] = a
        relax_up[5, 15] = a
        relax_up[10, 15] = a
        relax_up[5, 10] = a

        # lower triangle of the matrix
        relax_lo = np.transpose(relax_up)

        # diagonal elements of the matrix
        relax_diag = np.eye(16, 16, dtype=np.complex64)
        relax_diag *= -b
        relax_diag[0, 0] = -3*a
        relax_diag[5, 5] = -3*a
        relax_diag[10, 10] = -3*a
        relax_diag[15, 15] = -3*a

        relaxation_superop = relax_up + relax_lo + relax_diag

        np.testing.assert_array_equal(self.R, relaxation_superop)


class TestRelaxationOperatorToHamiltonianBasis:
    def setup(self):
        eigvec = np.array([[1+2j, 2+0j], [3-1j, 4+1j]])
        eigvec = np.array([eigvec, eigvec, eigvec])
        eigvec = np.array([eigvec, eigvec], dtype=np.complex64)

        self.relaxation = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                                    [2, 3, 4, 5], [0, 2, 4, 6]],
                                   dtype=np.complex64)

        a = 1+2j
        b = 2+0j
        c = 3-1j
        d = 4+1j

        eigvec_super = np.array([[a*a, a*b, a*b, b*b],
                                 [a*c, a*d, b*c, b*d],
                                 [a*c, b*c, a*d, b*d],
                                 [c*c, c*d, c*d, d*d]])

        eigvec_super_inv = np.conj(np.transpose(eigvec_super))

        eigvec_super = np.array([eigvec_super, eigvec_super, eigvec_super])
        self.eigvec_super = np.array([eigvec_super, eigvec_super])
        eigvec_super_inv = np.array([eigvec_super_inv, eigvec_super_inv,
                                     eigvec_super_inv])
        self.eigvec_super_inv = np.array([eigvec_super_inv, eigvec_super_inv])

        self.relaxation_basistransformed =\
            relax.relaxation_operator_to_hamiltonian_basis(
                self.relaxation, eigvec)

    def test_shape(self):
        assert self.relaxation_basistransformed.shape == (2, 3, 4, 4)

    def test_dtype(self):
        assert self.relaxation_basistransformed.dtype == "complex64"

    def test_value(self):
        comp = self.eigvec_super @ self.relaxation @ self.eigvec_super_inv
        np.testing.assert_array_equal(comp, self.relaxation_basistransformed)


class TestCreateRelaxationSuperoperator:
    def setup(self):
        self.sys = Sys()
        self.cal = Cal()
        eigvec = np.array([[1+2j, 2+0j], [3-1j, 4+1j]])
        eigvec = np.array([eigvec, eigvec, eigvec])
        self.cal.eigvec = np.array([eigvec, eigvec], dtype=np.complex64)
        self.cal.s = mt.Spinoperator(1/2)

    def test_dynamics(self):
        self.sys.dynamics = np.array([[1, 2], [3, 4]])
        comp = relax.superoperator_population_relaxation(self.sys.dynamics)
        comp = relax.relaxation_operator_to_hamiltonian_basis(comp,
                                                              self.cal.eigvec)
        r = relax.create_relaxation_superoperator(self.sys, self.cal)
        np.testing.assert_array_equal(comp, r)

    def test_dynamics_and_T_relax(self):
        self.sys.dynamics = np.array([[1, 2], [3, 4]])
        self.sys.T_relax_1 = 5
        comp = relax.superoperator_population_relaxation(self.sys.dynamics)
        comp = relax.relaxation_operator_to_hamiltonian_basis(comp,
                                                              self.cal.eigvec)
        r = relax.create_relaxation_superoperator(self.sys, self.cal)
        np.testing.assert_array_equal(comp, r)

    def test_T_relax(self):
        self.sys.dynamics = None
        self.sys.T_relax_1 = 5
        self.sys.T_relax_2 = 7
        comp = relax.phenomenological_relaxation_superoperator(
            self.sys.T_relax_1, self.sys.T_relax_2, self.cal.s.dimension)
        comp = relax.relaxation_operator_to_hamiltonian_basis(comp,
                                                              self.cal.eigvec)
        r = relax.create_relaxation_superoperator(self.sys, self.cal)
        np.testing.assert_array_equal(comp, r)

    def test_missing_dynamic(self):
        with pt.raises(AttributeError):
            relax.create_relaxation_superoperator(self.sys, self.cal)
