import sys
sys.path.append("./..")

import numpy as np
import teacups.relaxation as relax


class Test_superoperator_populations:
    def setup(self):
        pass

    def test_2_2(self):
        T = np.array([[0, 1],
                      [1, 0]])

        R = np.array([[-1, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, -1]])

        R_calc = relax.superoperator_population_relaxation(T)
        assert np.array_equal(R, R_calc)

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
        assert np.array_equal(R, R_calc)

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
        assert np.array_equal(R, R_calc)

    def test_diagonal_relaxation_to_no_equilibrium(self):
        T = np.array([[-3, 2], [2, 0]])

        R = np.array([[-5, 0, 0, 2],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [2, 0, 0, -2]])

        R_calc = relax.superoperator_population_relaxation(T)
        assert np.array_equal(R, R_calc)


class Test_superoperator_coherence_relaxation:
    def setup(self):
        self.T = 3

    def test_2_2(self):
        dimension = 2
        R = np.eye(4)
        R *= -self.T

        R[0, 0] = 0
        R[3, 3] = 0

        R_calc = relax.superoperator_coherence_relaxation(self.T, dimension)
        assert np.array_equal(R, R_calc)

    def test_3_3(self):
        dimension = 3
        R = np.eye(9)
        R *= -self.T

        R[0, 0] = 0
        R[4, 4] = 0
        R[8, 8] = 0

        R_calc = relax.superoperator_coherence_relaxation(self.T, dimension)
        assert np.array_equal(R, R_calc)

    def test_4_4(self):
        dimension = 4
        R = np.eye(16)
        R *= -self.T

        R[0, 0] = 0
        R[5, 5] = 0
        R[10, 10] = 0
        R[15, 15] = 0

        R_calc = relax.superoperator_coherence_relaxation(self.T, dimension)
        assert np.array_equal(R, R_calc)


class Test_phenomenological_superoperator:
    def setup(self):
        self.T_relax_1 = 5
        self.T_relax_2 = 4
        pass

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

        R = relax.phenomenological_relaxation_superoperator(self.T_relax_1,
                                                            self.T_relax_2, 4)

        assert np.array_equal(R, relaxation_superop)
