import teacups.grid as grid
import teacups.orientation_dependent_ham as odh
import numpy as np
import teacups.matrix_tools as mt
import pytest as pt
import sys
sys.path.append("./..")


# Testing of class matrix

class TestScalar:
    def setup(self):
        self.m = mt.Matrix(2)
        self.m.matrix = np.arange(1, 5).reshape((2, 2))
        self.m.matrix = self.m.matrix.astype(np.complex64)
        self.m.scalar(np.array([2, 3]))

    def test_value(self):
        comp = np.array([[6, 12], [18, 24]])
        assert np.array_equal(comp, self.m.matrix)

    def test_dtype(self):
        assert self.m.matrix.dtype == "complex64"


class TestProduct:
    def setup(self):
        self.m = mt.Matrix(2)
        self.m.matrix = np.arange(1, 5).reshape((2, 2))
        self.m.matrix = self.m.matrix.astype(np.complex64)
        self.scd_matrix = np.array(
            np.arange(2, 6).reshape((2, 2)), dtype=np.complex64)

    def test_wrong_shape(self):
        with pt.raises(IndexError):
            self.m.product(np.eye(3))

    def test_product_right(self):
        self.m.product(self.scd_matrix)
        comp = np.array([[10, 13], [22, 29]])

        assert np.array_equal(comp, self.m.matrix)
        assert self.m.matrix.dtype == "complex64"

    def test_product_left(self):
        self.m.product(self.scd_matrix, left=True)
        comp = np.array([[11, 16], [19, 28]])
        assert np.array_equal(comp, self.m.matrix)
        assert self.m.matrix.dtype == "complex64"


class TestBasisTransformation:
    def setup(self):
        self.m = mt.Matrix(4)
        self.m.matrix = np.array([[1+2j, 2+3j, 7+1j, 3+0j],
                                 [2+0j, 4+1j, 3+0j, 4-1j],
                                 [10+7j, 5-2j, 8-4j, 10+10j],
                                 [2+1j, 0+1j, 3-7j, 8+0j]],
                                 dtype=np.complex64)
        self.eig, self.vec = np.linalg.eig(self.m.matrix)
        self.eigm = np.diag(self.eig)

        b = 1j/np.sqrt(2)
        self.ort_trans = np.array([[1, 0, 0, 0], [0, b, b, 0],
                                   [0, -b, b, 0], [0, 0, 0, 1]])

    def test_wrong_shape(self):
        with pt.raises(IndexError):
            self.m.basis_transformation(np.eye(3))

    def test_inverse_left(self):
        self.m.basis_transformation(self.vec)
        np.testing.assert_allclose(self.m.matrix, self.eigm, atol=2e-6)
        assert self.m.matrix.dtype == "complex64"

    def test_inverse_right(self):
        c = mt.Matrix(3)
        c.matrix = self.eigm
        c.basis_transformation(self.vec, inverse_left=False)
        np.testing.assert_allclose(c.matrix, self.m.matrix, atol=2e-6)
        assert c.matrix.dtype == "complex64"

    def test_inverse_left_orthonormal(self):
        comp = np.conj(self.vec.T) @ self.m.matrix @ self.vec
        self.m.basis_transformation(self.vec, orthonormal=True)
        np.testing.assert_allclose(
            self.m.matrix, comp)
        assert self.m.matrix.dtype == "complex64"

    def test_inverse_right_orhtonormal(self):
        comp = self.vec @ self.eigm @ np.conj(self.vec.T)

        c = mt.Matrix(3)
        c.matrix = self.eigm
        c.basis_transformation(self.vec, inverse_left=False, orthonormal=True)
        np.testing.assert_allclose(c.matrix, comp)
        assert c.matrix.dtype == "complex64"

    def test_orthonormal_identical(self):
        normal = self.m
        onormal = self.m
        normal.basis_transformation(self.ort_trans, orthonormal=False)
        onormal.basis_transformation(self.ort_trans, orthonormal=True)
        np.testing.assert_allclose(normal.matrix, onormal.matrix)


# Testing of class tensor

class TestTensorInit:
    def setup(self):
        self.ten = mt.Tensor(np.arange(3))

    def test_shape(self):
        assert self.ten.matrix.shape == (3, 3)

    def test_dtype(self):
        assert self.ten.matrix.dtype == "float32"

    def test_value(self):
        comp = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
        assert np.array_equal(comp, self.ten.matrix)

    def test_wrong_number_of_diagonals(self):
        with pt.raises(IndexError):
            mt.Tensor([1, 2, 3, 4])


class TestRotation:
    def setup(self):
        self.tensor = mt.Tensor(np.arange(3, dtype=np.float32))
        self.unit = mt.Tensor([1, 1, 1])
        self.tensor.rotation(1, 2)
        self.unit.rotation(1, 2)

    def test_shape(self):
        assert self.tensor.rot.shape == (3, 3)

    def test_dtype(self):
        assert self.tensor.rot.dtype == "float32"

    def test_eye(self):
        np.testing.assert_allclose(self.unit.matrix, self.unit.rot, atol=2e-6)

    def test_values(self):
        comp = np.array([[1.77626649, -0.18920062,  0.48886663],
                         [-0.18920062,  0.29192658,  0.41341091],
                         [0.48886663,  0.41341091,  0.93180692]])
        np.testing.assert_allclose(comp, self.tensor.rot, atol=2e-6)


class TestMultiRotation:
    def setup(self):
        self.tensor = mt.Tensor(np.arange(3))
        theta, phi = grid.fibonacci_grid(4)
        self.tensor.multirotation(phi, theta)

    def test_shape(self):
        assert self.tensor.multirot.shape == (4, 3, 3)

    def test_dtype(self):
        assert self.tensor.multirot.dtype == "float32"

    def test_value(self):
        comp = np.array([[[0.63139576+0.j, -0.35790232+0.j,  0.75722593+0.j],
                          [-0.35790232+0.j,  0.7875647 + 0.j,  0.19802138+0.j],
                          [0.75722593+0.j,  0.19802138+0.j,  1.5810395 + 0.j]],

                         [[1.4968449 + 0.j,  0.28304195+0.j,  0.6284405 + 0.j],
                          [0.28304195+0.j,  0.288077 + 0.j, -0.3535193 + 0.j],
                          [0.6284405 + 0.j, -0.3535193 + 0.j,  1.2150781 + 0.j]],

                         [[1.770809 + 0.j, -0.18107158+0.j,  0.5665751 + 0.j],
                          [-0.18107158+0.j,  0.62980217+0.j,  0.44762093+0.j],
                          [0.5665751 + 0.j,  0.44762093+0.j,  0.5993888 + 0.j]],

                         [[1.9772456 + 0.j,  0.06226069+0.j,  0.18060814+0.j],
                          [0.06226069+0.j,  0.45628715+0.j, -0.49417892+0.j],
                          [0.18060814+0.j, -0.49417892+0.j,  0.56646734+0.j]]])
        np.testing.assert_allclose(comp, self.tensor.multirot)


# Testing of class operator
class TestBuildVector:
    def setup(self):
        self.o = mt.Operator(2)
        self.o.matrix = np.arange(1, 5, dtype="float32").reshape(2, 2)
        self.o.build_vector()

    def test_shape(self):
        assert self.o.vector.shape == (4,)

    def test_value(self):
        comp = np.arange(1, 5)
        assert np.array_equal(comp, self.o.vector)

    def test_dtype(self):
        assert self.o.vector.dtype == "float32"


class TestSuperop:
    def setup(self):
        self.o = mt.Operator(2)
        self.o.matrix = np.diag(np.array([1, 2], dtype=np.float32))

    def test_value1(self):
        comp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0],
                         [0, 0, 0, 2]])
        self.o.build_superoperator()
        np.testing.assert_array_equal(comp, self.o.superop)

    def test_value2(self):
        self.o.build_superoperator(swap=True)
        comp = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, 2]])
        np.testing.assert_array_equal(comp, self.o.superop)

    def test_shape(self):
        self.o.build_superoperator()
        assert self.o.superop.shape == (4, 4)

    def test_dtype(self):
        self.o.build_superoperator()
        assert self.o.superop.dtype == "float32"


# Testing of class spinoperator
class TestPauliMatrices:
    def test_spin_1_2(self):
        s = mt.Spinoperator(1/2)
        sx, sy, sz = s.pauli_matrices(1/2)
        s = np.array([sx, sy, sz])
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        z = np.array([[1, 0], [0, -1]])
        ges = np.array([x, y, z])/2
        np.testing.assert_array_equal(ges, s)

    def test_spin_1(self):
        s = mt.Spinoperator(1)
        sx, sy, sz = s.pauli_matrices(1)
        s = np.array([sx, sy, sz])
        st = np.sqrt(2)
        x = np.array([[0, st, 0], [st, 0, st], [0, st, 0]])
        y = np.array([[0, -1j*st, 0], [1j*st, 0, -1j*st], [0, 1j*st, 0]])
        z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        ges = np.array([1/2*x, 1/2*y, z], dtype=np.complex64)
        np.testing.assert_array_equal(ges, s)

    def test_spin_3_2(self):
        s = mt.Spinoperator(3/2)
        sx, sy, sz = s.pauli_matrices(3/2)
        s = np.array([sx, sy, sz])
        st = np.sqrt(3)
        sf = np.sqrt(4)
        x = np.array([[0, st, 0, 0], [st, 0, sf, 0],
                      [0, sf, 0, st], [0, 0, st, 0]])
        y = np.array([[0, -1j*st, 0, 0], [1j*st, 0, -1j*2, 0],
                      [0, 1j*2, 0, -1j*st], [0, 0, 1j*st, 0]])
        z = np.array([[3, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, -1, 0], [0, 0, 0, -3]])
        ges = np.array([x, y, z], dtype=np.complex64)/2
        np.testing.assert_array_equal(ges, s)


class TestSpinoperatorInitUncoupled:
    def setup(self):
        self.s = mt.Spinoperator(3/2)

    def test_dimension(self):
        assert self.s.dimension == 4

    def test_matrix(self):
        comp = self.s.pauli_matrices(3/2)
        comp = np.array(comp)
        np.testing.assert_array_equal(comp, self.s.matrix)

    def test_matrix_dtype(self):
        assert self.s.matrix.dtype == "complex64"

    def test_matrix_shape(self):
        assert self.s.matrix.shape == (3, 4, 4)

    def test_vector(self):
        comp = self.s.matrix.flatten()
        np.testing.assert_array_equal(comp, self.s.vector)

    def test_superoperator(self):
        comp = np.kron(self.s.matrix, np.eye(4))
        np.testing.assert_array_equal(comp, self.s.superop)



class TestSpinoperatorInitCoupled:
    def setup(self):
        self.s = mt.Spinoperator(1, 1/2)

    def test_dimension(self):
        assert self.s.dimension == 6

    def test_matrix(self):
        comp = np.kron(self.s.pauli_matrices(1), np.eye(2))
        comp = np.array(comp)
        np.testing.assert_array_equal(comp, self.s.matrix)

    def test_matrix_dtype(self):
        assert self.s.matrix.dtype == "complex64"

    def test_matrix_shape(self):
        assert self.s.matrix.shape == (3, 6, 6)

    def test_matrix_coupling_spins(self):
        comp = np.kron(np.eye(3), self.s.pauli_matrices(1/2))
        comp = np.array(comp)
        np.testing.assert_array_equal(comp, self.s.matrix_coupling_spins[0])

    def test_matrix_coupling_spins_dtype(self):
        assert self.s.matrix_coupling_spins.dtype == "complex64"

    def test_matrix_coupling_spins_shape(self):
        assert self.s.matrix_coupling_spins.shape == (1, 3, 6, 6)

    def test_vector(self):
        comp = self.s.matrix.flatten()
        np.testing.assert_array_equal(comp, self.s.vector)

    def test_superoperator(self):
        comp = np.kron(self.s.matrix, np.eye(6))
        np.testing.assert_array_equal(comp, self.s.superop)


class TestGet:
    def setup(self):
        self.s = mt.Spinoperator(1/2)

    def test_get_x(self):
        comp = 1/2*np.array([[0, 1], [1, 0]])
        assert np.array_equal(comp, self.s.get('x'))

    def test_get_y(self):
        comp = 1/2*np.array([[0, -1j], [1j, 0]])
        assert np.array_equal(comp, self.s.get('y'))

    def test_get_z(self):
        comp = 1/2*np.array([[1, 0], [0, -1]])
        assert np.array_equal(comp, self.s.get('z'))

    def test_invalid_coordinate(self):
        with pt.raises(ValueError):
            self.s.get('w')


# Testing of class hamiltonian
class TestExchangeCoupling:
    def setup(self):
        self.h = mt.Hamiltonian(4)
        self.s1 = mt.Spinoperator(1/2, 1/2)
        self.s2 = mt.Spinoperator(1/2, 1/2)
        self.s2.matrix = self.s1.matrix_coupling_spins[0]
        self.j = 7
        self.h.exchange_coupling(self.s1, self.j, self.s2)

    def test_value(self):
        comp = -self.j*(0.5*np.eye(4)+2*(self.s1.get('x')@self.s2.get('x') +
                                         self.s1.get('y') @ self.s2.get('y') +
                                         self.s1.get('z')@self.s2.get('z')))
        np.testing.assert_array_equal(comp, self.h.matrix)

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))

    def test_shape(self):
        assert self.h.matrix.shape == (4, 4)

    def test_dtype(self):
        assert self.h.matrix.dtype == "complex64"

    def test_vector(self):
        comp = self.h.matrix.flatten()
        np.testing.assert_array_equal(comp, self.h.vector)

    def test_superoperator(self):
        comp = np.kron(self.h.matrix, np.eye(4))
        np.testing.assert_array_equal(comp, self.h.superop)

    def test_wrong_spinoperator_dimension(self):
        with pt.raises(IndexError):
            self.h.exchange_coupling(self.s1, self.j, mt.Spinoperator(1/2))


class TestMicrowaveCoupling:
    def setup(self):
        self.h = mt.Hamiltonian(2)
        self.s = mt.Spinoperator(1/2)
        self.omega_nut = 2
        self.omega_mw = -2
        self.h.microwave_coupling(self.omega_mw, self.omega_nut, self.s)

    def test_value(self):
        comp = np.ones((2, 2))
        comp[1, 1] *= -1
        np.testing.assert_array_equal(comp, self.h.matrix)

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))

    def test_shape(self):
        assert self.h.matrix.shape == (2, 2)

    def test_dtype(self):
        assert self.h.matrix.dtype == "complex64"

    def test_vector(self):
        comp = self.h.matrix.flatten()
        np.testing.assert_array_equal(comp, self.h.vector)

    def test_superoperator(self):
        comp = np.kron(self.h.matrix, np.eye(2))
        np.testing.assert_array_equal(comp, self.h.superop)


class Test_zeeman_coupling:
    def setup(self):
        self.h_dig = mt.Hamiltonian(2)
        self.h_rot = mt.Hamiltonian(2)
        self.g = mt.Tensor(np.arange(1, 4))
        self.g.rotation(1, 2)
        self.B_z = 340.1
        self.s = mt.Spinoperator(1/2)
        self.h_dig.zeeman_coupling(self.g.matrix, self.B_z, self.s)
        self.h_rot.zeeman_coupling(self.g.rot, self.B_z, self.s)

    def test_value_dig(self):
        comp = odh.create_linear_hamiltonian(self.g.matrix, self.s.matrix)
        comp *= self.B_z
        np.testing.assert_array_equal(comp, self.h_dig.matrix)

    def test_value_rot(self):
        comp = odh.create_linear_hamiltonian(self.g.rot, self.s.matrix)
        comp *= self.B_z
        np.testing.assert_array_equal(comp, self.h_rot.matrix)

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h_rot.matrix, np.transpose(np.conj(self.h_rot.matrix)))

    def test_shape(self):
        assert self.h_dig.matrix.shape == (2, 2)

    def test_dtype(self):
        assert self.h_dig.matrix.dtype == "complex64"

    def test_vector(self):
        comp = self.h_dig.matrix.flatten()
        np.testing.assert_array_equal(comp, self.h_dig.vector)

    def test_superoperator(self):
        comp = np.kron(self.h_dig.matrix, np.eye(2))
        np.testing.assert_array_equal(comp, self.h_dig.superop)

class TestDipolCoupling:
    def setup(self):
        self.h_dig = mt.Hamiltonian(4)
        self.h_rot = mt.Hamiltonian(3)
        self.d = mt.Tensor(np.arange(1, 4))
        self.d.rotation(1, 2)
        self.s1 = mt.Spinoperator(1/2, 1/2)
        self.s2 = mt.Spinoperator(1/2, 1/2)
        self.s2.matrix = self.s1.matrix_coupling_spins[0]
        self.h_dig.dipol_coupling(self.s1, self.d.matrix, self.s2)
        self.h_rot.dipol_coupling(self.s1, self.d.rot, self.s2)

    def test_value_dig(self):
        comp = odh.create_bilinear_hamiltonian(self.s1.matrix, self.d.matrix,
                                               self.s2.matrix)
        np.testing.assert_array_equal(self.h_dig.matrix, comp)

    def test_value_rot(self):
        comp = odh.create_bilinear_hamiltonian(self.s1.matrix, self.d.rot,
                                               self.s2.matrix)
        np.testing.assert_array_equal(self.h_rot.matrix, comp)

    def test_hermitean(self):
        assert np.array_equal(
            self.h_rot.matrix, np.transpose(np.conj(self.h_rot.matrix)))

    def test_shape(self):
        assert self.h_dig.matrix.shape == (4, 4)

    def test_dtype(self):
        assert self.h_dig.matrix.dtype == "complex64"

    def test_vector(self):
        comp = self.h_dig.matrix.flatten()
        np.testing.assert_array_equal(comp, self.h_dig.vector)

    def test_superoperator(self):
        comp = np.kron(self.h_dig.matrix, np.eye(4))
        np.testing.assert_array_equal(comp, self.h_dig.superop)

    def test_wrong_spinoperator_dimension(self):
        with pt.raises(IndexError):
            self.h_dig.dipol_coupling(self.s1, self.d.matrix,
                                      mt.Spinoperator(1/2))
