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
        self.m = mt.Matrix(3)
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


class Test_rotation:
    tensor = mt.Tensor(np.arange(3))
    unit = mt.Tensor([1, 1, 1])
    tensor.rotation(1, 2)
    unit.rotation(1, 2)

    def test_shape(self):
        assert self.tensor.rot.shape == (3, 3)

    def test_type(self):
        assert type(self.tensor.rot) == np.ndarray

    def test_eye(self):
        np.testing.assert_allclose(
            self.unit.matrix, self.unit.rot, atol=1e-8, rtol=1e-5)

    def test_values(self):
        comp = np.array([[1.77626649, -0.18920062,  0.48886663],
                         [-0.18920062,  0.29192658,  0.41341091],
                         [0.48886663,  0.41341091,  0.93180692]])
        np.testing.assert_allclose(comp, self.tensor.rot, atol=1e-8, rtol=1e-5)


class Test_multi_rotation:
    tensor = mt.Tensor(np.arange(3))
    theta, phi = grid.fibonacci_grid(4)
    tensor.multirotation(phi, theta)

    def test_shape(self):
        assert self.tensor.multirot.shape == (4, 3, 3)

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


class Test_build_vector:
    o = mt.Operator(2)
    o.matrix[0, 0] = 1
    o.matrix[0, 1] = 2
    o.matrix[1, 0] = 3
    o.matrix[1, 1] = 4
    o.build_vector()

    def test_shape(self):
        assert self.o.vector.shape == (4,)

    def test_value(self):
        comp = np.arange(1, 5)
        assert np.array_equal(comp, self.o.vector)

    def test_type(self):
        assert type(self.o.vector) == np.ndarray


class Test_superop:
    o = mt.Operator(2)
    o.matrix[0, 0] = 1
    o.matrix[1, 1] = 2

    def test_value1(self):
        comp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0],
                         [0, 0, 0, 2]])
        self.o.build_superoperator()
        assert np.array_equal(comp, self.o.superop)

    def test_value2(self):
        o = mt.Operator(2)
        o.matrix[0, 0] = 1
        o.matrix[1, 1] = 2
        o.build_superoperator(swap=True)
        comp = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, 2]])
        assert np.array_equal(comp, o.superop)


# Testing of class spinoperator
class Test_spinoperator_init:
    def test_spin_1_2(self):
        s = mt.Spinoperator(1/2)
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        z = np.array([[1, 0], [0, -1]])
        ges = np.array([x, y, z])
        assert np.array_equal(ges, 2*s.matrix)

    def test_spin_1(self):
        s = mt.Spinoperator(1)
        st = np.sqrt(2)
        x = np.array([[0, st, 0], [st, 0, st], [0, st, 0]])
        y = np.array([[0, -1j*st, 0], [1j*st, 0, -1j*st], [0, 1j*st, 0]])
        z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        ges = np.array([1/2*x, 1/2*y, z], dtype=np.complex64)
        assert np.array_equal(ges, s.matrix)

    def test_spin_3_2(self):
        s = mt.Spinoperator(3/2)
        st = np.sqrt(3)
        sf = np.sqrt(4)
        x = np.array([[0, st, 0, 0], [st, 0, sf, 0],
                      [0, sf, 0, st], [0, 0, st, 0]])
        y = np.array([[0, -1j*st, 0, 0], [1j*st, 0, -1j*2, 0],
                      [0, 1j*2, 0, -1j*st], [0, 0, 1j*st, 0]])
        z = np.array([[3, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, -1, 0], [0, 0, 0, -3]])
        ges = np.array([x, y, z], dtype=np.complex64)
        assert np.array_equal(ges, 2*s.matrix)

    def test_spin_1_and_1_2(self):
        s = mt.Spinoperator(1, 1/2)
        spin_1 = mt.Spinoperator(1)
        spin_1_2 = mt.Spinoperator(1/2)
        matrix_spin_1 = np.kron(spin_1.matrix, np.eye(2))
        matrix_spin_1 = matrix_spin_1[np.newaxis, :, :, :]
        matrix_spin_2 = np.kron(np.eye(3), spin_1_2.matrix)
        matrix_spin_2 = matrix_spin_2[np.newaxis, :, :, :]
        assert np.allclose(matrix_spin_1, s.matrix)
        assert np.allclose(matrix_spin_2, s.matrix_coupling_spins)

    def test_spin_1_2_and_1_and_1_2(self):
        s_1_2 = mt.Spinoperator(1/2)
        s_1 = mt.Spinoperator(1)
        s = mt.Spinoperator(1/2, [1, 1/2])

        matrix_core_1 = np.kron(s_1.matrix, np.eye(2))
        matrix_core_1 = np.kron(np.eye(2), matrix_core_1)
        matrix_core_2 = np.kron(np.eye(3), s_1_2.matrix)
        matrix_core_2 = np.kron(np.eye(2), matrix_core_2)
        matrix_spin = np.kron(s_1_2.matrix, np.eye(6))

        np.testing.assert_allclose(s.matrix_coupling_spins[0], matrix_core_1)
        np.testing.assert_allclose(s.matrix_coupling_spins[1], matrix_core_2)
        np.testing.assert_allclose(s.matrix, matrix_spin)


class Test_get:
    s = mt.Spinoperator(1/2)

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


class Test_total_spin_radpair:
    s = mt.Spinoperator(1/2)

    def test_value(self):
        comp = np.array([[[0. + 0j,  0.5+0j,  0.5+0j,  0. + 0j],
                          [0.5+0j,  0. + 0j,  0. + 0j,  0.5+0j],
                          [0.5+0j,  0. + 0j,  0. + 0j,  0.5+0j],
                          [0. + 0j,  0.5+0j,  0.5+0j,  0. + 0j]],

                         [[0. + 0j,  0. - 0.5j,  0. - 0.5j,  0. + 0j],
                          [0. + 0.5j,  0. + 0j,  0. + 0j,  0. - 0.5j],
                          [0. + 0.5j,  0. + 0j,  0. + 0j,  0. - 0.5j],
                          [0. + 0j,  0. + 0.5j,  0. + 0.5j,  0. + 0j]],

                         [[1. + 0j,  0. + 0j,  0. + 0j,  0. + 0j],
                          [0. + 0j,  0. + 0j,  0. + 0j,  0. + 0j],
                          [0. + 0j,  0. + 0j,  0. + 0j,  0. + 0j],
                          [0. + 0j,  0. + 0j,  0. + 0j, -1. + 0j]]])
        self.s.total_spin_radpair()
        assert np.array_equal(comp, self.s.matrix)


# Testing of class hamiltonian
class Test_create_linear_operator:
    h = mt.Hamiltonian(3)
    s = mt.Spinoperator(1)
    g = mt.Tensor(np.arange(1, 4))

    def test_value(self):
        comp = odh.create_linear_hamiltonian(self.g.matrix, self.s.matrix)
        self.h.create_linear_operator(self.g.matrix, self.s)
        assert np.array_equal(comp, self.h.matrix)

    def test_rotated_tensor(self):
        self.g.rotation(1, 2)
        comp = odh.create_linear_hamiltonian(self.g.rot, self.s.matrix)
        self.h.create_linear_operator(self.g.rot, self.s)
        assert np.array_equal(comp, self.h.matrix)


class Test_create_bilinear_operator:
    h = mt.Hamiltonian(4)
    s = mt.Spinoperator(1/2)
    s.total_spin_radpair()
    D = mt.Tensor(np.arange(1, 4))

    def test_value(self):
        comp = odh.create_bilinear_hamiltonian(
            self.s.matrix, self.D.matrix, self.s.matrix)
        self.h.create_bilinear_operator(self.s, self.D.matrix, self.s)
        assert np.array_equal(comp, self.h.matrix)

    def test_rotated_tensor(self):
        self.D.rotation(1, 2)
        comp = odh.create_bilinear_hamiltonian(
            self.s.matrix, self.D.rot, self.s.matrix)
        self.h.create_bilinear_operator(self.s, self.D.rot, self.s)
        assert np.array_equal(comp, self.h.matrix)


class Test_exchange_coupling:
    h = mt.Hamiltonian(4)
    s1 = mt.Spinoperator(1/2)
    s1.build_superoperator()
    s1.matrix = s1.superop
    s2 = mt.Spinoperator(1/2)
    s2.build_superoperator(swap=True)
    s2.matrix = s2.superop
    j = 7
    h.exchange_coupling(s1, j, s2)

    def test_value(self):
        comp = -self.j*(0.5*np.eye(4)+2*(self.s1.get('x')@self.s2.get('x') +
                                         self.s1.get('y') @ self.s2.get('y') +
                                         self.s1.get('z')@self.s2.get('z')))
        assert np.array_equal(comp, self.h.matrix)

    def test_hermitesch(self):
        assert np.array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))


class Test_microwave_coupling:
    h = mt.Hamiltonian(2)
    s = mt.Spinoperator(1/2)
    omega_nut = 12.4
    omega_mw = 9.7
    h.microwave_coupling(omega_mw, omega_nut, s)

    def test_value(self):
        comp = -self.omega_mw*self.s.get('z')+self.omega_nut*self.s.get('x')
        assert np.array_equal(comp, self.h.matrix)

    def test_hermitesch(self):
        assert np.array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))


class Test_zeeman_coupling:
    h = mt.Hamiltonian(2)
    g = mt.Tensor(np.arange(1, 4))
    B_z = 340.1
    s = mt.Spinoperator(1/2)
    h.zeeman_coupling(g.matrix, B_z, s)

    def test_value(self):
        comp = mt.Hamiltonian(2)
        comp.create_linear_operator(self.g.matrix, self.s)
        comp.matrix *= self.B_z
        assert np.array_equal(comp.matrix, self.h.matrix)

    def test_hermitesch(self):
        assert np.array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))


class Test_dipol_coupling:
    h = mt.Hamiltonian(4)
    d = mt.Tensor(np.arange(1, 4))
    S = mt.Spinoperator(1/2)
    S.total_spin_radpair()
    h.dipol_coupling(d.matrix, S)

    def test_value(self):
        comp = mt.Hamiltonian(4)
        comp.create_bilinear_operator(self.S, self.d.matrix, self.S)
        assert np.array_equal(self.h.matrix, comp.matrix)

    def test_hermitesch(self):
        assert np.array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))
