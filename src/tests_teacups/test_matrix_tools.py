import sys
sys.path.append("./..")

import pytest as pt
import teacups.matrix_tools as mt
import numpy as np
import teacups.orientation_dependent_ham as odh
import teacups.grid as grid


# Testing of class matrix

class Test_scalar:
    m = mt.Matrix(2)
    m.matrix = np.arange(1, 5).reshape((2, 2))
    m.scalar(np.array([2, 3]))

    def test_value(self):
        comp = np.array([[6, 12], [18, 24]])
        assert np.array_equal(comp, self.m.matrix)


class Test_product:
    def test_wrong_shape(self):
        m = mt.Matrix(2)
        m.matrix = np.arange(1, 5).reshape((2, 2))
        with pt.raises(IndexError):
            m.product(np.eye(3))

    def test_value_right(self):
        m = mt.Matrix(2)
        m.matrix = np.arange(1, 5).reshape((2, 2))
        scd_matrix = np.array(np.arange(2, 6).reshape((2, 2)))
        m.product(scd_matrix)
        comp = np.array([[10, 13], [22, 29]])
        assert np.array_equal(comp, m.matrix)

    def test_value_left(self):
        m = mt.Matrix(2)
        m.matrix = np.arange(1, 5).reshape((2, 2))
        scd_matrix = np.array(np.arange(2, 6).reshape((2, 2)))
        m.product(scd_matrix, left=True)
        comp = np.array([[11, 16], [19, 28]])
        assert np.array_equal(comp, m.matrix)


class Test_basis_transformation:
    vec1 = np.arange(0, 16).reshape((4, 4))
    vec2 = np.arange(2, 18).reshape((4, 4))
    m1 = np.array([[1, 3, 5, 9], [14, 13, 0, 15], [7, 2, 18, 4], [1, 3, 3, 4]])
    b = 1/np.sqrt(2)
    m2 = np.array([[1, 0, 0, 0], [0, b, b, 0], [0, -b, b, 0], [0, 0, 0, 1]])

    def test_value1(self):
        test = mt.Matrix(4)
        test.matrix = self.vec1
        test.basis_transformation(self.m1)
        comp = np.linalg.inv(self.m1) @ self.vec1 @ self.m1
        assert np.array_equal(comp, test.matrix)

    def test_value2(self):
        test = mt.Matrix(4)
        test.matrix = self.vec1
        test.basis_transformation(self.m2, orthonormal=True)
        comp = np.linalg.inv(self.m2) @ self.vec1 @ self.m2
        np.testing.assert_allclose(comp, test.matrix, atol=1e-8, rtol=1e-5)

    def test_optional_argument(self):
        test = mt.Matrix(4)
        test.matrix = self.vec1
        test.basis_transformation(self.m1, orthonormal=True)
        comp = self.m1.T @ self.vec1 @ self.m1
        assert np.array_equal(comp, test.matrix)

# Testing of class tensor


class Test_tensor__init__:
    ten = mt.Tensor(np.arange(3))

    def test_value(self):
        comp = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
        assert np.array_equal(comp, self.ten.tensor)


class Test_rotation:
    tensor = mt.Tensor(np.arange(3))
    unit = mt.Tensor(np.eye(3))
    tensor.rotation(1, 2)
    unit.rotation(1, 2)

    def test_shape(self):
        assert self.tensor.rot.shape == (3, 3)

    def test_type(self):
        assert type(self.tensor.rot) == np.ndarray

    def test_eye(self):
        np.testing.assert_allclose(
            self.unit.tensor, self.unit.rot, atol=1e-8, rtol=1e-5)

    def test_values(self):
        comp = np.array([[1.77626649, -0.18920062,  0.48886663],
                         [-0.18920062,  0.29192658,  0.41341091],
                         [0.48886663,  0.41341091,  0.93180692]])
        np.testing.assert_allclose(comp, self.tensor.rot, atol=1e-8, rtol=1e-5)


class Test_multi_rotation:
    tensor = mt.Tensor(np.arange(3))
    theta, phi = grid.get_theta_phi(4)
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
        comp = odh.create_linear_hamiltonian(self.g.tensor, self.s.matrix)
        self.h.create_linear_operator(self.g.tensor, self.s)
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
            self.s.matrix, self.D.tensor, self.s.matrix)
        self.h.create_bilinear_operator(self.s, self.D.tensor, self.s)
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
    h.zeeman_coupling(g.tensor, B_z, s)

    def test_value(self):
        comp = mt.Hamiltonian(2)
        comp.create_linear_operator(self.g.tensor, self.s)
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
    h.dipol_coupling(d.tensor, S)

    def test_value(self):
        comp = mt.Hamiltonian(4)
        comp.create_bilinear_operator(self.S, self.d.tensor, self.S)
        assert np.array_equal(self.h.matrix, comp.matrix)

    def test_hermitesch(self):
        assert np.array_equal(
            self.h.matrix, np.transpose(np.conj(self.h.matrix)))
