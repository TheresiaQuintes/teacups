import sys
sys.path.append("./..")

import numpy as np
import teacups.matrix_tools as mt
import teacups.multioperator_tools as mut
import teacups.orientation_dependent_ham as odh
import teacups.grid as grid

# Testing of class Multimatrix
class TestInitMultimatrix:
    def setup(self):
        self.dimension = 2
        self.gp = 4
        self.B = np.array([1, 2, 3])
        self.m = mut.Multimatrix(self.dimension, self.gp, self.B)

    def test_dimensions(self):
        assert self.m.dimension == 2
        assert self.m.angle_shape == (4, 2, 2)
        assert self.m.B_angle_shape == (3, 4, 2, 2)

    def test_matrix(self):
        assert self.m.matrix.shape == (2, 2)
        assert self.m.matrix.dtype == "complex64"

    def test_angle_matrix(self):
        assert self.m.angle_matrix.shape == (4, 2, 2)
        assert self.m.angle_matrix.dtype == "complex64"

    def test_B_angle_matrix(self):
        assert self.m.B_angle_matrix.shape == (3, 4, 2, 2)
        assert self.m.B_angle_matrix.dtype == "complex64"



class Test_init_multioperator:
    s = mt.Spinoperator(1/2)
    ten = mt.Tensor(np.arange(1, 4))
    theta, phi = grid.fibonacci_grid(3)
    ten.multirotation(phi, theta)
    B = np.linspace(2, 3, 2)
    grid_points = 3

    a = np.arange(1, 5).reshape((2, 2))
    b = np.array([[a, a, a],
                  [a, a, a],
                  [a, a, a]])
    c = np.array([b, b])

    def test_built_objects(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        assert h.dimension == 2
        assert h.angle_shape == (3, 2, 2)
        assert h.B_angle_shape == (2, 3, 2, 2)
        assert np.array_equal(h.matrix, np.zeros((2, 2)))
        assert np.array_equal(h.angle_matrix, np.zeros((3, 2, 2)))
        assert np.array_equal(h.B_angle_matrix, np.zeros((2, 3, 2, 2)))

    def test_complex_type(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        assert type(h.matrix[0, 0]) == np.complex64
        assert type(h.angle_matrix[0, 0, 0]) == np.complex64
        assert type(h.B_angle_matrix[0, 0, 0, 0]) == np.complex64


class Test_get_and_change:
    s = mt.Spinoperator(1/2)
    ten = mt.Tensor(np.arange(1, 4))
    theta, phi = grid.fibonacci_grid(3)
    ten.multirotation(phi, theta)
    B = np.linspace(2, 3, 2)
    grid_points = 3

    a = np.arange(1, 5).reshape((2, 2))
    b = np.array([a, a, a])
    c = np.array([b, b])

    def test_matrix_changed(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = self.a
        h.matrix_changed()
        assert np.array_equal(h.matrix, self.a)
        assert np.array_equal(h.angle_matrix, self.b)
        assert np.array_equal(h.B_angle_matrix, self.c)

    def test_angle_matrix_changed(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.angle_matrix = self.b
        h.angle_matrix_changed()
        assert np.array_equal(h.angle_matrix, self.b)
        assert np.array_equal(h.B_angle_matrix, self.c)


class Test_multiplications:
    s = mt.Spinoperator(1/2)
    ten = mt.Tensor(np.arange(1, 4))
    theta, phi = grid.fibonacci_grid(3)
    ten.multirotation(phi, theta)
    B = np.linspace(2, 3, 2)
    grid_points = 3

    a = np.arange(1, 5).reshape((2, 2))
    b = np.array([a, a, a])
    c = np.array([b, b])

    p = np.arange(2, 6).reshape(2, 2)

    def test_scalar(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        h.scalar(np.array([2, 3]))
        assert np.array_equal(h.B_angle_matrix, self.c*6)

    def test_scalar_varied_multimat(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        h.B_angle_matrix[:, 0, 0] = np.ones((2, 2))
        self.c[:, 0, 0] = np.ones((2, 2))
        h.scalar(np.array([2, 3]))
        assert np.array_equal(h.B_angle_matrix, self.c*6)

    def test_product_right(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        h.product(self.p)
        d = np.array([[10, 13], [22, 29]])
        e = np.array([d, d, d])
        f = np.array([e, e])
        assert np.array_equal(h.B_angle_matrix, f)

    def test_product_left(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        h.product(self.p, left=True)
        d = np.array([[11, 16], [19, 28]])
        e = np.array([d, d, d])
        f = np.array([e, e])
        assert np.array_equal(h.B_angle_matrix, f)

    def test_basistransformation_inv(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        t = np.array([[0, 1], [1, 0]])
        d = np.array([[4, 3], [2, 1]])
        e = np.array([d, d, d])
        f = np.array([e, e])
        h.basistransformation(t)
        assert np.array_equal(h.B_angle_matrix, f)

    def test_basistransformation_T(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        t = np.array([[0, 1], [1, 0]])
        d = np.array([[4, 3], [2, 1]])
        e = np.array([d, d, d])
        f = np.array([e, e])
        h.basistransformation(t, orthonormal=True)
        assert np.array_equal(h.B_angle_matrix, f)


class Test_build_vector:
    s = mt.Spinoperator(1/2)
    ten = mt.Tensor(np.arange(1, 4))
    theta, phi = grid.fibonacci_grid(3)
    ten.multirotation(phi, theta)
    B = np.linspace(2, 3, 2)
    grid_points = 3

    def test_value(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        h.build_vector()
        d = np.arange(1, 5)
        e = np.array([d, d, d])
        f = np.array([e, e])
        assert np.array_equal(h.B_angle_vector, f)


class Test_build_superoperator:
    s = mt.Spinoperator(1/2)
    ten = mt.Tensor(np.arange(1, 4))
    theta, phi = grid.fibonacci_grid(3)
    ten.multirotation(phi, theta)
    B = np.linspace(2, 3, 2)
    grid_points = 3

    def test_value(self):
        h = mut.Multioperator(self.s, self.grid_points, self.B)
        h.matrix = np.arange(1, 5).reshape((2, 2))
        h.matrix_changed()
        h.build_superoperator(swap=True)
        d = np.array([[1., 2., 0., 0.],
                      [3., 4., 0., 0.],
                      [0., 0., 1., 2.],
                      [0., 0., 3., 4.]])
        e = np.array([d, d, d])
        f = np.array([e, e])
        assert np.array_equal(h.B_angle_superop, f)

        h.build_superoperator()
        d = np.array([[1., 0., 2., 0.],
                      [0., 1., 0., 2.],
                      [3., 0., 4., 0.],
                      [0., 3., 0., 4.]])
        e = np.array([d, d, d])
        f = np.array([e, e])
        assert np.array_equal(h.B_angle_superop, f)


class Test_create_linear_operator:
    s = mt.Spinoperator(1/2)
    g = mt.Tensor(np.array([1, 2, 3]))
    theta, phi = grid.fibonacci_grid(3)
    g.multirotation(phi, theta)
    grid_points = 3
    h = mut.Multioperator(s, grid_points, np.linspace(1, 2, 2))

    def test_value(self):
        comp = np.array([odh.create_linear_hamiltonian(self.g.multirot[0],
                                                       self.s.matrix),
                         odh.create_linear_hamiltonian(self.g.multirot[1],
                                                       self.s.matrix),
                         odh.create_linear_hamiltonian(self.g.multirot[2],
                                                       self.s.matrix)])
        self.h.create_linear_operator(self.g)
        np.testing.assert_allclose(comp, self.h.angle_matrix)


class Test_create_bilinear_operator:
    s = mt.Spinoperator(1/2)
    g = mt.Tensor(np.array([1, 2, 3]))
    theta, phi = grid.fibonacci_grid(3)
    g.multirotation(phi, theta)
    grid_points = 3
    h = mut.Multioperator(s, grid_points, np.linspace(1, 2, 2))
    s2 = mt.Spinoperator(1/2)
    s2.matrix = np.arange(1, 13).reshape((3, 2, 2))

    def test_value(self):
        comp = np.array([odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[0], self.s2.matrix),
            odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[1], self.s2.matrix),
            odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[2], self.s2.matrix)])
        self.h.create_bilinear_operator(self.g, self.s2)
        np.testing.assert_allclose(comp, self.h.angle_matrix)


class Test_zeeman_coupling:
    s = mt.Spinoperator(1/2)
    g = mt.Tensor(np.array([1, 2, 3]))
    theta, phi = grid.fibonacci_grid(2)
    g.multirotation(phi, theta)
    B = np.linspace(1, 4, 4)
    grid_points = 2
    h = mut.Multioperator(s, grid_points, B)

    def test_value(self):
        self.h.zeeman_coupling(self.g)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[0], self.h.angle_matrix*1)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[1], self.h.angle_matrix*2)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[2], self.h.angle_matrix*3)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[3], self.h.angle_matrix*4)


class Test_isotropic_couplings:
    s = mt.Spinoperator(1/2)
    g = mt.Tensor(np.array([1, 2, 3]))
    theta, phi = grid.fibonacci_grid(2)
    g.multirotation(phi, theta)
    B = np.linspace(1, 4, 4)
    grid_points = 2
    h = mut.Multioperator(s, grid_points, B)

    def test_microwave_coupling(self):
        self.h.microwave_coupling(2, 2)
        comp = np.array([[-1, 1], [1, 1]])
        assert np.array_equal(self.h.matrix, comp)

    def test_exchange_coupling(self):
        spinop2 = mt.Spinoperator(1/2)
        spinop2.matrix *= 3
        self.h.exchange_coupling(3, spinop2)
        comp = mt.Hamiltonian(2)
        comp.exchange_coupling(self.h.spinop, 3, spinop2)
        assert np.array_equal(comp.matrix, self.h.matrix)
