import sys
sys.path.append("./..")

import numpy as np
import teacups.matrix_tools as mt
import teacups.multioperator_tools as mut
import teacups.orientation_dependent_ham as odh
import teacups.grid as grid
import pytest as pt

# Testing of class Multimatrix
class TestInitMultimatrix:
    def setup(self):
        self.dimension = 2
        self.gp = 4
        self.B = 3
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

class TestMatrixChanged:
    def setup(self):
        self.a = np.arange(1, 5).reshape((2, 2))
        self.b = np.array([self.a, self.a, self.a])
        self.c = np.array([self.b, self.b])

        dimension = 2
        gp = 3
        b = 2

        self.m = mut.Multimatrix(dimension, gp, b)

    def test_matrix_changed(self):
        self.m.matrix = self.a
        self.m.matrix_changed()
        np.testing.assert_array_equal(self.m.angle_matrix, self.b)
        np.testing.assert_array_equal(self.m.B_angle_matrix, self.c)

    def test_angle_matrix_changed(self):
        self.m.angle_matrix = self.b
        self.m.angle_matrix_changed()
        np.testing.assert_array_equal(self.m.B_angle_matrix, self.c)

class TestScalar:
    def setup(self):
        self.m = mut.Multimatrix(2, 4, 3)
        self.m.matrix = np.arange(1, 5).reshape((2, 2))
        self.m.matrix = self.m.matrix.astype(np.complex64)
        self.m.matrix_changed()
        self.m.scalar(np.array([2, 3]))

    def test_value(self):
        comp = np.array([[6, 12], [18, 24]])
        comp = np.array([comp, comp, comp, comp])
        comp = np.array([comp, comp, comp])
        np.testing.assert_array_equal(comp, self.m.B_angle_matrix)

    def test_dtype(self):
        assert self.m.B_angle_matrix.dtype == "complex64"


class TestProduct:
    def setup(self):
        self.m = mut.Multimatrix(2, 3, 2)
        self.m.matrix = np.arange(1, 5).reshape((2, 2))
        self.m.matrix = self.m.matrix.astype(np.complex64)
        self.m.matrix_changed()

        self.scd_matrix_matrix = np.array(
            np.arange(2, 6).reshape((2, 2)), dtype=np.complex64)
        self.scd_matrix_angle_matrix = np.array(
            np.arange(2, 14).reshape((3, 2, 2)), dtype=np.complex64)
        self.scd_matrix_B_angle_matrix = np.array(
            np.arange(2, 26).reshape((2, 3, 2, 2)), dtype=np.complex64)

    def test_wrong_shape(self):
        with pt.raises(IndexError):
            self.m.product(np.eye(3))

    def test_product_right(self):
        self.m.product(self.scd_matrix_matrix)
        comp = np.array([[10, 13], [22, 29]])
        comp = np.array([comp, comp, comp])
        comp = np.array([comp, comp])

        np.testing.assert_array_equal(comp, self.m.B_angle_matrix)
        assert self.m.B_angle_matrix.dtype == "complex64"

    def test_product_left(self):
        self.m.product(self.scd_matrix_matrix, left=True)
        comp = np.array([[11, 16], [19, 28]])
        comp = np.array([comp, comp, comp])
        comp = np.array([comp, comp])

        np.testing.assert_array_equal(comp, self.m.B_angle_matrix)
        assert self.m.B_angle_matrix.dtype == "complex64"

    def test_product_angle_matrix(self):
        comp1 = np.array([[10, 13], [22, 29]])
        comp2 = np.array([[22, 25], [50, 57]])
        comp3 = np.array([[34, 37], [78, 85]])

        comp = np.array([comp1, comp2, comp3])
        comp = np.array([comp, comp])

        self.m.product(self.scd_matrix_angle_matrix)
        np.testing.assert_array_equal(self.m.B_angle_matrix, comp)

    def test_product_B_angle_matrix(self):
        comp1 = np.array([[10, 13], [22, 29]])
        comp2 = np.array([[22, 25], [50, 57]])
        comp3 = np.array([[34, 37], [78, 85]])
        comp4 = np.array([[46, 49], [106, 113]])
        comp5 = np.array([[58, 61], [134, 141]])
        comp6 = np.array([[70, 73], [162, 169]])

        comp1a = np.array([comp1, comp2, comp3])
        comp2a = np.array([comp4, comp5, comp6])
        comp = np.array([comp1a, comp2a])

        self.m.product(self.scd_matrix_B_angle_matrix)
        np.testing.assert_array_equal(self.m.B_angle_matrix, comp)

class TestBasisTransformation:
    def setup(self):
        self.m = mut.Multimatrix(4, 3, 2)
        self.m.matrix = np.array([[1+2j, 2+3j, 7+1j, 3+0j],
                                 [2+0j, 4+1j, 3+0j, 4-1j],
                                 [10+7j, 5-2j, 8-4j, 10+10j],
                                 [2+1j, 0+1j, 3-7j, 8+0j]],
                                 dtype=np.complex64)
        self.m.matrix_changed()
        self.eig, self.vec = np.linalg.eig(self.m.matrix)
        # auch für eig(self.m.B_angle_matrix)
        self.eigm = np.diag(self.eig)
        self.eigm = np.array([self.eigm, self.eigm, self.eigm])
        self.eigm = np.array([self.eigm, self.eigm])

        b = 1j/np.sqrt(2)
        self.ort_trans = np.array([[1, 0, 0, 0], [0, b, b, 0],
                                   [0, -b, b, 0], [0, 0, 0, 1]])

    def test_wrong_shape(self):
        with pt.raises(IndexError):
            self.m.basis_transformation(np.eye(3))

    def test_inverse_left(self):
        self.m.basis_transformation(self.vec)
        np.testing.assert_allclose(self.m.B_angle_matrix, self.eigm, atol=2e-6)
        assert self.m.B_angle_matrix.dtype == "complex64"

    def test_inverse_right(self):
        c = mut.Multimatrix(4, 3, 2)
        c.B_angle_matrix = self.eigm
        c.basis_transformation(self.vec, inverse_left=False)
        np.testing.assert_allclose(c.B_angle_matrix, self.m.B_angle_matrix,
                                   atol=2e-6)
        assert c.matrix.dtype == "complex64"

    def test_inverse_left_orthonormal(self):
        comp = np.conj(self.vec.T) @ self.m.B_angle_matrix @ self.vec
        self.m.basis_transformation(self.vec, orthonormal=True)
        np.testing.assert_allclose(
            self.m.B_angle_matrix, comp)
        assert self.m.B_angle_matrix.dtype == "complex64"

    def test_inverse_right_orhtonormal(self):
        comp = self.vec @ self.eigm @ np.conj(self.vec.T)

        c = mut.Multimatrix(4, 3, 2)
        c.B_angle_matrix = self.eigm
        c.basis_transformation(self.vec, inverse_left=False, orthonormal=True)
        np.testing.assert_allclose(c.B_angle_matrix, comp)
        assert c.matrix.dtype == "complex64"

    def test_orthonormal_identical(self):
        normal = self.m
        onormal = self.m
        normal.basis_transformation(self.ort_trans, orthonormal=False)
        onormal.basis_transformation(self.ort_trans, orthonormal=True)
        np.testing.assert_allclose(normal.B_angle_matrix,
                                   onormal.B_angle_matrix)

    def test_transformation_angle_matrix(self):
        _, vec = np.linalg.eig(self.m.angle_matrix)
        self.m.basis_transformation(vec)
        np.testing.assert_allclose(self.m.B_angle_matrix, self.eigm, atol=2e-6)

    def test_transformation_B_angle_matrix(self):
        _, vec = np.linalg.eig(self.m.B_angle_matrix)
        self.m.basis_transformation(vec)
        np.testing.assert_allclose(self.m.B_angle_matrix, self.eigm, atol=2e-6)

# Testing of class Multioperator

class TestBuildVector:
    def setup(self):
        self.o = mut.Multioperator_(2, 3, 2)
        self.o.B_angle_matrix = np.arange(2, 26, dtype=np.float32
                                          ).reshape((2, 3, 2, 2))
        self.o.build_vector()

    def test_shape(self):
        assert self.o.B_angle_vector.shape == (2, 3, 4)

    def test_value(self):
        comp = np.arange(2, 26).reshape((2, 3, 4))
        np.testing.assert_array_equal(comp, self.o.B_angle_vector)

    def test_dtype(self):
        assert self.o.B_angle_vector.dtype == "float32"


class TestSuperop:
    def setup(self):
        self.o = mut.Multioperator_(2, 3, 2)
        self.o.matrix = np.diag(np.array([1, 2], dtype=np.float32))
        self.o.matrix_changed()

    def test_value1(self):
        comp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0],
                         [0, 0, 0, 2]])
        comp = np.array([comp, comp, comp])
        comp = np.array([comp, comp])
        self.o.build_superoperator()
        np.testing.assert_array_equal(comp, self.o.B_angle_superop)

    def test_value2(self):
        self.o.build_superoperator(swap=True)
        comp = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, 2]])
        comp = np.array([comp, comp, comp])
        comp = np.array([comp, comp])
        np.testing.assert_array_equal(comp, self.o.B_angle_superop)

    def test_shape(self):
        self.o.build_superoperator()
        assert self.o.B_angle_superop.shape == (2, 3, 4, 4)

    def test_dtype(self):
        self.o.build_superoperator()
        assert self.o.B_angle_superop.dtype == "float32"


# Testing of class Multioperator
class TestCreateLinearOperator:
    def setup(self):
        self.s = mt.Spinoperator(1/2)
        self.g = mt.Tensor(np.array([1, 2, 3]))
        theta = np.linspace(0, 0.9, 5)*np.pi
        phi = np.linspace(0.1, 1.5, 5)*np.pi
        self.g.multirotation(phi, theta)
        grid_points = 5
        self.h = mut.Multioperator(self.s, grid_points, np.linspace(1, 2, 2))
        self.h.create_linear_operator(self.g)

    def test_value(self):
        comp = np.array([odh.create_linear_hamiltonian(self.g.multirot[0],
                                                       self.s.matrix),
                         odh.create_linear_hamiltonian(self.g.multirot[1],
                                                       self.s.matrix),
                         odh.create_linear_hamiltonian(self.g.multirot[2],
                                                       self.s.matrix),
                        odh.create_linear_hamiltonian(self.g.multirot[3],
                                                      self.s.matrix),
                        odh.create_linear_hamiltonian(self.g.multirot[4],
                                                      self.s.matrix)])
        np.testing.assert_allclose(comp, self.h.angle_matrix)

    def test_dtype(self):
        assert self.h.angle_matrix.dtype == "complex64"

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h.angle_matrix,
            np.conj(np.transpose(self.h.angle_matrix, (0, 2, 1))))

    def test_vector(self):
        self.h.B_angle_vector.shape == (2, 5, 4)

    def test_superop(self):
        self.h.B_angle_superop.shape == (2, 5, 4, 4)


class TestCreateBilinearOperator:
    def setup(self):
        self.s = mt.Spinoperator(1/2)
        self.g = mt.Tensor(np.array([1, 2, 3]))
        theta = np.linspace(0, 0.9, 5)*np.pi
        phi = np.linspace(0.1, 1.5, 5)*np.pi
        self.g.multirotation(phi, theta)
        grid_points = 5
        self.h = mut.Multioperator(self.s, grid_points, np.linspace(1, 2, 2))
        self.s2 = mt.Spinoperator(1/2)
        self.s2.matrix = np.arange(1, 21).reshape((5, 2, 2))
        self.h.create_bilinear_operator(self.g, self.s2)

    def test_value(self):
        comp = np.array([odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[0], self.s2.matrix),
            odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[1], self.s2.matrix),
            odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[2], self.s2.matrix),
            odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[3], self.s2.matrix),
            odh.create_bilinear_hamiltonian(
            self.s.matrix, self.g.multirot[4], self.s2.matrix)])

        np.testing.assert_allclose(comp, self.h.angle_matrix)

    def test_dtype(self):
        assert self.h.angle_matrix.dtype == "complex64"

    def test_wrong_spinoperator_dimension(self):
        with pt.raises(IndexError):
            self.h.create_bilinear_operator(self.g, mt.Spinoperator(1))

    def test_hermitean(self):
        s2 = mt.Spinoperator(1/2)
        self.h.create_bilinear_operator(self.g, s2)
        np.testing.assert_array_equal(
            self.h.angle_matrix,
            np.conj(np.transpose(self.h.angle_matrix, (0, 2, 1))))

    def test_vector(self):
        self.h.B_angle_vector.shape == (2, 5, 4)

    def test_superop(self):
        self.h.B_angle_superop.shape == (2, 5, 4, 4)


class TestExchangeCoupling:
    def setup(self):
        self.s1 = mt.Spinoperator(1/2, 1/2)
        self.s2 = mt.Spinoperator(1/2, 1/2)
        self.s2.matrix = self.s1.matrix_coupling_spins[0]
        self.j = 7
        grid_points = 3
        B = np.linspace(1, 4, 2)
        self.h = mut.Multioperator(self.s1, grid_points, B)
        self.h.exchange_coupling(self.j, self.s2)

    def test_wrong_spinoperator_dimension(self):
        with pt.raises(IndexError):
            self.h.exchange_coupling(self.j, mt.Spinoperator(1))

    def test_value(self):
        comp = self.j*(self.s1.get('x')@self.s2.get('x') +
                                         self.s1.get('y') @ self.s2.get('y') +
                                         self.s1.get('z')@self.s2.get('z'))
        comp = np.array([comp, comp, comp])
        comp = np.array([comp, comp])

        np.testing.assert_array_equal(comp, self.h.B_angle_matrix)

    def test_dtype(self):
        assert self.h.B_angle_matrix.dtype == "complex64"

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h.B_angle_matrix,
            np.conj(np.transpose(self.h.B_angle_matrix, (0, 1, 3, 2))))

    def test_vector(self):
        self.h.B_angle_vector.shape == (2, 3, 8)

    def test_superop(self):
        self.h.B_angle_superop.shape == (2, 3, 8, 8)


class TestZeemanCoupling:
    def setup(self):
        s = mt.Spinoperator(1/2)
        g = mt.Tensor(np.array([1, 2, 3]))
        theta = np.linspace(0, 0.9, 5)*np.pi
        phi = np.linspace(0.1, 1.5, 5)*np.pi
        g.multirotation(phi, theta)
        B = np.linspace(1, 4, 4)
        grid_points = 5
        self.h = mut.Multioperator(s, grid_points, B)
        self.h.zeeman_coupling(g)

        self.comp = mut.Multioperator(s, grid_points, B)
        self.comp.create_linear_operator(g)

    def test_value(self):
        np.testing.assert_allclose(
            self.h.B_angle_matrix[0], self.comp.angle_matrix*1)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[1], self.comp.angle_matrix*2)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[2], self.comp.angle_matrix*3)
        np.testing.assert_allclose(
            self.h.B_angle_matrix[3], self.comp.angle_matrix*4)

    def test_dtype(self):
        assert self.h.B_angle_matrix.dtype == "complex64"

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h.B_angle_matrix,
            np.conj(np.transpose(self.h.B_angle_matrix, (0, 1, 3, 2))))

    def test_vector(self):
        self.h.B_angle_vector.shape == (4, 5, 4)

    def test_superop(self):
        self.h.B_angle_superop.shape == (4, 5, 4, 4)


class TestMicrowaveCoupling:
    def setup(self):
        self.s = mt.Spinoperator(1/2)
        B = np.linspace(1, 5, 4)
        grid_points = 3
        self.omega_nut = 2
        self.omega_mw = -2
        self.h = mut.Multioperator(self.s, grid_points, B)
        self.h.microwave_coupling(self.omega_nut, self.omega_mw)

    def test_value(self):
        comp = np.ones((2, 2))
        comp[1, 1] *= -1
        comp = np.array([comp, comp, comp])
        comp = np.array([comp, comp, comp, comp])
        np.testing.assert_array_equal(comp, self.h.B_angle_matrix)

    def test_dtype(self):
        assert self.h.B_angle_matrix.dtype == "complex64"

    def test_hermitean(self):
        np.testing.assert_array_equal(
            self.h.B_angle_matrix,
            np.conj(np.transpose(self.h.B_angle_matrix, (0, 1, 3, 2))))

    def test_vector(self):
        self.h.B_angle_vector.shape == (4, 3, 4)

    def test_superop(self):
        self.h.B_angle_superop.shape == (4, 3, 4, 4)
