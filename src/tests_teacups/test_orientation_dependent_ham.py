import sys
sys.path.append("./..")

import teacups.orientation_dependent_ham as odh
import numpy as np


class TestTensorRotation:
    def setup(self):
        self.tensor = np.array(([1., 2., 3.], [0, 0, 0], [0, 0, 0]),
                               dtype=np.float32)
        self.unit = np.eye(3, dtype=np.complex64)
        self.rotated_tensor = odh.tensor_rotation(self.tensor, 1, 2)
        self.rotated_unit = odh.tensor_rotation(self.unit, 1, 2)

    def test_shape(self):
        assert self.rotated_tensor.shape == (3, 3)

    def test_type(self):
        assert type(self.rotated_tensor) == np.ndarray

    def test_dtype(self):
        assert self.rotated_unit.dtype == "complex64"
        assert self.rotated_tensor.dtype == "float32"

    def test_eye(self):
        np.testing.assert_allclose(self.rotated_unit, self.unit, atol=1e-6)

    def test_values(self):
        num_rot_tensor = np.array([[0.821379, -0.05376802, -0.17383894],
                                   [3.07396785, -0.20122401, -0.65058311],
                                   [-1.79474586,  0.11748527,  0.37984501]])
        np.testing.assert_allclose(self.rotated_tensor,
                                   num_rot_tensor, atol=1e-6)

    def test_trace(self):
        assert np.trace(self.rotated_tensor) == np.trace(self.tensor)

    def test_vector_product(self):
        vec1 = np.arange(1, 4, dtype=np.float64)
        vec2 = np.arange(4, 7, dtype = np.float64)
        prod = vec1@vec2
        comp = odh.tensor_rotation(vec1, 2, 3)@odh.tensor_rotation(vec2, 2, 3)
        np.testing.assert_allclose(prod, comp)


class TestCreateLinearHamiltonian:
    def setup(self):
        a = np.arange(1, 17).reshape(4, 4)
        b = np.arange(17, 33).reshape(4, 4)
        c = np.arange(33, 49).reshape(4, 4)
        A = np.array([a, b, c], dtype=np.complex64)
        T = np.arange(1, 10).reshape(3, 3)

        self.ham_z = odh.create_linear_hamiltonian(T, A)
        self.ham_x = odh.create_linear_hamiltonian(T, A, z=False)

    def test_value_z(self):
        result = np.array([[402, 420, 438, 456],
                           [474, 492, 510, 528],
                           [546, 564, 582, 600],
                           [618, 636, 654, 672]])
        np.testing.assert_array_equal(self.ham_z, result)

    def test_value_x(self):
        result = np.array([[300, 312, 324, 336],
                           [348, 360, 372, 384],
                           [396, 408, 420, 432],
                           [444, 456, 468, 480]])
        np.testing.assert_array_equal(self.ham_x, result)

    def test_type(self):
        assert type(self.ham_z) == np.ndarray
        assert type(self.ham_x) == np.ndarray

    def test_shape(self):
        assert self.ham_z.shape == (4, 4)
        assert self.ham_x.shape == (4, 4)

    def test_dtype(self):
        assert self.ham_z.dtype == "complex64"
        assert self.ham_x.dtype == "complex64"


class TestCreateBilinearHamiltonian:
    def setup(self):
        a = np.arange(1, 17).reshape(4, 4)
        b = np.arange(17, 33).reshape(4, 4)
        c = np.arange(33, 49).reshape(4, 4)
        A = np.array([a, b, c], dtype=np.complex64)
        B = np.array([b, c, a], dtype=np.complex64)
        T = np.arange(1, 10, dtype=np.complex64).reshape(3, 3)
        self.ham = odh.create_bilinear_hamiltonian(A, T, B)

    def test_value(self):
        result = np.array([[100434, 104916, 109398, 113880],
                           [116226, 121428, 126630, 131832],
                           [132018, 137940, 143862, 149784],
                           [147810, 154452, 161094, 167736]])
        np.testing.assert_array_equal(self.ham, result)

    def test_shape(self):
        assert self.ham.shape == (4, 4)

    def test_type(self):
        assert type(self.ham) == np.ndarray

    def test_dtype(self):
        assert self.ham.dtype == "complex64"
