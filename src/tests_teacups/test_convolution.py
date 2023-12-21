import sys
sys.path.append("./..")

import numpy as np
import pytest as pt
import teacups.convolution as conv


class Test_voigt_convolution:
    field = np.arange(0, 200)
    spec = np.zeros((100, 200))
    spec[:, 50:150] = 1
    C = conv.voigt_convolution(50, spec)

    def test_shape(self):
        assert self.C.shape == (100, 200)

    def test_type(self):
        assert type(self.C) == np.ndarray

    def test_deltafunction(self):
        D = conv.voigt_convolution(0, self.spec)
        assert np.array_equal(D, self.spec)

    def test_maximum(self):
        maximum = self.C.argmax()
        assert maximum == len(self.field)//2-1


class Test_generalized_pascal:
    row_4_12 = conv.generalized_pascal(4, 1/2)
    row_3_11 = conv.generalized_pascal(3, 1)
    row_0_12 = conv.generalized_pascal(0, 1/2)
    row_1_00 = conv.generalized_pascal(1, 0)

    def test_shape(self):
        assert self.row_4_12.shape == (5,)
        assert self.row_3_11.shape == (7,)

    def test_type(self):
        assert type(self.row_4_12) == np.ndarray

    def test_values(self):
        assert np.array_equal(self.row_4_12, np.array([1, 4, 6, 4, 1]))
        assert np.array_equal(self.row_3_11, np.array([1, 3, 6, 7, 6, 3, 1]))

    def test_no_nucleus(self):
        assert self.row_0_12 == np.array([1.])

    def test_no_spin(self):
        assert self.row_1_00 == np.array([1.])

    def test_value_errors(self):
        with pt.raises(ValueError):
            conv.generalized_pascal(1.4, 2)
            conv.generalized_pascal(1, 2.1)
