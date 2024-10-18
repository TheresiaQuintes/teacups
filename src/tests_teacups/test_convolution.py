import teacups.convolution as conv
import pytest as pt
import numpy as np
import sys
sys.path.append("./..")


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
