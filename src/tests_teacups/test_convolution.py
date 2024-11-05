import teacups.convolution as conv
import pytest as pt
import numpy as np
import sys
sys.path.append("./..")


class TestVoigtConvolution:
    def setup(self):
        self.field = np.arange(0, 200)
        self.spec = np.zeros((100, 200), dtype=np.complex64)
        self.spec[:, 50:150] = 1
        self.C = conv.voigt_convolution(50, self.spec)

    def test_shape(self):
        assert self.C.shape == (100, 200)

    def test_type(self):
        assert type(self.C) == np.ndarray

    def test_dtype(self):
        assert self.C.dtype == 'complex64'

    def test_deltafunction(self):
        D = conv.voigt_convolution(0, self.spec)
        assert np.array_equal(D, self.spec)

    def test_maximum(self):
        maximum = self.C.argmax()
        assert maximum == len(self.field)//2-1
