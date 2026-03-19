import teacups.convolution as conv
import numpy as np
import sys
sys.path.append("./..")


class TestVoigtConvolutionMagnetic:
    def setup(self):
        self.field = np.arange(0, 200)
        self.spec = np.zeros((100, 200), dtype=np.complex64)
        self.spec[:, 50:150] = 1
        self.C = conv.voigt_convolution(0, 50, self.spec)

    def test_shape(self):
        assert self.C.shape == (100, 200)

    def test_type(self):
        assert type(self.C) == np.ndarray

    def test_dtype(self):
        assert self.C.dtype == 'complex64'

    def test_deltafunction(self):
        D = conv.voigt_convolution(0, 0, self.spec)
        assert np.array_equal(D, self.spec)

    def test_maximum(self):
        maximum = self.C.argmax()
        assert maximum == len(self.field)//2-1


class TestVoigtConvolutionTime:
    def setup(self):
        self.t = np.linspace(0, 10, 100)
        signal = (np.sin(2*np.pi*1*self.t) +
                  0.5*np.sin(2*np.pi*3*self.t))*np.exp(-self.t/5)
        self.spec = np.stack((signal, signal), axis=1)
        self.spec = self.spec.astype(np.complex64)
        self.C = conv.voigt_convolution(1, 0, self.spec)
        self.C_ext = conv.voigt_convolution(1, 0, self.spec, extend_t=True)

    def test_shape(self):
        assert self.C.shape == (100, 2)
        assert self.C_ext.shape == (105, 2)

    def test_type(self):
        assert type(self.C) == np.ndarray

    def test_dtype(self):
        assert self.C.dtype == 'complex64'

    def test_deltafunction(self):
        D = conv.voigt_convolution(0, 0, self.spec)
        assert np.array_equal(D, self.spec)

    def test_first_value_extended_t(self):
        assert self.C_ext[0, 0] == 0

    def test_convolution(self):
        sig = np.sin(2*np.pi*1*self.t) * np.exp(-self.t/5)
        sig /= max(abs(sig))
        c = self.C[:, 0]
        c /= max(abs(c))
        assert np.allclose(c[20:], sig[20:], atol=0.1)
