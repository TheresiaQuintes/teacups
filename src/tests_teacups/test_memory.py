import numpy as np
import teacups.memory as mem
import psutil

class Sys:
    def __init__(self):
        return


class TestChunkSize:
    def setup(self):
        self.bottleneck = None
        self.bp = 5
        self.gp = 3
        self.available_memory = psutil.virtual_memory().available

    def test_no_bottleneck(self):
        cs = mem.chunk_size(self.bottleneck, self.bp, self.gp)
        assert cs == 1

    def test_set_up_density_matrix(self):
        self.bottleneck = "set_up_density_matrix"
        self.bottleneck = "set_up_density_matrix"
        bp = np.sqrt(self.available_memory/(0.8*14*36*8))
        bp = int(bp)+1
        gp = int(bp)+2
        cs_big = mem.chunk_size(self.bottleneck, bp, gp)
        cs_small = mem.chunk_size(self.bottleneck, self.bp, self.gp)
        assert cs_big == 3
        assert cs_small == 1


class TestDefineBottleneck:
    def setup(self):
        self.sys = Sys()

    def test_none(self):
        self.sys.spin_system = "bla"
        self.sys.precursor = "blub"
        bn = mem.define_bottleneck(self.sys)
        assert bn == None

    def test_tdp_zf(self):
        self.sys.spin_system = "tdp"
        self.sys.precursor = "triplet-zf"
        bn = mem.define_bottleneck(self.sys)
        assert bn == "set_up_density_matrix"


class TestNomTdpZfSetUpDensityMatrix:
    def setup(self):
        self.bp = 5
        self.gp = 3
        self.nom = mem.nom_tdp_zf_set_up_density_matrix(self.bp, self.gp)

    def test_value(self):
        assert self.nom == 60480
