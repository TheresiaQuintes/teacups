import teacups.memory as mem
import psutil


class Sys:
    def __init__(self):
        return


class TestChunkSize:
    def setup_method(self):
        self.bottleneck = None
        self.bp = 5
        self.gp = 3
        self.available_memory = psutil.virtual_memory().available

    def test_no_bottleneck(self):
        cs = mem.chunk_size(self.bottleneck, self.bp, self.gp)
        assert cs == 1


class TestDefineBottleneck:
    def setup_method(self):
        self.sys = Sys()

    def test_none(self):
        self.sys.spin_system = "bla"
        self.sys.precursor = "blub"
        bn = mem.define_bottleneck(self.sys)
        assert bn is None

    def test_tdp_zf(self):
        self.sys.spin_system = "tdp"
        self.sys.precursor = "triplet-zf"
        bn = mem.define_bottleneck(self.sys)
        assert bn == "set_up_density_matrix"


class TestNomTdpZfSetUpDensityMatrix:
    def setup_method(self):
        self.bp = 5
        self.gp = 3
        self.nom = mem.nom_tdp_zf_set_up_density_matrix(self.bp, self.gp)

    def test_value(self):
        assert self.nom == 60480 * 15
