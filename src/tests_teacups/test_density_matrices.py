import scipy.constants as const
import teacups.grid as grid
import teacups.matrix_tools as mt
import teacups.hamiltonians as ham
import teacups.creators as cr
import teacups.density_matrices as dm
import numpy as np
import pytest
import sys
sys.path.append("./..")


MU_B = const.physical_constants['Bohr magneton in Hz/T'][0]


class Sys:
    def __init__(self):
        return


class Exp:
    def __init__(self):
        return


class Opt:
    def __init__(self):
        return


class Cal:
    def __init__(self):
        return


def initialize_classes(self):
    self.sys = Sys()
    self.opt = Opt()
    self.cal = Cal()
    self.exp = Exp()
    return


class TestErrorMessages:
    def setup(self):
        initialize_classes(self)

    def test_zf_error(self):
        self.sys.precursor = "zf"
        self.sys.spin_system = "bla"
        with pytest.raises(AttributeError):
            dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

    def test_eigen_error(self):
        self.sys.precursor = "eigen"
        self.sys.spin_system = "bla"
        with pytest.raises(AttributeError):
            dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

    def test_singlet_error(self):
        self.sys.precursor = "singlet"
        self.sys.spin_system = "bla"
        with pytest.raises(AttributeError):
            dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

    def test_triplet_zf_error(self):
        self.sys.precursor = "triplet-zf"
        self.sys.spin_system = "bla"
        with pytest.raises(AttributeError):
            dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

    def test_precursor_error(self):
        self.sys.precursor = "bla"
        with pytest.raises(AttributeError):
            dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)


class TestDoublets:
    def setup(self):
        initialize_classes(self)
        self.opt.CUPY = False
        self.opt.space = 'hilbert'

        self.cal.theta, self.cal.phi = grid.fibonacci_grid(1)
        self.opt.grid_points = len(self.cal.theta)
        self.exp.B_z = np.array([1/(2*MU_B), 2/(2*MU_B), 3/(2*MU_B)])

        self.sys.spin_system = "doub"
        self.sys.s = 1/2
        cr.set_up_spinoperator(self.sys, self.cal)

        self.sys.g = [1.9, 2., 2.1]
        self.sys.population = [1, 2]

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham_sys = ham.set_up_doublet_hamiltonian(self.exp, self.opt,
                                                          self.cal)

        pop = np.diag(np.arange(1, 3))
        self.rho_basis = np.array([[pop], [pop], [pop]])

    def test_eigen_precursor(self):
        self.sys.precursor = "eigen"
        _, vec = np.linalg.eigh(self.cal.ham_sys)
        rho_eigen = vec@self.rho_basis@np.linalg.inv(vec)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        np.testing.assert_allclose(rho_eigen, self.cal.rho)
        assert self.cal.rho.dtype == "complex64"

    def test_liouville_space_shape(self):
        self.sys.precursor = "eigen"
        self.opt.space = "liouville"
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        assert (self.cal.rho.shape == (3, 1, 4))


class TestTriplets:
    def setup(self):
        initialize_classes(self)
        self.opt.CUPY = False
        self.opt.space = 'hilbert'

        self.cal.theta, self.cal.phi = grid.fibonacci_grid(1)
        self.opt.grid_points = len(self.cal.theta)
        self.exp.B_z = np.array([1/(2*MU_B), 2/(2*MU_B), 3/(2*MU_B)])

        self.sys.spin_system = "trip"
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, self.cal)

        self.sys.g_tri = [1.9, 2., 2.1]
        self.sys.D_tri = 1000
        self.sys.E_tri = -500
        self.sys.population = [1, 2, 3]

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham_sys = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                          self.cal)
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)

        pop = np.diag(np.arange(1, 4))
        self.rho_basis = np.array([[pop], [pop], [pop]])

    def test_eigen_precursor(self):
        self.sys.precursor = "eigen"
        _, vec = np.linalg.eigh(self.cal.ham_sys)
        rho_eigen = vec@self.rho_basis@np.linalg.inv(vec)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        np.testing.assert_allclose(rho_eigen, self.cal.rho)
        assert self.cal.rho.dtype == "complex64"

    def test_zf_precursor(self):
        self.sys.precursor = "zf"
        _, vec = np.linalg.eigh(self.cal.ham_tri_hf)
        rho_zf = np.linalg.inv(vec)@self.rho_basis@vec
        rho_zf *= np.eye(3)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        np.testing.assert_allclose(rho_zf, self.cal.rho)
        assert self.cal.rho.dtype == "complex64"


class TestRps:
    def setup(self):
        initialize_classes(self)
        self.opt.CUPY = False
        self.opt.space = 'hilbert'

        self.cal.theta, self.cal.phi = grid.fibonacci_grid(1)
        self.opt.grid_points = len(self.cal.theta)
        self.exp.B_z = np.array([1/(2*MU_B), 2/(2*MU_B), 3/(2*MU_B)])

        self.sys.spin_system = "rp"
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, self.cal)

        self.sys.g1 = [1.98, 2., 2.01]
        self.sys.g2 = [2.0, 2.0, 1.999]
        self.sys.J_ex = -4
        self.sys.D = -60
        self.sys.E = -5
        self.sys.g_tri = [2, 2, 2]
        self.sys.D_tri = 1000
        self.sys.E_tri = -500
        self.sys.population = [1, 2, 3, 4]

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham_sys = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                     self.cal)
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)

        pop = np.diag(np.arange(1, 5))
        self.rho_basis = np.array([[pop], [pop], [pop]])

    def test_eigen_precursor(self):
        self.sys.precursor = "eigen"
        _, vec = np.linalg.eigh(self.cal.ham_sys)
        rho_eigen = vec@self.rho_basis@np.linalg.inv(vec)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        np.testing.assert_allclose(rho_eigen, self.cal.rho)
        assert self.cal.rho.dtype == "complex64"

    def test_singlet_precursor(self):
        self.sys.precursor = 'singlet'
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

        r = np.zeros((4, 4))
        r[1, 1] = 1
        r = np.array([[r], [r], [r]])

        assert np.array_equal(r, self.cal.rho)
        assert self.cal.rho.dtype == "complex64"

    def test_triplet_zf_precursor(self):
        self.sys.population = np.arange(1, 4, dtype=np.complex64)
        self.sys.precursor = 'triplet-zf'
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

        pop = np.diag(self.sys.population)
        self.rho_trip_basis = np.array([[pop], [pop], [pop]])

        _, vec = np.linalg.eigh(self.cal.ham_tri_hf)
        rho_trip_zf = np.linalg.inv(vec)@self.rho_trip_basis@vec
        rho_trip_zf *= np.eye(3, dtype=np.float32)

        comp = np.zeros((3, 1, 4, 4), dtype=np.complex64)
        comp[:, :, 0, 0] = rho_trip_zf[:, :, 0, 0]
        comp[:, :, 2, 2] = rho_trip_zf[:, :, 1, 1]
        comp[:, :, 3, 3] = rho_trip_zf[:, :, 2, 2]

        np.testing.assert_allclose(comp, self.cal.rho, atol=2e-6)
        assert self.cal.rho.dtype == "complex64"


class TestTdps:
    def setup(self):
        initialize_classes(self)
        self.opt.CUPY = False
        self.opt.space = 'hilbert'

        self.cal.theta, self.cal.phi = grid.fibonacci_grid(1)
        self.opt.grid_points = len(self.cal.theta)
        self.exp.B_z = np.array([1/(2*MU_B), 2/(2*MU_B), 3/(2*MU_B)])

        self.sys.spin_system = "tdp"
        self.sys.s = [1/2, 1]
        cr.set_up_spinoperator(self.sys, self.cal)

        self.sys.g = [2.0, 2.0, 1.999]
        self.sys.J_ex = 20000
        self.sys.D = -60
        self.sys.E = -5
        self.sys.g_tri = [2, 2, 2]
        self.sys.D_tri = 1000
        self.sys.E_tri = -500
        self.sys.population = [1, 2, 3, 4, 5, 6]

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham_sys = ham.set_up_tdp_hamiltonian(self.sys, self.exp,
                                                      self.opt, self.cal)
        self.cal.ham_hf = ham.set_up_tdp_full_high_field_hamiltonian(
            self.sys, self.exp, self.opt, self.cal)
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)

        pop = np.diag(np.arange(1, 7, dtype=np.complex64))
        self.rho_basis = np.array([[pop], [pop], [pop]])

    def test_eigen_precursor(self):
        self.sys.precursor = "eigen"
        _, vec = np.linalg.eigh(self.cal.ham_sys)
        rho_eigen = vec@self.rho_basis@np.linalg.inv(vec)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        # Toleranz nötig wegen mumerischen Unterschieds zwischen inverser und adjungierter Matrix
        np.testing.assert_allclose(rho_eigen, self.cal.rho, rtol=1e-6)
        assert self.cal.rho.dtype == "complex64"

    def test_triplet_zf_precursor(self):
        self.sys.precursor = 'triplet-zf'
        self.sys.population = np.array([0.5, 0.51, 1, 2, 3])

        pop = np.diag(np.arange(1, 4, dtype=np.complex64))
        rho_basis = np.array([[pop], [pop], [pop]])
        rho_basis = np.kron(rho_basis, np.eye(2, dtype=np.float32))

        _, vec_xyz = np.linalg.eigh(self.cal.ham_hf)
        _, vec_sys = np.linalg.eigh(self.cal.ham_sys)

        rho = np.linalg.inv(vec_xyz)@rho_basis@vec_xyz
        rho *= np.eye(6, dtype=np.float32)
        rho = vec_sys@rho@np.linalg.inv(vec_sys)
        rho *= np.eye(6, dtype=np.float32)

        rho_doub = np.diag(np.array([0.5, 0.51], dtype=np.float32))
        rho_doub = np.kron(rho_doub, np.eye(3, dtype=np.float32))
        rho += rho_doub

        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        # Toleranz nötig wegen mumerischen Unterschieds zwischen inverser und adjungierter Matrix
        np.testing.assert_allclose(rho, self.cal.rho, rtol=1e-6)
        assert self.cal.rho.dtype == "complex64"

    def test_triplet_zf_precursor_changing_with_J(self):
        self.sys.precursor = 'triplet-zf'
        self.sys.population = np.array([0.5, 0.51, 1, 2, 3])

        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        rho_J_20000 = self.cal.rho

        self.sys.J_ex = 20
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        rho_J_20 = self.cal.rho

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(rho_J_20, rho_J_20000)


class TestTripletZfDensityMatrixConditions:
    def setup(self):
        self.opt = Opt()
        self.sys = Sys()
        self.exp = Exp()
        self.cal = Cal()

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)
        self.opt.space = 'hilbert'
        self.exp.B_z = np.linspace(1, 3, 3)
        self.exp.B_mw = 7
        self.exp.freq_mw = 3
        self.cal.s = mt.Spinoperator(1)

        self.sys.g_tri = [1, 2, 3]
        self.sys.D_tri = 5
        self.sys.E_tri = -2
        self.sys.population = [0.1, 0.3, 0.6]
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'zf'

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

    def test_trace(self):
        ones = np.ones((3, 3))
        trace = self.cal.rho.trace(axis1=-2, axis2=-1)
        np.testing.assert_allclose(ones, trace, atol=1e-6)

    def test_offdiagonals(self):
        rho_zeroed = self.cal.rho
        for b in range(3):
            for a in range(3):
                np.fill_diagonal(rho_zeroed[a, b], 0)
        zeros = np.zeros((3, 3, 3, 3))
        np.testing.assert_allclose(zeros, rho_zeroed, atol=1e-8, rtol=1e-5)

    def test_nonequal_after_rotation(self):
        if np.allclose(self.cal.rho[0, 0], self.cal.rho[1, 1]):
            assert False
        else:
            assert True
