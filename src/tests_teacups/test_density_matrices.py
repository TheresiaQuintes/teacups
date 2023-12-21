import sys
sys.path.append("./..")

import numpy as np
import teacups.density_matrices as dm
import teacups.creators as cr
import teacups.hamiltonians as ham
import teacups.matrix_tools as mt
import teacups.grid as grid
import tests_teacups.set_up_comparison_arrays as comp
import scipy.constants as const
from copy import deepcopy

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


class Test_set_up_density_matrix:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.exp = Exp()

        self.sys.D = 1
        self.sys.E = 0
        self.sys.J_ex = 1
        self.sys.D_tri = 1
        self.sys.E_tri = 0
        self.sys.g1 = [2, 2, 2]
        self.sys.g2 = [2, 2, 2]
        self.sys.g_tri = [2, 2, 2]
        self.sys.g = [2, 2, 2]

        self.exp.B_z = np.array([1/(2*MU_B), 2/(2*MU_B), 3/(2*MU_B)])
        self.opt.grid_points = 1
        self.cal.theta, self.cal.phi = grid.get_theta_phi(1)
        self.opt.space = 'hilbert'

        cr.set_up_tensors(self.sys, self.cal)

        rho_doub = np.diag(np.arange(1, 3))
        self.rho_doub = np.array([[rho_doub], [rho_doub], [rho_doub]])
        rho_trip = np.diag(np.arange(1, 4))
        self.rho_trip = np.array([[rho_trip], [rho_trip], [rho_trip]])
        rho_rp = np.diag(np.arange(1, 5))
        self.rho_rp = np.array([[rho_rp], [rho_rp], [rho_rp]])
        rho_tdp = np.diag(np.arange(1, 7))
        self.rho_tdp = np.array([[rho_tdp], [rho_tdp], [rho_tdp]])

    def test_basis_doub(self):
        cal = Cal()
        self.sys.spin_system = 'doub'
        self.sys.precursor = 'basis'
        self.sys.population = np.arange(1, 3)
        self.sys.s = 1/2
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        assert np.array_equal(self.rho_doub, cal.rho)

    def test_basis_trip(self):
        cal = Cal()
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'basis'
        self.sys.population = np.arange(1, 4)
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        assert np.array_equal(self.rho_trip, cal.rho)

    def test_basis_rp(self):
        cal = Cal()
        self.sys.spin_system = 'rp'
        self.sys.precursor = 'basis'
        self.sys.population = np.arange(1, 5)
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        assert np.array_equal(self.rho_rp, cal.rho)

    def test_basis_tdp(self):
        cal = Cal()
        self.sys.spin_system = 'tdp'
        self.sys.precursor = 'basis'
        self.sys.population = np.arange(1, 7)
        self.sys.s = [1/2, 1]
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        assert np.array_equal(self.rho_tdp, cal.rho)

    def test_eigen_doublet(self):
        cal = Cal()
        self.sys.spin_system = 'doub'
        self.sys.precursor = 'eigen'
        self.sys.population = np.arange(1, 3)
        self.sys.s = 1/2
        cr.set_up_spinoperator(self.sys, self.cal)
        cal.s = self.cal.s
        cal.ham_sys = ham.set_up_doublet_hamiltonian(self.exp, self.opt,
                                                     self.cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)
        eig, vec = np.linalg.eigh(cal.ham_sys)
        comp = vec @ self.rho_doub @ np.linalg.inv(vec)
        comp *= np.eye(2)

        np.testing.assert_allclose(cal.rho, comp)

    def test_eigen_triplet(self):
        cal = Cal()
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'eigen'
        self.sys.population = np.arange(1, 4)
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, self.cal)
        cal.s = self.cal.s
        cal.ham_sys = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                     self.cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        eig, vec = np.linalg.eigh(cal.ham_sys)
        comp = vec @ self.rho_trip @ np.linalg.inv(vec)
        comp *= np.eye(3)

        np.testing.assert_allclose(cal.rho, comp)

    def test_eigen_rp(self):
        cal = Cal()
        self.sys.spin_system = 'rp'
        self.sys.precursor = 'eigen'
        self.sys.population = np.arange(1, 5)
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, self.cal)
        cal.s = self.cal.s
        cal.ham_sys = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                self.cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        eig, vec = np.linalg.eigh(cal.ham_sys)
        comp = vec @ self.rho_rp @ np.linalg.inv(vec)
        comp *= np.eye(4)

        np.testing.assert_allclose(cal.rho, comp)

    def test_eigen_tdp(self):
        cal = Cal()
        self.sys.spin_system = 'tdp'
        self.sys.precursor = 'eigen'
        self.sys.population = np.arange(1, 7)
        self.sys.s = [1/2, 1]
        cr.set_up_spinoperator(self.sys, self.cal)
        cal.s = self.cal.s
        cal.ham_sys = ham.set_up_tdp_hamiltonian(self.sys, self.exp, self.opt,
                                                 self.cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        eig, vec = np.linalg.eigh(cal.ham_sys)
        comp = vec @ self.rho_tdp @ np.linalg.inv(vec)

        np.testing.assert_allclose(cal.rho, comp, atol=1e-8, rtol=1e-5)

    def test_singlet_rp(self):
        cal = Cal()
        self.sys.spin_system = 'rp'
        self.sys.precursor = 'singlet'
        self.sys.population = np.arange(1, 5)
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        r = np.zeros((4, 4))
        r[1, 1] = 1
        r = np.array([[r], [r], [r]])

        assert np.array_equal(r, cal.rho)

    def test_coupled_tdp(self):
        cal = Cal()
        self.sys.spin_system = 'tdp'
        self.sys.precursor = 'coupled'
        self.sys.population = np.arange(1, 7)
        self.sys.s = [1/2, 1]
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        s13 = np.sqrt(1/3)
        s23 = np.sqrt(2/3)
        trans = np.array([[1, 0, 0, 0, 0, 0],
                          [0, s23, 0, s13, 0, 0],
                          [0, 0, s13, 0, s23, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, -s13, 0, s23, 0, 0],
                          [0, 0, s23, 0, -s13, 0]])
        trans = np.array([[trans], [trans], [trans]])

        comp = np.linalg.inv(trans) @ self.rho_tdp @ trans
        comp *= np.eye(6)

        np.testing.assert_allclose(comp, cal.rho, atol=1e-8, rtol=1e-5)

    def test_zf_triplet(self):
        cal = Cal()
        cal.theta, cal.phi = grid.get_theta_phi(1)
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'zf'
        self.sys.population = np.arange(1, 4)
        self.sys.population = np.array(self.sys.population, dtype=np.float32)
        self.sys.rho_0_tri = self.sys.population
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, cal)

        ham_hf = ham.set_up_triplet_high_field_hamiltonian(self.exp, self.opt,
                                                           self.cal)
        cal.ham_tri_hf = ham_hf

        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        eig_hf, vec_hf = np.linalg.eigh(ham_hf)

        rho_trip = cr.create_tensor(self.sys.rho_0_tri, cal.phi, cal.theta)
        comp = np.linalg.inv(vec_hf) @ rho_trip.multirot @ vec_hf
        comp *= np.eye(3, dtype=np.float32)

        np.testing.assert_allclose(cal.rho, comp, atol=1e-8, rtol=1e-5)

    def test_zf_triplet_rp(self):
        cal_trip = Cal()
        cal_trip.theta, cal_trip.phi = grid.get_theta_phi(1)
        cal_trip.g_tri_tensor = self.cal.g_tri_tensor
        cal_trip.D_tri_tensor = self.cal.D_tri_tensor
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'zf'
        self.sys.population = np.arange(1, 4)
        self.sys.rho_0_tri = np.array(self.sys.population, dtype=np.float32)
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, cal_trip)

        cal_trip.ham_tri_zf = ham.set_up_triplet_zero_field_hamiltonian(
            self.exp, self.opt, cal_trip)
        cal_trip.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, cal_trip)

        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal_trip)

        comp = np.zeros((3, 1, 4, 4), dtype=np.complex64)
        comp[:, :, 0, 0] = cal_trip.rho[:, :, 0, 0]
        comp[:, :, 2, 2] = cal_trip.rho[:, :, 1, 1]
        comp[:, :, 3, 3] = cal_trip.rho[:, :, 2, 2]

        cal = self.cal
        self.sys.spin_system = 'rp'
        self.sys.precursor = 'triplet-zf'
        self.sys.population = np.arange(1, 4, dtype=np.float32)
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, cal)
        cal.ham_tri_hf = cal_trip.ham_tri_hf
        cal.ham_tri_zf = cal_trip.ham_tri_zf
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        np.testing.assert_allclose(comp, cal.rho)

    def test_eigen_triplet_rp(self):
        cal_trip = deepcopy(self.cal)
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'eigen'
        self.sys.population = np.arange(1, 4)
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, cal_trip)
        cal_trip.ham_sys = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                          cal_trip)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal_trip)

        comp = np.zeros((3, 1, 4, 4))
        comp[:, :, 0, 0] = cal_trip.rho[:, :, 0, 0]
        comp[:, :, 2, 2] = cal_trip.rho[:, :, 1, 1]
        comp[:, :, 3, 3] = cal_trip.rho[:, :, 2, 2]

        cal = self.cal
        self.sys.spin_system = 'rp'
        self.sys.precursor = 'triplet-eigen'
        self.sys.population = np.arange(1, 4)
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        np.testing.assert_allclose(comp, cal.rho)

    def test_uncoupled_zf_tdp(self):
        cal_trip = deepcopy(self.cal)
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'zf'
        self.sys.population = np.arange(1, 4)
        self.sys.rho_0_tri = self.sys.population
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, cal_trip)

        cal_trip.ham_tri_zf = ham.set_up_triplet_zero_field_hamiltonian(
            self.exp, self.opt, cal_trip)
        cal_trip.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, cal_trip)

        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal_trip)

        comp = np.kron(np.diag(np.arange(1, 3)), cal_trip.rho)

        cal = self.cal
        self.sys.spin_system = 'tdp'
        self.sys.precursor = 'triplet-zf'
        self.sys.population = np.array([1, 2, 1, 2, 3])
        self.sys.s = [1/2, 1]
        cr.set_up_spinoperator(self.sys, cal)
        cal.ham_tri_hf = cal_trip.ham_tri_hf
        cal.ham_tri_zf = cal_trip.ham_tri_zf
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        np.testing.assert_allclose(comp, cal.rho)

    def test_uncpupled_eigen_tdp(self):
        cal_trip = deepcopy(self.cal)
        self.sys.spin_system = 'trip'
        self.sys.precursor = 'eigen'
        self.sys.population = np.arange(1, 4)
        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, cal_trip)
        cal_trip.ham_sys = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                          cal_trip)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal_trip)

        comp = np.kron(np.diag(np.arange(1, 3)), cal_trip.rho)

        cal = self.cal
        self.sys.spin_system = 'tdp'
        self.sys.precursor = 'triplet-eigen'
        self.sys.population = np.array([1, 2, 1, 2, 3])
        self.sys.s = [1/2, 1]
        cr.set_up_spinoperator(self.sys, cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, cal)

        np.testing.assert_allclose(comp, cal.rho)


class Test_set_up_rp_density_matrix:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.exp = Exp()

        self.sys.s = [1/2, 1/2]
        self.sys.spin_system = 'rp'
        self.sys.population = [0, 0.33, 0.67]
        self.sys.D_tri = -700
        self.sys.E_tri = -200
        self.sys.g_tri = [2, 2, 2]
        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.get_theta_phi(3)
        self.opt.space = 'hilbert'

        self.exp.B_z = np.linspace(342, 348, 4)
        cr.set_up_spinoperator(self.sys, self.cal)
        self.cal.g1_tensor = cr.create_tensor(
            [1, 2, 3], self.cal.phi, self.cal.theta)

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)
        self.cal.ham_tri_zf = ham.set_up_triplet_zero_field_hamiltonian(
            self.exp, self.opt, self.cal)
        self.cal.s_tri = mt.Spinoperator(1)

    def test_triplet_rho_liouville(self):
        self.opt.space = 'liouville'
        self.sys.precursor = 'triplet-zf'
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        rho = comp.triplet_rho_liouville
        np.testing.assert_allclose(self.cal.rho, rho)

    def test_singlet_rho_hilbert(self):
        self.sys.precursor = 'singlet'
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        rho = comp.singlet_rho_hilbert
        np.testing.assert_allclose(self.cal.rho, rho)


class Test_set_up_triplet_density_matrix:
    def setup(self):
        self.opt = Opt()
        self.sys = Sys()
        self.exp = Exp()
        self.cal = Cal()

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.get_theta_phi(3)
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
        self.cal.ham_tri_zf = ham.set_up_triplet_zero_field_hamiltonian(
            self.exp, self.opt, self.cal)
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)

    def test_trace(self):
        ones = np.ones((3, 3))
        trace = self.cal.rho.trace(axis1=-2, axis2=-1)
        np.testing.assert_allclose(ones, trace)

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


class Test_set_up_doublet_density_matrix:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()
        self.cal = Cal()

        self.sys.s = 1/2
        self.sys.g = [1, 2, 3]
        self.exp.B_z = np.linspace(1, 3, 3)
        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.get_theta_phi(3)
        self.opt.space = 'hilbert'
        self.cal.s_doub = mt.Spinoperator(1/2)
        self.sys.population = [0, 1]
        self.sys.precursor = 'basis'
        self.sys.spin_system = 'doub'
        cr.set_up_tensors(self.sys, self.cal)
        cr.set_up_spinoperator(self.sys, self.cal)

    def test_elements_in_multimatrix(self):
        comparison = np.zeros((2, 2))
        comparison[1, 1] = 1
        dm.set_up_density_matrix(self.sys, self.exp, self.opt, self.cal)
        for b in range(0, 3):
            for a in range(0, 3):
                assert np.array_equal(comparison, self.cal.rho[b, a])
