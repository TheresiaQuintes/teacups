import numpy.linalg as la
import scipy.constants as const
import tests_teacups.set_up_comparison_arrays as comp
import teacups.grid as grid
import teacups.input_handler as inputs
import teacups.creators as cr
import teacups.matrix_tools as mt
import teacups.hamiltonians as ham
import numpy as np
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


class TestMwHamiltonian:
    def setup(self):
        initialize_classes(self)

        self.opt.grid_points = 1

        self.exp.B_z = [1]
        self.exp.B_mw = 1
        self.exp.freq_mw = 1

    def test_hermitean(self):
        self.sys.spin_system = 'trip'
        self.cal.s = mt.Spinoperator(1)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)

        np.testing.assert_array_equal(ham_mw, np.conj(
            np.transpose(ham_mw, (0, 1, 3, 2))))

    def test_hermitean_rp(self):
        self.sys.spin_system = 'rp'
        self.cal.s = mt.Spinoperator(1/2, 1/2)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)

        np.testing.assert_array_equal(ham_mw, np.conj(
            np.transpose(ham_mw, (0, 1, 3, 2))))

    def test_rp(self):
        self.sys.spin_system = 'rp'
        self.cal.s = mt.Spinoperator(1/2, 1/2)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)
        ham_mw /= (2*np.pi)

        sqrt_2 = 1/np.sqrt(2)
        T = np.array([[1, 0, 0, 0], [0, sqrt_2, sqrt_2, 0],
                      [0, -sqrt_2, sqrt_2, 0], [0, 0, 0, 1]])
        sx = T.T @ self.cal.s.get('x') @ T
        sx /= sqrt_2
        sz = T.T @ self.cal.s.get('z') @ T

        comp = -sz + sx

        assert np.allclose(comp, ham_mw)

    def test_doub(self):
        self.sys.spin_system = 'doub'
        self.cal.s = mt.Spinoperator(1/2)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)
        comp = -self.cal.s.get('z') + self.cal.s.get('x')

        assert np.allclose(comp, ham_mw)

    def test_trip(self):
        self.sys.spin_system = 'trip'
        self.cal.s = mt.Spinoperator(1)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)
        comp = -self.cal.s.get('z') + self.cal.s.get('x')

        assert np.allclose(comp, ham_mw)

    def test_tdp(self):
        self.sys.spin_system = 'tdp'
        self.cal.s = mt.Spinoperator(1/2, 1)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)
        comp = -self.cal.s.get('z') + self.cal.s.get('x')

        assert np.allclose(comp, ham_mw)


    def test_freq_mw(self):
        self.exp.freq_mw = 7
        self.cal.s = mt.Spinoperator(1/2)
        self.sys.spin_system = 'doub'
        self.cal.g_iso = 1/MU_B

        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)

        assert ham_mw[0, 0, 0, 0] ==  -7/2
        assert ham_mw[0, 0, 1, 1] == 7/2

    def test_b_mw(self):
        self.exp.freq_mw = 1
        self.cal.s = mt.Spinoperator(1/2)
        self.sys.spin_system = 'doub'
        self.cal.g_iso = 7/MU_B

        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)

        assert ham_mw[0, 0, 0, 1] ==  7/2
        assert ham_mw[0, 0, 1, 0] == 7/2


class TestSetUpDoubletHamiltonian:

    def setup(self):
        initialize_classes(self)

        self.sys.s = 1/2
        self.sys.g = [2, 2, 2]
        self.sys.g_frame = [0, 0, 0]
        self.exp.B_z = np.linspace(1, 3, 3)/MU_B
        self.exp.B_mw = 7
        self.exp.freq_mw = 9
        self.opt.grid_points = 10
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(self.opt.grid_points)

        cr.set_up_spinoperator(self.sys, self.cal)
        self.cal.g_tensor = cr.create_tensor(
            self.sys.g, self.cal.phi, self.cal.theta)

        self.cal.ham = ham.set_up_doublet_hamiltonian(self.exp, self.opt,
                                                      self.cal)

    def test_shape(self):
        assert self.cal.ham.shape == (3, 10, 2, 2)

    def test_off_diagonals(self):
        assert np.array_equal(self.cal.ham[:, :, 0, 1], np.zeros((3, 10)))
        assert np.array_equal(self.cal.ham[:, :, 1, 0], np.zeros((3, 10)))

    def test_isotropic_zeeman(self):
        zeeman = np.broadcast_to((self.sys.g[0]*self.exp.B_z), (10, 3))
        zeeman = 0.5*MU_B*zeeman.T
        assert np.array_equal(self.cal.ham[:, :, 0, 0], zeeman)
        assert np.array_equal(self.cal.ham[:, :, 1, 1], -zeeman)

    def test_anisotripic_zeeman(self):
        self.sys.g = [2, 3, 4]
        self.cal.g_tensor = cr.create_tensor(
            self.sys.g, self.cal.phi, self.cal.theta)
        self.cal.ham = ham.set_up_doublet_hamiltonian(self.exp, self.opt,
                                                      self.cal)

        zeeman = np.broadcast_to((self.sys.g[0]*self.exp.B_z), (10, 3))
        zeeman = 0.5*MU_B*zeeman.T
        zeeman_1 = self.cal.g_tensor.multirot[3, 2, 2]*self.exp.B_z[0]*MU_B*0.5
        zeeman_2 = self.cal.g_tensor.multirot[5, 2, 2]*self.exp.B_z[0]*MU_B*-0.5
        print(self.cal.ham[0, 3, 0, 0])
        print(zeeman_2)
        np.testing.assert_allclose(self.cal.ham[0, 3, 0, 0], zeeman_1)
        np.testing.assert_allclose(self.cal.ham[0, 5, 1, 1], zeeman_2)

        if np.array_equal(zeeman_1, zeeman_2) is False:
            assert True
        else:
            assert False


class Test_set_up_triplet_hamiltonian:
    def setup(self):
        initialize_classes(self)

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)
        self.exp.B_z = np.linspace(1, 3, 3)
        self.exp.B_mw = 7
        self.exp.freq_mw = 3
        self.cal.s = mt.Spinoperator(1)

        self.sys.g_tri = [1, 2, 3]
        self.sys.D_tri = 5
        self.sys.E_tri = -2

        cr.set_up_tensors(self.sys, self.cal)

    def test_dipolar_coupling_elements(self):
        self.cal.g_tri_tensor = cr.create_tensor(
            [0, 0, 0], self.cal.phi, self.cal.theta)
        self.exp.B_mw = 0
        self.exp.freq_mw = 0
        self.cal.ham = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                      self.cal)

        d_zz = self.cal.D_tri_tensor.multirot[:, 2, 2]

        ham_dip = np.zeros((3, 3, 3), dtype=np.complex64)
        ham_dip[0, 0, 0] = +0.5*d_zz[0]
        ham_dip[0, 1, 1] = -1*d_zz[0]
        ham_dip[0, 2, 2] = +0.5*d_zz[0]

        ham_dip[1, 0, 0] = 0.5*d_zz[1]
        ham_dip[1, 1, 1] = -1*d_zz[1]
        ham_dip[1, 2, 2] = 0.5*d_zz[1]

        ham_dip[2, 0, 0] = 0.5*d_zz[2]
        ham_dip[2, 1, 1] = -1*d_zz[2]
        ham_dip[2, 2, 2] = 0.5*d_zz[2]

        assert np.array_equal(ham_dip, self.cal.ham[1])

    def test_zeeman_coupling_elements(self):
        self.exp.B_mw = 0
        self.exp.freq_mw = 0
        self.cal.D_tri_tensor = cr.create_tensor(
            [0, 0, 0], self.cal.phi, self.cal.theta)
        self.cal.ham = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                      self.cal)
        zeeman_interaction = self.exp.B_z*MU_B

        ham_zeeman = np.zeros((3, 3, 3, 3), dtype=np.complex64)
        for b in range(0, 3):
            for a in range(0, 3):
                ham_zeeman[b, a, 0, 0] = zeeman_interaction[b]\
                    * self.cal.g_tri_tensor.multirot[a, 2, 2]
                ham_zeeman[b, a, 2, 2] = -zeeman_interaction[b]\
                    * self.cal.g_tri_tensor.multirot[a, 2, 2]

        assert np.array_equal(ham_zeeman, self.cal.ham)


class Test_set_up_triplet_high_field_hamiltonian:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.exp = Exp()

        self.sys.g_tri = [2.00370, 2.00285, 2.00246]
        self.sys.rho_0_tri = [0, 0.33, 0.67]
        self.sys.D_tri = -700

        self.sys.E_tri = -200

        self.exp.B_z = np.linspace(342, 348, 4)

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)

        cr.set_up_tensors(self.sys, self.cal)

    def test_triplet_ham(self):
        H = comp.triplet_hamiltonian_analytical
        self.cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)
        np.testing.assert_allclose(self.cal.ham_tri_hf, H, atol=2e6)




class Test_set_up_rp_hamiltonian:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.exp = Exp()

        self.sys.spin_system = 'rp'
        self.sys.s = [1/2, 1/2]
        self.sys.g1 = [2.00431, 2.00360, 2.00217]
        self.sys.g1_frame = [0, 0, 0]
        self.sys.g2 = [2.00370, 2.00285, 2.00246]
        self.sys.g2_frame = [2.21656815, 1.34390352, 4.31096325]

        self.sys.precursor = 'singlet'

        self.sys.D = 10.0890*1e6
        self.sys.D_frame = [0, 1.9198621771937625, 1.9198621771937625]
        self.sys.E = 0*1e6
        self.sys.J_ex = 2.0458*1e6

        self.sys.dynamics = np.zeros((4, 4))

        self.exp.B_z = np.linspace(0.342, 0.348, 4)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 4
        self.exp.B_mw = 0.001*1e-3
        self.exp.freq_mw = 9.68*1e9

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)
        self.opt.space = 'hilbert'

        cr.set_up_tensors(self.sys, self.cal)
        inputs.predefinitions(self.sys, self.exp, self.cal)
        cr.set_up_spinoperator(self.sys, self.cal)

    def test_hilbert(self):
        self.cal.ham_rp = ham.set_up_rp_hamiltonian(self.sys, self.exp,
                                                    self.opt, self.cal)
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp,
                                           self.opt, self.cal)
        self.cal.ham = self.cal.ham_rp + ham_mw
        H = comp.H_RP_hilbert_analytical
        np.testing.assert_allclose(H, self.cal.ham, rtol=1e-4)

    def test_liouville(self):
        self.opt.space = 'liouville'
        self.cal.ham_rp = ham.set_up_rp_hamiltonian(self.sys, self.exp,
                                                    self.opt, self.cal)
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp,
                                           self.opt, self.cal)
        self.cal.ham = self.cal.ham_rp + ham_mw
        val, self.cal.eigvec = la.eigh(self.cal.ham)
        ham.set_up_commutator_superoperator(self.sys, self.opt, self.cal)
        H = comp.H_RP_hilbert_analytical
        H_adj = np.transpose(np.conjugate(H), [0, 1, 3, 2])
        H_super = np.kron(np.eye(4), H[:, :, :]) - \
            np.kron(H_adj[:, :, :], np.eye(4))
        np.testing.assert_allclose(H_super, self.cal.ham_superop, rtol=1e-4)


class Test_set_up_tdp_hamiltonian:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.exp = Exp()

        self.opt.grid_points = 1
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(1)
        self.exp.B_z = np.zeros(1)

        self.sys.g_tri = [2, 2, 2]
        self.sys.D_tri = 0
        self.sys.E_tri = 0
        self.sys.g = [3, 3, 3]
        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 0
        self.sys.s = [1/2, 1]

        cr.set_up_spinoperator(self.sys, self.cal)

        cr.set_up_tensors(self.sys, self.cal)

    def test_zeeman(self):
        ham_comp = np.array([3/2+2, 3/2, 3/2-2, -3/2+2, -3/2, -3/2-2])
        ham_comp = np.diag(ham_comp)
        self.exp.B_z = np.linspace(1, 1, 1)*1/MU_B
        ham_tdp = ham.set_up_tdp_hamiltonian(self.sys, self.exp, self.opt,
                                             self.cal)
        np.testing.assert_allclose(ham_tdp[0, 0], ham_comp)

    def test_dipole(self):
        wt = np.sqrt(2)
        ham_comp = np.array([[2, 0, 0, 0, 0, 0],
                             [0, 0, 0, -wt, 0, 0],
                             [0, 0, -2, 0, -wt, 0],
                             [0, -wt, 0, -2, 0, 0],
                             [0, 0, -wt, 0, 0, 0],
                             [0, 0, 0, 0, 0, 2]])*3/2
        self.cal.D_tensor = cr.create_tensor(
            [6, 6, 6], self.cal.phi, self.cal.theta)
        ham_tdp = ham.set_up_tdp_hamiltonian(self.sys, self.exp, self.opt,
                                             self.cal)
        np.testing.assert_allclose(ham_tdp[0, 0], ham_comp)

    def test_exchange(self):
        wt = np.sqrt(2)
        ham_comp = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 0, 0, wt, 0, 0],
                             [0, 0, -1, 0, wt, 0],
                             [0, wt, 0, -1, 0, 0],
                             [0, 0, wt, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1]])
        self.sys.J_ex = 2
        ham_tdp = ham.set_up_tdp_hamiltonian(self.sys, self.exp, self.opt,
                                             self.cal)
        np.testing.assert_allclose(ham_tdp[0, 0], ham_comp)

    def test_zfs(self):
        ham_comp = np.array([5, -10, 5, 5, -10, 5])
        ham_comp = np.diag(ham_comp)/2
        self.cal.D_tri_tensor = cr.create_tensor(
            [5, 5, 5], self.cal.phi, self.cal.theta)
        ham_tdp = ham.set_up_tdp_hamiltonian(self.sys, self.exp, self.opt,
                                             self.cal)
        np.testing.assert_allclose(ham_tdp[0, 0], ham_comp)


class Test_set_up_commutator_superoperator:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.exp = Exp()

        self.opt.space = 'liouville'
        self.sys.dynamics = np.zeros((3, 3))

    def test_commutator_value(self):
        hamiltonian = np.arange(1, 10).reshape(3, 3)
        self.cal.ham = np.broadcast_to(hamiltonian, (2, 2, 3, 3))
        val, self.cal.eigvec = la.eigh(self.cal.ham)
        ham_adj = np.conjugate(hamiltonian.T)
        superop = np.kron(np.eye(3), hamiltonian) - np.kron(ham_adj, np.eye(3))
        superop_3d = np.broadcast_to(superop, (2, 2, 9, 9))

        ham.set_up_commutator_superoperator(self.sys, self.opt, self.cal)

        assert np.array_equal(self.cal.ham_superop, superop_3d)
