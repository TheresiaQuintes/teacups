import numpy.linalg as la
import scipy.constants as const
import teacups.grid as grid
import teacups.creators as cr
import teacups.matrix_tools as mt
import teacups.hamiltonians as ham
import teacups.multioperator_tools as mut
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

    def test_dtype(self):
        self.sys.spin_system = 'trip'
        self.cal.s = mt.Spinoperator(1)
        self.cal.g_iso = 1/MU_B
        ham_mw = ham.set_up_mw_hamiltonian(self.sys, self.exp, self.opt,
                                           self.cal)
        assert ham_mw.dtype == 'complex64'

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
        self.opt.grid_points = 10
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(self.opt.grid_points)

        cr.set_up_spinoperator(self.sys, self.cal)
        self.cal.g_tensor = cr.create_tensor(
            self.sys.g, self.cal.phi, self.cal.theta)

        self.cal.ham = ham.set_up_doublet_hamiltonian(self.exp, self.opt,
                                                      self.cal)

    def test_shape(self):
        assert self.cal.ham.shape == (3, 10, 2, 2)

    def test_dtype(self):
        assert self.cal.ham.dtype == 'complex64'

    def test_hermitean(self):
        np.testing.assert_array_equal(self.cal.ham, np.conj(
            np.transpose(self.cal.ham, (0, 1, 3, 2))))

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

        np.testing.assert_allclose(self.cal.ham[0, 3, 0, 0], zeeman_1)
        np.testing.assert_allclose(self.cal.ham[0, 5, 1, 1], zeeman_2)

        if np.array_equal(zeeman_1, zeeman_2) is False:
            assert True
        else:
            assert False


class TestSetUpTripletHamiltonian:
    def setup(self):
        initialize_classes(self)

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)
        self.exp.B_z = np.linspace(1, 4, 4)/MU_B
        self.cal.s = mt.Spinoperator(1)

        self.sys.g_tri = [1, 2, 3]
        self.sys.D_tri = 5
        self.sys.E_tri = -2

        cr.set_up_tensors(self.sys, self.cal)

        self.cal.ham = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                      self.cal)

    def test_shape(self):
        assert self.cal.ham.shape == (4, 3, 3, 3)

    def test_dtype(self):
        assert self.cal.ham.dtype == 'complex64'

    def test_hermitean(self):
        np.testing.assert_array_equal(self.cal.ham, np.conj(
            np.transpose(self.cal.ham, (0, 1, 3, 2))))

    def test_isotropic_zeeman(self):
        self.sys.D_tri = 0
        self.sys.E_tri = 0
        self.sys.g_tri = [2, 2, 2]
        cr.set_up_tensors(self.sys, self.cal)

        zeeman = np.broadcast_to((self.sys.g_tri[0]*self.exp.B_z), (3, 4))
        zeeman = MU_B*zeeman.T
        self.cal.ham = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                      self.cal)

        assert np.array_equal(self.cal.ham[:, :, 0, 0], zeeman)
        assert np.array_equal(self.cal.ham[:, :, 1, 1], 0*zeeman)
        assert np.array_equal(self.cal.ham[:, :, 2, 2], -zeeman)

    def test_dipolar_coupling_elements(self):
        self.cal.g_tri_tensor = cr.create_tensor(
            [0, 0, 0], self.cal.phi, self.cal.theta)
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

        assert np.array_equal(ham_dip, self.cal.ham[0])
        assert np.array_equal(ham_dip, self.cal.ham[1])
        assert np.array_equal(ham_dip, self.cal.ham[2])


    def test_zeeman_coupling_elements(self):
        self.cal.D_tri_tensor = cr.create_tensor(
            [0, 0, 0], self.cal.phi, self.cal.theta)
        self.cal.ham = ham.set_up_triplet_hamiltonian(self.exp, self.opt,
                                                      self.cal)
        zeeman_interaction = self.exp.B_z*MU_B

        ham_zeeman = np.zeros((4, 3, 3, 3), dtype=np.complex64)
        for b in range(0, 4):
            for a in range(0, 3):
                ham_zeeman[b, a, 0, 0] = zeeman_interaction[b]\
                    * self.cal.g_tri_tensor.multirot[a, 2, 2]
                ham_zeeman[b, a, 2, 2] = -zeeman_interaction[b]\
                    * self.cal.g_tri_tensor.multirot[a, 2, 2]

        assert np.array_equal(ham_zeeman, self.cal.ham)


class TestSetUpTripletHighFieldHamiltonian:
    def setup(self):
        initialize_classes(self)

        self.sys.g_tri = [2.00370, 2.00285, 2.00246]
        self.sys.rho_0_tri = [0, 0.33, 0.67]
        self.sys.D_tri = -700

        self.sys.E_tri = -200

        self.exp.B_z = np.linspace(342, 348, 4)

        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)

        cr.set_up_tensors(self.sys, self.cal)

        self.cal.ham = ham.set_up_triplet_high_field_hamiltonian(
            self.exp, self.opt, self.cal)

    def test_shape(self):
        assert self.cal.ham.shape == (4, 3, 3, 3)

    def test_dtype(self):
        assert self.cal.ham.dtype == 'complex64'

    def test_hermitean(self):
        np.testing.assert_allclose(self.cal.ham, np.conj(
            np.transpose(self.cal.ham, (0, 1, 3, 2))), atol=2e6)


    def test_basis_transformation(self):
        s_tri = mt.Spinoperator(1)
        ham_d = mut.Multioperator(s_tri, self.opt.grid_points, self.exp.B_z*MU_B)
        ham_d.create_bilinear_operator(self.cal.D_tri_tensor, s_tri)
        ham_d.angle_matrix_changed()

        eig_d, vec_d = np.linalg.eigh(ham_d.B_angle_matrix)

        ham_tri_hf = mut.Multioperator(s_tri, self.opt.grid_points, self.exp.B_z*MU_B)
        ham_tri_hf.zeeman_coupling(self.cal.g_tri_tensor)
        ham_tri_hf.B_angle_matrix += ham_d.B_angle_matrix

        back_transformed = vec_d @ self.cal.ham @ np.conj(
            np.transpose(vec_d, (0, 1, 3, 2)))
        np.testing.assert_allclose(
            ham_tri_hf.B_angle_matrix, back_transformed, atol=2e6)

class TestSetUpRpHamiltonian:
    def setup(self):
        initialize_classes(self)

        self.sys.s = [1/2, 1/2]
        self.sys.g1 = [2, 2, 2]
        self.sys.g2 = [3, 3, 3]
        self.sys.D = 0/(2*np.pi)
        self.sys.E = 0/(2*np.pi)
        self.sys.J_ex = 3/(2*np.pi)

        self.exp.B_z = np.linspace(1, 3, 3)/(MU_B*2*np.pi)
        self.cal.s = mt.Spinoperator(1/2, 1/2)

        self.opt.grid_points = 5
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(self.opt.grid_points)

        cr.set_up_tensors(self.sys, self.cal)

        self.cal.ham = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                      self.cal)

    def test_shape(self):
        assert self.cal.ham.shape == (3, 5, 4, 4)

    def test_dtype(self):
        assert self.cal.ham.dtype == 'complex64'

    def test_hermitean(self):
        np.testing.assert_array_equal(self.cal.ham, np.conj(
            np.transpose(self.cal.ham, (0, 1, 3, 2))))

    def test_zeros_off_diagonals(self):
        z = np.zeros((3, 5))
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 0, 1])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 0, 2])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 0, 3])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 1, 0])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 1, 3])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 2, 0])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 2, 3])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 3, 0])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 3, 1])
        np.testing.assert_array_equal(z, self.cal.ham[:, :, 3, 2])

    def test_isotropic_g(self):
        self.sys.J_ex = 0

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                 self.cal)
        omega = np.linspace(1, 3, 3)*(2+3)/2
        delta_omega = np.linspace(1, 3, 3)*(2-3)/2

        np.testing.assert_array_equal(omega, self.cal.ham[:, 0, 0, 0])
        np.testing.assert_array_equal(-omega, self.cal.ham[:, 0, 3, 3])
        np.testing.assert_array_equal(delta_omega, self.cal.ham[:, 0, 1, 2])
        np.testing.assert_array_equal(delta_omega, self.cal.ham[:, 0, 2, 1])


    def test_exchange_interaction(self):
        self.sys.g1 = [0, 0, 0]
        self.sys.g2 = [0, 0, 0]

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                 self.cal)

        J = np.ones((3, 5))*3
        np.testing.assert_array_equal(-J, self.cal.ham[:, :, 0, 0])
        np.testing.assert_array_equal(J, self.cal.ham[:, :, 1, 1])
        np.testing.assert_array_equal(-J, self.cal.ham[:, :, 2, 2])
        np.testing.assert_array_equal(-J, self.cal.ham[:, :, 3, 3])

    def test_anisotropic_g(self):
        self.sys.g1 = [1, 2, 3]
        self.sys.g2 = [4, 5, 6]
        self.sys.J_ex = 0

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                 self.cal)

        sum_g = (self.cal.g1_tensor.multirot[:, 2, 2]+
                 self.cal.g2_tensor.multirot[:, 2, 2])/2

        diff_g = (self.cal.g1_tensor.multirot[:, 2, 2]-
                 self.cal.g2_tensor.multirot[:, 2, 2])/2

        omega = np.linspace(1, 3, 3)[:, np.newaxis]*sum_g
        delta_omega = np.linspace(1, 3, 3)[:, np.newaxis]*diff_g

        np.testing.assert_allclose(omega, self.cal.ham[:, :, 0, 0])
        np.testing.assert_allclose(-omega, self.cal.ham[:, :, 3, 3])
        np.testing.assert_allclose(delta_omega, self.cal.ham[:, :, 1, 2])
        np.testing.assert_allclose(delta_omega, self.cal.ham[:, :, 2, 1])


    def test_dipolar_coupling(self):
        self.sys.g1 = [0, 0, 0]
        self.sys.g2 = [0, 0, 0]
        self.sys.J_ex = 0

        self.sys.D = 3/(2*np.pi)
        self.sys.E = 1/(2*np.pi)

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_rp_hamiltonian(self.sys, self.exp, self.opt,
                                                 self.cal)

        D = self.cal.D_tensor.multirot[:, 2, 2]*2*np.pi
        np.testing.assert_array_equal(-1/2*D, self.cal.ham[0, :, 0, 0])


class TestSetUpTdpHamiltonianIsotrope:
    def setup(self):
        initialize_classes(self)

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


class TestSetUpTdpHamiltonianAnisotrope:
    def setup(self):
        initialize_classes(self)

        self.opt.grid_points = 5
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(5)
        self.exp.B_z = np.linspace(1, 4, 4)/MU_B

        self.sys.g_tri = [1, 2, 3]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [4, 5, 6]
        self.sys.D = 5
        self.sys.E = 6
        self.sys.J_ex = 0
        self.sys.s = [1/2, 1]

        cr.set_up_spinoperator(self.sys, self.cal)

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_hamiltonian(self.sys, self.exp,
                                                  self.opt, self.cal)

    def test_shape(self):
        assert self.cal.ham.shape == (4, 5, 6, 6)

    def test_dtype(self):
        assert self.cal.ham.dtype == 'complex64'

    def test_hermitean(self):
        np.testing.assert_array_equal(self.cal.ham, np.conj(
            np.transpose(self.cal.ham, (0, 1, 3, 2))))

    def test_doublet_part(self):
        self.sys.g_tri = [0, 0, 0]
        self.sys.D_tri = 0
        self.sys.E_tri = 0
        self.sys.D = 0
        self.sys.E = 0

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_hamiltonian(self.sys, self.exp,
                                                  self.opt, self.cal)

        self.sys.s = 1/2
        cr.set_up_spinoperator(self.sys, self.cal)
        d = ham.set_up_doublet_hamiltonian(self.exp, self.opt, self.cal)
        kron_d = np.kron(d, np.eye(3))

        np.testing.assert_allclose(kron_d, self.cal.ham)

    def test_triplet_part(self):
        self.sys.g = [0, 0, 0]
        self.sys.D = 0
        self.sys.E = 0
        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_hamiltonian(self.sys, self.exp,
                                                  self.opt, self.cal)

        self.sys.s = 1
        cr.set_up_spinoperator(self.sys, self.cal)
        t = ham.set_up_triplet_hamiltonian(self.exp, self.opt, self.cal)
        kron_t = np.kron(np.eye(2), t)

        np.testing.assert_allclose(kron_t, self.cal.ham)


    def test_dipolar_couplings(self):
        self.sys.g = [0, 0, 0]
        self.sys.g_tri = [0, 0, 0]
        self.sys.D_tri = 0
        self.sys.E_tri = 0

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_hamiltonian(self.sys, self.exp,
                                                  self.opt, self.cal)

        d = np.ones((4, 5))*self.cal.D_tensor.multirot[:, 2, 2]
        wt = 1/np.sqrt(2)

        np.testing.assert_allclose(self.cal.ham[:, :, 0, 0], d)
        np.testing.assert_allclose(self.cal.ham[:, :, 1, 3], -wt*d)
        np.testing.assert_allclose(self.cal.ham[:, :, 2, 2], -d)
        np.testing.assert_allclose(self.cal.ham[:, :, 2, 4], -wt*d)
        np.testing.assert_allclose(self.cal.ham[:, :, 3, 1], -wt*d)
        np.testing.assert_allclose(self.cal.ham[:, :, 3, 3], -d)
        np.testing.assert_allclose(self.cal.ham[:, :, 4, 2], -wt*d)
        np.testing.assert_allclose(self.cal.ham[:, :, 5, 5], d)


class TestSetUpTdpFullHighFieldHamiltonian:
    def setup(self):
        initialize_classes(self)

        self.opt.grid_points = 5
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(5)
        self.exp.B_z = np.linspace(1, 4, 4)/MU_B

        self.sys.g_tri = [1, 2, 3]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [4, 5, 6]
        self.sys.D = 5
        self.sys.E = 6
        self.sys.J_ex = 7
        self.sys.s = [1/2, 1]

        cr.set_up_spinoperator(self.sys, self.cal)

        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_full_high_field_hamiltonian(self.sys,
                                                                  self.exp,
                                                                  self.opt,
                                                                  self.cal)

        s_trip = mt.Spinoperator(1/2, 1)
        s_trip.matrix = s_trip.matrix_coupling_spins[0]

        s_doub = mt.Spinoperator(1/2, 1)

        self.ham_zfs = mut.Multioperator(s_trip, self.opt.grid_points,
                                         self.exp.B_z*MU_B)
        self.ham_zfs.create_bilinear_operator(self.cal.D_tri_tensor, s_trip)
        self.ham_zfs.angle_matrix_changed()
        self.ham_zfs = self.ham_zfs.B_angle_matrix
        _, self.vec_zfs = np.linalg.eigh(self.ham_zfs)

        self.ham_zeeman_doub = mut.Multioperator(s_doub,
                                                 self.opt.grid_points,
                                                 self.exp.B_z*MU_B)
        self.ham_zeeman_doub.zeeman_coupling(self.cal.g_tensor)

        self.ham_zeeman_trip = mut.Multioperator(s_trip,
                                                 self.opt.grid_points,
                                                 self.exp.B_z*MU_B)
        self.ham_zeeman_trip.zeeman_coupling(self.cal.g_tri_tensor)

        self.ham_dip = mut.Multioperator(s_doub, self.opt.grid_points,
                                         self.exp.B_z*MU_B)
        self.ham_dip.create_bilinear_operator(self.cal.D_tensor, s_trip)
        self.ham_dip.angle_matrix_changed()

        self.ham_ex = mut.Multioperator(s_doub, self.opt.grid_points,
                                         self.exp.B_z*MU_B)
        self.ham_ex.exchange_coupling(-1/2*self.sys.J_ex, s_trip)


    def test_shape(self):
        assert self.cal.ham.shape == (4, 5, 6, 6)

    def test_dtype(self):
        assert self.cal.ham.dtype == 'complex64'

    def test_hermitean(self):
        np.testing.assert_allclose(self.cal.ham, np.conj(
            np.transpose(self.cal.ham, (0, 1, 3, 2))), atol=2e6)

    def test_zfs(self):
        self.sys.g_tri = [0, 0, 0]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [0, 0, 0]
        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 0
        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_full_high_field_hamiltonian(self.sys,
                                                                  self.exp,
                                                                  self.opt,
                                                                  self.cal)

        ham_zfs = np.conj(np.transpose(self.vec_zfs, (0, 1, 3, 2))) @\
            self.ham_zfs @ self.vec_zfs
        np.testing.assert_allclose(self.cal.ham, ham_zfs, atol=2e-6)

    def test_zeeman_triplet(self):
        self.sys.g_tri = [1, 2, 3]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [0, 0, 0]
        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 0
        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_full_high_field_hamiltonian(self.sys,
                                                                  self.exp,
                                                                  self.opt,
                                                                  self.cal)

        self.cal.ham = self.vec_zfs @ self.cal.ham @\
            np.conj(np.transpose(self.vec_zfs, (0, 1, 3, 2)))-self.ham_zfs

        zeeman = self.ham_zeeman_trip.B_angle_matrix
        np.testing.assert_allclose(self.cal.ham, zeeman, atol=2e-6)

    def test_zeeman_doublet(self):
        self.sys.g_tri = [0, 0, 0]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [4, 5, 6]
        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 0
        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_full_high_field_hamiltonian(self.sys,
                                                                  self.exp,
                                                                  self.opt,
                                                                  self.cal)

        self.cal.ham = self.vec_zfs @ self.cal.ham @\
            np.conj(np.transpose(self.vec_zfs, (0, 1, 3, 2)))-self.ham_zfs

        zeeman = self.ham_zeeman_doub.B_angle_matrix
        np.testing.assert_allclose(self.cal.ham, zeeman, atol=3e-6)

    def test_dipolar_coupling(self):
        self.sys.g_tri = [0, 0, 0]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [0, 0, 0]
        self.sys.D = 5
        self.sys.E = 6
        self.sys.J_ex = 0
        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_full_high_field_hamiltonian(self.sys,
                                                                  self.exp,
                                                                  self.opt,
                                                                  self.cal)

        self.cal.ham = self.vec_zfs @ self.cal.ham @\
            np.conj(np.transpose(self.vec_zfs, (0, 1, 3, 2)))-self.ham_zfs

        dipolar = self.ham_dip.B_angle_matrix
        np.testing.assert_allclose(self.cal.ham, dipolar, atol=2e-6)

    def test_exchange_coupling(self):
        self.sys.g_tri = [0, 0, 0]
        self.sys.D_tri = 4
        self.sys.E_tri = 3
        self.sys.g = [0, 0, 0]
        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 7
        cr.set_up_tensors(self.sys, self.cal)
        self.cal.ham = ham.set_up_tdp_full_high_field_hamiltonian(self.sys,
                                                                  self.exp,
                                                                  self.opt,
                                                                  self.cal)

        self.cal.ham = self.vec_zfs @ self.cal.ham @\
            np.conj(np.transpose(self.vec_zfs, (0, 1, 3, 2)))-self.ham_zfs

        exchange = self.ham_ex.B_angle_matrix
        np.testing.assert_allclose(self.cal.ham, exchange, atol=2e-6)


class TestSetUpCommutatorSuperoperator:
    def setup(self):
        initialize_classes(self)

        self.opt.space = 'liouville'

        self.hamiltonian = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        self.cal.ham = np.broadcast_to(self.hamiltonian, (2, 2, 3, 3))
        _, self.cal.eigvec = la.eigh(self.cal.ham)
        self.sys.dynamics = np.diag([1, 2, 3])
        ham.set_up_commutator_superoperator(self.sys, self.opt, self.cal)
        self.superop = self.cal.ham_superop

    def test_shape(self):
        assert self.superop.shape == (2, 2, 9, 9)

    def test_dtype(self):
        assert self.superop.dtype == 'complex64'

    def test_commutator(self):
        self.sys.dynamics = np.zeros((3, 3))

        ham_adj = np.conjugate(self.hamiltonian.T)
        superop = np.kron(self.hamiltonian, np.eye(3))\
            - np.kron(np.eye(3), ham_adj)
        superop_3d = np.broadcast_to(superop, (2, 2, 9, 9))

        ham.set_up_commutator_superoperator(self.sys, self.opt, self.cal)

        np.testing.assert_array_equal(self.cal.ham_superop, superop_3d)

    def test_relaxation(self):
        hamiltonian = np.zeros((3, 3))
        self.cal.ham = np.broadcast_to(hamiltonian, (2, 2, 3, 3))
        val, self.cal.eigvec = la.eigh(self.cal.ham)

        ham.set_up_commutator_superoperator(self.sys, self.opt, self.cal)

        relaxation = np.zeros((2, 2, 9, 9), dtype=np.complex128)
        relaxation[:, :, 0, 0] = 1j
        relaxation[:, :, 4, 4] = 2j
        relaxation[:, :, 8, 8] = 3j
        np.testing.assert_array_equal(self.cal.ham_superop, relaxation)

    def test_shape_hilbert(self):
        self.opt.space = 'hilbert'

        self.hamiltonian = np.arange(11, 20, dtype=np.float32).reshape(3, 3)
        self.cal.ham = np.broadcast_to(self.hamiltonian, (2, 2, 3, 3))
        _, self.cal.eigvec = la.eigh(self.cal.ham)
        self.sys.dynamics = np.diag([3, 4, 5])

        ham.set_up_commutator_superoperator(self.sys, self.opt, self.cal)
        # superoperator-attribute is not changed if opt.space='hilbert'
        np.testing.assert_array_equal(self.superop, self.cal.ham_superop)
