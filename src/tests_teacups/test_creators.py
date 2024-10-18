import teacups.grid as grid
import teacups.matrix_tools as mt
import teacups.creators as cr
import numpy as np
import sys
sys.path.append("./..")


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


class Test_create_tensor:
    def setup(self):
        diag = [1, 2, 3]
        self.theta, self.phi = grid.get_theta_phi(3)
        g1Frame = [1, 2, 3]
        self.ten1 = cr.create_tensor(diag, self.phi, self.theta)
        self.ten2 = cr.create_tensor(diag, self.phi, self.theta, g1Frame)

    def test_tensor(self):
        comp1 = np.array([[1., 0., 0.],
                          [0., 2., 0.],
                          [0., 0., 3.]])

        comp2 = np.array([[2.79957166,  0.02570897, -0.42563375],
                          [0.02570897,  1.26862141, -0.47826256],
                          [-0.42563375, -0.47826256,  1.93180692]])
        np.testing.assert_allclose(
            comp1, self.ten1.tensor, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(
            comp2, self.ten2.tensor, atol=1e-8, rtol=1e-5)

    def test_multirot(self):
        comp1 = mt.Tensor(np.array([1, 2, 3]))
        comp1.multirotation(self.phi, self.theta)

        comp2 = mt.Tensor(np.array([1, 2, 3]))
        comp2.rotation(1, 2, 3)
        comp2.tensor = comp2.rot
        comp2.multirotation(self.phi, self.theta)

        np.testing.assert_allclose(
            comp1.multirot, self.ten1.multirot, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(
            comp2.multirot, self.ten2.multirot, atol=1e-8, rtol=1e-5)


class Test_create_dipol_tensor_diagonals:
    D = 3
    E = 1
    a = -1/3*D+E
    b = -1/3*D-E
    c = 2/3*D
    dig = cr.create_dipol_tensor_diagonals(D, E)

    def test_values(self):
        comp = np.array([self.a, self.b, self.c])
        assert np.array_equal(comp, self.dig)


class Test_set_up_tensors_doublet:

    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()
        self.cal = Cal()

        self.sys.g = [1, 2, 3]
        self.sys.g_frame = [0, 0, 0]
        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.get_theta_phi(3)

    def test_g(self):
        g_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tensor.multirot, self.cal.g_tensor.multirot)

    def test_g_with_frame(self):
        self.sys.g_frame = [1, 2, 3]
        self.opt.grid_points = 8
        g_tensor = cr.create_tensor(
            [1, 2, 3], self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tensor.multirot, self.cal.g_tensor.multirot)

    def test_g_iso(self):
        g_iso = 1/3*np.sum(self.sys.g)
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class Test_set_up_tensors_rp:

    def setup(self):
        self.opt = Opt()
        self.sys = Sys()
        self.cal = Cal()

        self.opt.grid_points = 20
        self.cal.theta, self.cal.phi = grid.get_theta_phi(20)

        self.sys.g1 = [1, 2, 3]
        self.sys.g2 = [2, 3, 4]
        self.sys.g2_frame = [0.5, 1, 1.5]
        self.sys.g_tri = [4, 5, 6]
        self.sys.precursor = 'singlet'

    def test_g1(self):
        g1_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g1_tensor.multirot, self.cal.g1_tensor.multirot)

    def test_g2(self):
        g2_tensor = cr.create_tensor(
            [2, 3, 4], self.cal.phi, self.cal.theta, [0.5, 1, 1.5])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g2_tensor.multirot, self.cal.g2_tensor.multirot)

    def test_no_Ds(self):
        cr.set_up_tensors(self.sys, self.cal)
        D_tensor = cr.create_tensor(cr.create_dipol_tensor_diagonals(
            0, 0), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_D_1(self):
        self.sys.D = 5
        D_tensor = cr.create_tensor(cr.create_dipol_tensor_diagonals(
            5, 0), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_2(self):
        self.sys.E = 5
        D_tensor = cr.create_tensor(cr.create_dipol_tensor_diagonals(
            0, 5), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_3(self):
        self.sys.D = 5
        self.sys.E = 1
        D_tensor = cr.create_tensor(cr.create_dipol_tensor_diagonals(
            5, 1), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_4(self):
        self.sys.D = 5
        self.sys.E = 1
        self.sys.D_frame = [1, 2, 3]
        D_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_gT_1(self):
        self.sys.precursor = 'triplet'
        gT_tensor = cr.create_tensor([4, 5, 6], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(gT_tensor.multirot,
                              self.cal.g_tri_tensor.multirot)

    def test_gT_2(self):
        self.sys.precursor = 'triplet'
        self.sys.g_tri_frame = [1, 2, 3]
        gT_tensor = cr.create_tensor(
            [4, 5, 6], self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(gT_tensor.multirot,
                              self.cal.g_tri_tensor.multirot)

    def test_DT_1(self):
        self.sys.precursor = 'triplet'
        self.sys.D_tri = 5
        DT_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 0), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_DT_2(self):
        self.sys.precursor = 'triplet'
        self.sys.E_tri = 5
        DT_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(0, 5), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_DT_3(self):
        self.sys.precursor = 'triplet'
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        DT_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_DT_4(self):
        self.sys.precursor = 'triplet'
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        self.sys.D_tri_frame = [1, 2, 3]
        DT_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_g_iso(self):
        g_iso = 0.5*(1/3*np.sum(self.sys.g1) + 1/3*np.sum(self.sys.g2))
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class Test_set_up_tensors_triplet:
    def setup(self):
        self.opt = Opt()
        self.sys = Sys()
        self.cal = Cal()

        self.opt.grid_points = 20
        self.cal.theta, self.cal.phi = grid.get_theta_phi(20)
        self.sys.g_tri = [4, 5, 6]

    def test_g(self):
        g_tri_tensor = cr.create_tensor(
            [4, 5, 6], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tri_tensor.multirot,
                              self.cal.g_tri_tensor.multirot)

    def test_g_with_frame(self):
        self.sys.g_tri_frame = [1, 2, 3]
        g_tri_tensor = cr.create_tensor(
            [4, 5, 6], self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tri_tensor.multirot,
                              self.cal.g_tri_tensor.multirot)

    def test_dipole_tensor_only_D(self):
        self.sys.D_tri = 5
        D_tri_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 0), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_only_E(self):
        self.sys.E_tri = 5
        D_tri_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(0, 5), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_D_and_E(self):
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        D_tri_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_with_frame(self):
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        self.sys.D_tri_frame = [1, 2, 3]
        D_tri_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_no_D_and_E(self):
        D_tri_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(0, 0), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_g_iso(self):
        g_iso = 1/3*(np.sum(self.sys.g_tri))
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class Test_set_up_tensors_tdp:
    def setup(self):
        self.opt = Opt()
        self.sys = Sys()
        self.cal = Cal()

        self.opt.grid_points = 20
        self.cal.theta, self.cal.phi = grid.get_theta_phi(20)
        self.sys.g = [1, 2, 3]
        self.sys.g_tri = [4, 5, 6]
        self.sys.D_tri = 5
        self.sys.E_tri = 1

    def test_g(self):
        g_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tensor.multirot, self.cal.g_tensor.multirot)

    def test_g_tri(self):
        g_tri_tensor = cr.create_tensor(
            [4, 5, 6], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tri_tensor.multirot,
                              self.cal.g_tri_tensor.multirot)

    def test_D(self):
        self.sys.D = 5
        self.sys.E = 1
        D_tensor = cr.create_tensor(cr.create_dipol_tensor_diagonals(
            5, 1), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_tri(self):
        D_tri_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot,
                              self.cal.D_tri_tensor.multirot)

    def test_g_iso(self):
        g_iso = 1/2*(1/3*(np.sum(self.sys.g_tri))+1/3*(np.sum(self.sys.g)))
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class Test_set_up_spinoperator:
    def setup(self):
        self.cal = Cal()
        self.sys = Sys()

    def test_s_coupled(self):
        # radical pair
        s1 = np.array([[[0. + 0.j,  0. + 0.j,  0.5+0.j,  0. + 0.j],
                        [0. + 0.j,  0. + 0.j,  0. + 0.j,  0.5+0.j],
                        [0.5+0.j,  0. + 0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j,  0.5+0.j,  0. + 0.j,  0. + 0.j]],

                       [[0. + 0.j,  0. + 0.j,  0. - 0.5j,  0. + 0.j],
                        [0. + 0.j,  0. + 0.j,  0. + 0.j,  0. - 0.5j],
                        [0. + 0.5j,  0. + 0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j,  0. + 0.5j,  0. + 0.j,  0. + 0.j]],

                       [[0.5+0.j,  0. + 0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j,  0.5+0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j,  0. + 0.j, -0.5+0.j, -0. + 0.j],
                        [0. + 0.j,  0. + 0.j, -0. + 0.j, -0.5+0.j]]])

        s2 = np.array([[[0. + 0.j,  0.5+0.j,  0. + 0.j,  0. + 0.j],
                        [0.5+0.j,  0. + 0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j,  0. + 0.j,  0. + 0.j,  0.5+0.j],
                        [0. + 0.j,  0. + 0.j,  0.5+0.j,  0. + 0.j]],

                       [[0. + 0.j,  0. - 0.5j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.5j,  0. + 0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j,  0. + 0.j,  0. + 0.j,  0. - 0.5j],
                        [0. + 0.j,  0. + 0.j,  0. + 0.5j,  0. + 0.j]],

                       [[0.5+0.j,  0. + 0.j,  0. + 0.j,  0. + 0.j],
                        [0. + 0.j, -0.5+0.j,  0. + 0.j, -0. + 0.j],
                        [0. + 0.j,  0. + 0.j,  0.5+0.j,  0. + 0.j],
                        [0. + 0.j, -0. + 0.j,  0. + 0.j, -0.5+0.j]]])
        S = s1 + s2
        self.sys.s = [1/2, 1/2]
        cr.set_up_spinoperator(self.sys, self.cal)
        assert np.array_equal(self.cal.s.matrix, S)

    def test_s_uncoupled(self):
        # Doublet
        s = np.array([[[0.+0.j,  0.5+0.j],
                       [0.5+0.j, 0.+0.j]],

                      [[0.-0.j, 0.-0.5j],
                       [0.+0.5j, 0.-0.j]],

                      [[0.5+0.j, 0.+0.j],
                       [0.+0.j, -0.5+0.j]]])

        self.sys.s = 1/2
        cr.set_up_spinoperator(self.sys, self.cal)
        assert np.array_equal(self.cal.s.matrix, s)


class Test_set_up_observable:
    def setup(self):
        self.opt = Opt()
        self.cal = Cal()
        self.sys = Sys()

        self.sys.spin_system = 'rp'
        self.sys.s = [1/2, 1/2]
        self.opt.space = 'hilbert'
        cr.set_up_spinoperator(self.sys, self.cal)

    def test_hilbert_signal_rp(self):
        sig = np.array([[0.-0.j, 0.-0.j, 0.-0.70710678j, 0.-0.j],
                        [0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j],
                        [0.+0.70710678j, 0.-0.j, 0.-0.j, 0.-0.70710678j],
                        [0.-0.j, 0.-0.j, 0.+0.70710678j, 0.-0.j]])
        cr.set_up_observable(self.sys, self.opt, self.cal)
        np.testing.assert_allclose(sig, self.cal.observable)

    def test_liouville_signal_rp(self):
        sig = np.array([0.-0.j, 0.-0.j, 0.-0.70710678j, 0.-0.j,
                        0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j,
                        0.+0.70710678j, 0.-0.j, 0.-0.j, 0.-0.70710678j,
                        0.-0.j, 0.-0.j, 0.+0.70710678j, 0.-0.j])
        self.opt.space = 'liouville'
        cr.set_up_observable(self.sys, self.opt, self.cal)
        np.testing.assert_allclose(sig, self.cal.observable)

    def test_hilbert_signal_doub(self):
        self.sys.spin_system = 'doub'
        self.sys.s = [1/2]
        cr.set_up_spinoperator(self.sys, self.cal)
        cr.set_up_observable(self.sys, self.opt, self.cal)
        sig = self.cal.s.get('y')
        assert np.array_equal(sig, self.cal.observable)

    def test_liouville_signal_trip(self):
        self.sys.spin_system = 'trip'
        self.sys.s = [1]
        self.opt.space = 'liouville'
        cr.set_up_spinoperator(self.sys, self.cal)
        cr.set_up_observable(self.sys, self.opt, self.cal)
        sig = self.cal.s.get('y')
        sig = sig.flatten()
        assert np.array_equal(sig, self.cal.observable)
