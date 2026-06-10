import teacups.grid as grid
import teacups.matrix_tools as mt
import teacups.creators as cr
import numpy as np


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


class TestCreateTensor:
    def setup_method(self):
        diag = [1, 2, 3]
        self.theta, self.phi = grid.fibonacci_grid(3)
        g1Frame = [1, 2, 3]
        self.ten1 = cr.create_tensor(diag, self.phi, self.theta)
        self.ten2 = cr.create_tensor(diag, self.phi, self.theta, g1Frame)

    def test_dtype(self):
        assert self.ten1.matrix.dtype == "float32"
        assert self.ten2.matrix.dtype == "float32"
        assert self.ten1.multirot.dtype == "float32"
        assert self.ten2.multirot.dtype == "float32"

    def test_tensor(self):
        comp1 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

        comp2 = np.array(
            [
                [2.79957166, 0.02570897, -0.42563375],
                [0.02570897, 1.26862141, -0.47826256],
                [-0.42563375, -0.47826256, 1.93180692],
            ]
        )
        np.testing.assert_allclose(comp1, self.ten1.matrix)
        np.testing.assert_allclose(comp2, self.ten2.matrix, atol=1e-7)

    def test_multirot(self):
        comp1 = mt.Tensor(np.array([1, 2, 3]))
        comp1.multirotation(self.phi, self.theta)

        comp2 = mt.Tensor(np.array([1, 2, 3]))
        comp2.rotation(1, 2, 3)
        comp2.matrix = comp2.rot
        comp2.multirotation(self.phi, self.theta)

        np.testing.assert_allclose(comp1.multirot, self.ten1.multirot, atol=1e-7)
        np.testing.assert_allclose(comp2.multirot, self.ten2.multirot, atol=1e-7)


class TestCreateZfsTensorDiagonals:
    def setup_method(self):
        D = 3
        E = 1
        self.a = -1 / 3 * D + E
        self.b = -1 / 3 * D - E
        self.c = 2 / 3 * D
        self.dig = cr.create_zfs_tensor_diagonals(D, E)

    def test_type(self):
        assert self.dig.dtype == "float32"

    def test_values(self):
        comp = np.array([self.a, self.b, self.c])
        assert np.array_equal(comp, self.dig)


class TestCreateDipolTensorDiagonals:
    def setup_method(self):
        D = 3
        E = 1
        self.a = D + E
        self.b = D - E
        self.c = -2 * D
        self.dig = cr.create_dipol_tensor_diagonals(D, E)

    def test_type(self):
        assert self.dig.dtype == "float32"

    def test_values(self):
        comp = np.array([self.a, self.b, self.c])
        assert np.array_equal(comp, self.dig)


class TestSetUpTensorsDoublet:
    def setup_method(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()
        self.cal = Cal()

        self.sys.g = [1, 2, 3]
        self.sys.g_frame = [0, 0, 0]
        self.opt.grid_points = 3
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(3)

    def test_g(self):
        g_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tensor.multirot, self.cal.g_tensor.multirot)

    def test_g_with_frame(self):
        self.sys.g_frame = [1, 2, 3]
        self.opt.grid_points = 8
        g_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tensor.multirot, self.cal.g_tensor.multirot)

    def test_g_iso(self):
        g_iso = 1 / 3 * np.sum(self.sys.g)
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class TestSetUpTensorsRp:
    def setup_method(self):
        self.opt = Opt()
        self.sys = Sys()
        self.cal = Cal()

        self.opt.grid_points = 20
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(20)

        self.sys.g1 = [1, 2, 3]
        self.sys.g2 = [2, 3, 4]
        self.sys.g2_frame = [0.5, 1, 1.5]
        self.sys.g_tri = [4, 5, 6]
        self.sys.precursor = "singlet"

    def test_g1(self):
        g1_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g1_tensor.multirot, self.cal.g1_tensor.multirot)

    def test_g2(self):
        g2_tensor = cr.create_tensor(
            [2, 3, 4], self.cal.phi, self.cal.theta, [0.5, 1, 1.5]
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g2_tensor.multirot, self.cal.g2_tensor.multirot)

    def test_no_Ds(self):
        cr.set_up_tensors(self.sys, self.cal)
        D_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(0, 0.01), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_D_1(self):
        self.sys.D = 5
        D_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 0), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_2(self):
        self.sys.E = 5
        D_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(0, 5), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_3(self):
        self.sys.D = 5
        self.sys.E = 1
        D_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_4(self):
        self.sys.D = 5
        self.sys.E = 1
        self.sys.D_frame = [1, 2, 3]
        D_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1),
            self.cal.phi,
            self.cal.theta,
            [1, 2, 3],
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_gT_1(self):
        self.sys.precursor = "triplet"
        gT_tensor = cr.create_tensor([4, 5, 6], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(gT_tensor.multirot, self.cal.g_tri_tensor.multirot)

    def test_gT_2(self):
        self.sys.precursor = "triplet"
        self.sys.g_tri_frame = [1, 2, 3]
        gT_tensor = cr.create_tensor([4, 5, 6], self.cal.phi, self.cal.theta, [1, 2, 3])
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(gT_tensor.multirot, self.cal.g_tri_tensor.multirot)

    def test_DT_1(self):
        self.sys.precursor = "triplet"
        self.sys.D_tri = 5
        DT_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 0.01), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_DT_2(self):
        self.sys.precursor = "triplet"
        self.sys.E_tri = 5
        DT_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(0, 5), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_DT_3(self):
        self.sys.precursor = "triplet"
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        DT_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_DT_4(self):
        self.sys.precursor = "triplet"
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        self.sys.D_tri_frame = [1, 2, 3]
        DT_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 1),
            self.cal.phi,
            self.cal.theta,
            [1, 2, 3],
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(DT_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_g_iso(self):
        g_iso = 0.5 * (1 / 3 * np.sum(self.sys.g1) + 1 / 3 * np.sum(self.sys.g2))
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class TestSetUpTensorsTriplet:
    def setup_method(self):
        self.opt = Opt()
        self.sys = Sys()
        self.cal = Cal()

        self.opt.grid_points = 20
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(20)
        self.sys.g_tri = [4, 5, 6]

    def test_g(self):
        g_tri_tensor = cr.create_tensor([4, 5, 6], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tri_tensor.multirot, self.cal.g_tri_tensor.multirot)

    def test_g_with_frame(self):
        self.sys.g_tri_frame = [1, 2, 3]
        g_tri_tensor = cr.create_tensor(
            [4, 5, 6], self.cal.phi, self.cal.theta, [1, 2, 3]
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tri_tensor.multirot, self.cal.g_tri_tensor.multirot)

    def test_dipole_tensor_only_D(self):
        self.sys.D_tri = 5
        D_tri_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 0.01), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_only_E(self):
        self.sys.E_tri = 5
        D_tri_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(0, 5), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_D_and_E(self):
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        D_tri_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_with_frame(self):
        self.sys.D_tri = 5
        self.sys.E_tri = 1
        self.sys.D_tri_frame = [1, 2, 3]
        D_tri_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 1),
            self.cal.phi,
            self.cal.theta,
            [1, 2, 3],
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_dipole_tensor_no_D_and_E(self):
        D_tri_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(0, 0.01), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_g_iso(self):
        g_iso = 1 / 3 * (np.sum(self.sys.g_tri))
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class TestSetUpTensorsTdp:
    def setup_method(self):
        self.opt = Opt()
        self.sys = Sys()
        self.cal = Cal()

        self.opt.grid_points = 20
        self.cal.theta, self.cal.phi = grid.fibonacci_grid(20)
        self.sys.g = [1, 2, 3]
        self.sys.g_tri = [4, 5, 6]
        self.sys.D_tri = 5
        self.sys.E_tri = 1

    def test_g(self):
        g_tensor = cr.create_tensor([1, 2, 3], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tensor.multirot, self.cal.g_tensor.multirot)

    def test_g_tri(self):
        g_tri_tensor = cr.create_tensor([4, 5, 6], self.cal.phi, self.cal.theta)
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(g_tri_tensor.multirot, self.cal.g_tri_tensor.multirot)

    def test_D(self):
        self.sys.D = 5
        self.sys.E = 1
        D_tensor = cr.create_tensor(
            cr.create_dipol_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tensor.multirot, self.cal.D_tensor.multirot)

    def test_D_tri(self):
        D_tri_tensor = cr.create_tensor(
            cr.create_zfs_tensor_diagonals(5, 1), self.cal.phi, self.cal.theta
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert np.array_equal(D_tri_tensor.multirot, self.cal.D_tri_tensor.multirot)

    def test_g_iso(self):
        g_iso = (
            1 / 2 * (1 / 3 * (np.sum(self.sys.g_tri)) + 1 / 3 * (np.sum(self.sys.g)))
        )
        cr.set_up_tensors(self.sys, self.cal)
        assert g_iso == self.cal.g_iso


class TestSetUpSpinoperator:
    def setup_method(self):
        self.cal = Cal()
        self.sys = Sys()

    def test_s_coupled(self):
        # radical pair
        s1 = np.array(
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.5j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.5j],
                    [0.0 + 0.5j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.0j, -0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j, -0.5 + 0.0j],
                ],
            ]
        )

        s2 = np.array(
            [
                [
                    [0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.5j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.5j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.5j, 0.0 + 0.0j],
                ],
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, -0.5 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, -0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.0j],
                ],
            ]
        )
        S = s1 + s2
        self.sys.s = [1 / 2, 1 / 2]
        cr.set_up_spinoperator(self.sys, self.cal)
        assert np.array_equal(self.cal.s.matrix, S)

    def test_dtype_coupled(self):
        self.sys.s = [1 / 2, 1 / 2]
        cr.set_up_spinoperator(self.sys, self.cal)
        assert self.cal.s.matrix.dtype == "complex64"

    def test_s_uncoupled(self):
        # Doublet
        s = np.array(
            [
                [[0.0 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.0 + 0.0j]],
                [[0.0 - 0.0j, 0.0 - 0.5j], [0.0 + 0.5j, 0.0 - 0.0j]],
                [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.5 + 0.0j]],
            ]
        )

        self.sys.s = 1 / 2
        cr.set_up_spinoperator(self.sys, self.cal)
        assert np.array_equal(self.cal.s.matrix, s)

    def test_dtype_uncoupled(self):
        self.sys.s = [1 / 2]
        cr.set_up_spinoperator(self.sys, self.cal)
        assert self.cal.s.matrix.dtype == "complex64"


class TestSetUpObservable:
    def setup_method(self):
        self.opt = Opt()
        self.cal = Cal()
        self.sys = Sys()

        self.sys.spin_system = "rp"
        self.sys.s = [1 / 2, 1 / 2]
        self.opt.space = "hilbert"
        cr.set_up_spinoperator(self.sys, self.cal)

    def test_hilbert_observable_rp(self):
        sig = np.array(
            [
                [0.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.70710678j, 0.0 - 0.0j],
                [0.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.0j],
                [0.0 + 0.70710678j, 0.0 - 0.0j, 0.0 - 0.0j, 0.0 - 0.70710678j],
                [0.0 - 0.0j, 0.0 - 0.0j, 0.0 + 0.70710678j, 0.0 - 0.0j],
            ]
        )
        cr.set_up_observable(self.sys, self.opt, self.cal)
        np.testing.assert_allclose(sig, self.cal.observable)
        assert self.cal.observable.dtype == "complex64"

    def test_liouville_observable_rp(self):
        sig = np.array(
            [
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 - 0.70710678j,
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 + 0.70710678j,
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 - 0.70710678j,
                0.0 - 0.0j,
                0.0 - 0.0j,
                0.0 + 0.70710678j,
                0.0 - 0.0j,
            ]
        )
        self.opt.space = "liouville"
        cr.set_up_observable(self.sys, self.opt, self.cal)
        np.testing.assert_allclose(sig, self.cal.observable)
        assert self.cal.observable.dtype == "complex64"

    def test_hilbert_observable_doub(self):
        self.sys.spin_system = "doub"
        self.sys.s = [1 / 2]
        cr.set_up_spinoperator(self.sys, self.cal)
        cr.set_up_observable(self.sys, self.opt, self.cal)
        sig = self.cal.s.get("y")
        assert np.array_equal(sig, self.cal.observable)
        assert self.cal.observable.dtype == "complex64"

    def test_liouville_observable_trip(self):
        self.sys.spin_system = "trip"
        self.sys.s = [1]
        self.opt.space = "liouville"
        cr.set_up_spinoperator(self.sys, self.cal)
        cr.set_up_observable(self.sys, self.opt, self.cal)
        sig = self.cal.s.get("y")
        sig = sig.flatten()
        assert np.array_equal(sig, self.cal.observable)
        assert self.cal.observable.dtype == "complex64"
