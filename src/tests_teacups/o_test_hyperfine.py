import numpy as np
import teacups.hyperfine as hf
import teacups.matrix_tools as mt
import teacups.simulations as sim
import hyperfine_comparison_arrays as hca


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


class Test_set_up_hyperfine_tensors:
    def setup(self):
        self.sys = Sys()
        self.opt = Opt()
        self.cal = Cal()
        self.sys.A = [[[1, 2, 3]], [[1, 1, 1]]]
        self.opt.grid_points = 3

        hf.set_up_hyperfine_tensors(self.sys, self.opt, self.cal)

    def test_list_len(self):
        assert len(self.cal.A_tensor) == 2

    def test_type_list(self):
        assert type(self.cal.A_tensor) == list

    def test_type_list_element(self):
        assert isinstance(self.cal.A_tensor[0], list)

    def test_type_list_list_element(self):
        assert isinstance(self.cal.A_tensor[0][0], mt.Tensor)

    def test_dimension(self):
        assert self.cal.A_tensor[0][0].multirot.shape == (3, 3, 3)

    def test_first_rotation(self):
        without = self.cal.A_tensor[0][0]
        self.sys.A_Frame = [[[1, 1, 1]], [[1, 1, 1]]]
        hf.set_up_hyperfine_tensors(self.sys, self.opt, self.cal)
        assert without != self.cal.A_tensor[0][0]

    def test_one_spin(self):
        self.sys.A = [[[1, 1, 1]]]
        hf.set_up_hyperfine_tensors(self.sys, self.opt, self.cal)
        assert isinstance(self.cal.A_tensor[0][0], mt.Tensor)


class Test_create_hf_hamiltonian:
    def setup(self):
        self.A_iso_2 = mt.Tensor([2, 2, 2])
        self.A_iso_2.multirotation(1)
        self.A_iso_4 = mt.Tensor([4, 4, 4])
        self.A_iso_4.multirotation(1)
        self.A_multiple_orientations = mt.Tensor([1, 2, 3])
        self.A_multiple_orientations.multirotation(3)

        self.ham_multiple_orientations = hf.create_hf_hamiltonian(
            [1/2], [[1/2]], [[self.A_multiple_orientations]])
        self.ham_1_1 = hf.create_hf_hamiltonian([1/2], [[1/2]],
                                                [[self.A_iso_2]])
        self.ham_1_2 = hf.create_hf_hamiltonian([1/2], [[1/2, 1]],
                                                [[self.A_iso_4, self.A_iso_2]])
        self.ham_2_1_A = hf.create_hf_hamiltonian([1/2, 1/2], [[1/2], []],
                                                  [[self.A_iso_4], []])
        self.ham_2_1_B = hf.create_hf_hamiltonian([1/2, 1], [[], [1/2]],
                                                  [[], [self.A_iso_2]])

    def test_multiple_orientations_shape(self):
        assert self.ham_multiple_orientations[0].shape == (3, 4)

    def test_multiple_orientations_value_1(self):
        assert self.ham_multiple_orientations[0][1, 0] ==\
            self.A_multiple_orientations.multirot[1, 2, 2]*1/4

    def test_multiple_orientations_value_2(self):
        assert self.ham_multiple_orientations[0][2, 0] ==\
            self.A_multiple_orientations.multirot[2, 2, 2]*1/4

    def test_1_1_shape(self):
        assert len(self.ham_1_1[0][0]) == 4

    def test_1_1_values(self):
        np.testing.assert_allclose(np.array([1/2, -1/2, -1/2, 1/2]),
                                   self.ham_1_1[0][0])

    def test_1_2_shape(self):
        assert len(self.ham_1_2[0][0]) == 12

    def test_1_2_value(self):
        np.testing.assert_allclose(np.array([2, 1, 0, 0, -1, -2, -2, -1, 0, 0, 1, 2]),
                                   self.ham_1_2[0][0])

    def test_2_1_A_shapes(self):
        assert len(self.ham_2_1_A) == 2
        assert self.ham_2_1_A[0].shape == (1, 4)
        assert self.ham_2_1_A[1].shape == (1, 2)

    def test_2_1_A_value_hamA(self):
        np.testing.assert_allclose(
            np.array([1, -1, -1, 1]), self.ham_2_1_A[0][0])

    def test_2_1_A_value_hamB(self):
        np.testing.assert_allclose(np.array([0, 0]), self.ham_2_1_A[1][0])

    def test_2_1_B_shapes(self):
        assert len(self.ham_2_1_B) == 2
        assert self.ham_2_1_B[0].shape == (1, 2)
        assert self.ham_2_1_B[1].shape == (1, 6)

    def test_2_1_B_value_hamA(self):
        np.testing.assert_allclose(np.array([0, 0]), self.ham_2_1_B[0][0])

    def test_2_1_B_value_hamB(self):
        np.testing.assert_allclose(np.array([1, -1, 0, 0, -1, 1]),
                                   self.ham_2_1_B[1][0])


class Test_hyperfine_of_coupled_system:
    def setup(self):
        self.A = mt.Tensor([1, 1, 1])
        self.A.multirotation(1)
        self.ham = hf.create_hf_hamiltonian([1/2, 1], [[1/2, 1/2], []],
                                            [[self.A, self.A], []])
        self.ham_coup = hf.hyperfine_of_coupled_system(self.ham, [4, 1],
                                                       [2, 3], 6)

    def test_hyperfine_coupled_first_spin(self):
        first_spin = np.array(
            [[[0.5+0.j,  0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j]],
             [[0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]],
             [[0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]],
             [[-0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j]]])

        assert np.array_equal(first_spin, self.ham_coup[0])

    def test_hyperfine_coupled_second_spin(self):
        second_spin = np.array(
            [[[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]])

        assert np.array_equal(second_spin, self.ham_coup[1])

    def test_no_first_coupling(self):
        ham = hf.create_hf_hamiltonian([1/2, 1], [[], [1/2]], [[], [self.A]])
        ham_coup = hf.hyperfine_of_coupled_system(ham, [1, 2], [2, 3], 6)

        assert len(ham_coup[0]) == 1
        assert len(ham_coup[1]) == 2


class Test_make_signal_with_hyperfine_two_spins:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.g1 = [2.003, 2.003, 2.003]
        self.sys.g1_frame = [0, 0, 0]
        self.sys.g2 = [2.00, 2.00, 2.00]
        self.sys.decay = 1e-6
        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 1
        self.sys.width_gauss = 0.04
        self.sys.spin_system = 'rp'
        self.sys.precursor = 'singlet'

        # set up experimental parameters
        self.exp.B_z = np.linspace(342, 344, 200)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 2
        self.exp.B_mw = 0.01
        self.exp.freq_mw = 9.602e9
        self.exp.B_z_rp = np.linspace(342, 344, 300)
        self.exp.B_z_tri = np.linspace(300, 400, 800)
        self.exp.temperature = 0.1

        # set up simulation options
        self.opt.grid_points = 2
        self.opt.space = 'hilbert'
        self.opt.mode = 'fitting'
        self.opt.eigval_mode = False

    def test_no_hyperfines_given(self):
        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.no_hyperfines)

    def test_no_nuclei_given(self):
        self.sys.A = [[[4, 4, 4]], []]
        self.sys.I = [[], []]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.no_hyperfines)

    def test_hf_1_0(self):
        self.sys.A = [[[4, 4, 4]], []]
        self.sys.I = [[1], []]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.hf_1_0, atol=1e-7, rtol=2e-7)

    def test_hf_0_2(self):
        self.sys.A = [[], [[4, 4, 4], [4, 4, 4]]]
        self.sys.I = [[], [1/2, 1/2]]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.hf_0_2, atol=1e-7, rtol=2e-7)

    def test_hf_1_1(self):
        self.sys.A = [[[4, 4, 4]], [[4, 4, 4]]]
        self.sys.I = [[1/2], [1/2]]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.hf_1_1, atol=1e-7, rtol=2e-7)


class Test_make_signal_with_hyperfine_one_spin:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.g = [2, 2, 2]
        self.sys.g_frame = [0, 0, 0]
        self.sys.decay = 1e-6
        self.sys.width_gauss = 3
        self.sys.population = [0, 1]
        self.sys.spin_system = 'doub'
        self.sys.precursor = 'basis'

        self.exp.B_z = np.linspace(320, 380, 600)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 2
        self.exp.B_mw = 0.01
        self.exp.freq_mw = 9.68e9

        self.opt.grid_points = 2
        self.opt.space = 'hilbert'
        self.opt.mode = 'fitting'
        self.opt.eigval_mode = False

    def test_no_hyperfines_given(self):
        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.no_hyperfines_doub)

    def test_no_nuclei_given(self):
        self.sys.A = [[[4, 4, 4]]]
        self.sys.I = [[]]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.no_hyperfines_doub)

    def test_hf_1(self):
        self.sys.A = [[[250, 250, 250], [250, 250, 250]]]
        self.sys.I = [[1]]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.hf_1)

    def test_hf_2(self):
        self.sys.A = [[[250, 250, 250], [250, 250, 250]]]
        self.sys.I = [[1/2, 1/2]]

        spc = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(spc[1], hca.hf_2)
