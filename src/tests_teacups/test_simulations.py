import scipy.signal as ssg
import numpy as np
import tests_teacups.simulation_comparison_arrays as comp
import teacups.simulations as sim
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


class Test_teacups:

    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.spin_system = 'rp'
        self.sys.g1 = [2.00110, 2.00110, 2.00110]
        self.sys.g1_frame = [0, 0, 0]
        self.sys.g2 = [2.00060, 2.00060, 2.00060]
        self.sys.g2_frame = [0, 0, 0]

        self.sys.g_tri = [2.00431, 2.00431, 2.00431]
        self.sys.population = [0.25, 0.75, 0]
        self.sys.D_tri = 2802.5
        self.sys.E_tri = 0

        self.sys.T_relax_1 = 1
        self.sys.T_relax_2 = 1
        self.sys.dynamics = None

        self.sys.D = 0
        self.sys.E = 0
        self.sys.J_ex = 2

        self.sys.width_gauss = 0.

        self.exp.B_z = np.linspace(345.2, 346.2, 50)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 2
        self.exp.B_mw = 0.001
        self.exp.freq_mw = 9.68e9

        self.opt.grid_points = 3
        self.opt.grid = 'fibonacci'
        self.opt.space = 'liouville'
        self.opt.mode = 'fitting'
        self.opt.pop_evolution = False
        self.opt.eigval_mode = False
        self.opt.cpu_cores = 1

    def test_isotrope_singlet(self):
        self.sys.precursor = 'singlet'
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.isotrope_singlet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_isotrope_triplet(self):
        self.sys.precursor = 'triplet-zf'
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.isotrope_triplet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_anisotrope_singlet(self):
        self.sys.precursor = 'singlet'
        self.sys.g2 = [2.00060, 2.00160, 2.00260]
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.anisotrope_singlet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_anisotrope_triplet(self):
        self.sys.precursor = 'triplet-zf'
        self.sys.g2 = [2.00060, 2.00160, 2.00260]
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.anisotrope_triplet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_changed_J_triplet(self):
        self.sys.precursor = 'triplet-zf'
        self.sys.J_ex = 4
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.changed_J_triplet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_D_singlet(self):
        self.sys.precursor = 'singlet'
        self.sys.D = 2
        self.sys.E = 1
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.D_singlet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_D_triplet(self):
        self.sys.precursor = 'triplet-zf'
        self.sys.D = 2
        self.sys.E = 1
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.D_triplet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_lw_singlet(self):
        self.sys.precursor = 'singlet'
        self.sys.width_gauss = 0.05
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.lw_singlet, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_anisotrope_triplet_precursor(self):
        self.sys.precursor = 'triplet-zf'
        self.sys.g_tri = [1, 2, 3]
        cal = sim.teacups(self.sys, self.exp, self.opt,
                          development=True)
        np.testing.assert_allclose(
            comp.anisotrope_triplet_precursor, cal.spec_sim[1]/max(abs(cal.spec_sim[1])), atol=1e-5, rtol=1e-6)

    def test_no_development_mode(self):
        self.sys.precursor = 'singlet'
        signal = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(
            comp.isotrope_singlet, signal[1]/max(abs(signal[1])), atol=1e-5, rtol=1e-6)


class Test_treper_all_on:

    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.spin_system = 'rp'
        self.sys.g1 = [2.00431, 2.00360, 2.00217]
        self.sys.g1_frame = [0, 0, 0]
        self.sys.g2 = [2.00370, 2.00285, 2.00246]
        self.sys.g2_frame = [2.21656815, 1.34390352, 4.31096325]

        self.sys.precursor = 'singlet'
        self.sys.g_tri = [2.00370, 2.00285, 2.00246]
        self.sys.population = [0.67, 0.33, 0]
        self.sys.D_tri = 1.9217e+03
        self.sys.E_tri = -525.4678

        self.sys.T_relax_1 = 1e-6
        self.sys.T_relax_2 = 1e-6
        self.sys.decay = 1e-6
        self.sys.dynamics = None

        self.sys.D = -10.0890
        self.sys.D_frame = [0, 1.9198621771937625, 1.9198621771937625]
        self.sys.E = 0
        self.sys.J_ex = 2.0458

        self.sys.width_gauss = 0.371

        self.exp.B_z = np.linspace(342, 348, 150)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 4
        self.exp.B_mw = 0.001
        self.exp.freq_mw = 9.68e9

        self.opt.grid_points = 7
        self.opt.grid = 'fibonacci'
        self.opt.space = 'hilbert'
        self.opt.mode = 'fitting'
        self.opt.pop_evolution = False
        self.opt.eigval_mode = False
        self.opt.cpu_cores = 2

    def test_hilbert_analytical_singlet(self):
        self.cal = sim.teacups(self.sys, self.exp, self.opt,
                               development=True)

        np.testing.assert_allclose(
            self.cal.spec_sim/(abs(self.cal.spec_sim.max())), comp.a_hilbert_analytical_singlet, atol=1e-5, rtol=1e-6)

    def test_liouville_analytical_singlet(self):
        self.opt.space = 'liouville'
        self.cal = sim.teacups(self.sys, self.exp, self.opt,
                               development=True)

        np.testing.assert_allclose(
            comp.a_liouville_analytical_singlet, self.cal.spec_sim/(abs(self.cal.spec_sim.max())), atol=1e-5, rtol=1e-6)

    def test_hilbert_analytical_triplet(self):
        self.sys.precursor = 'triplet-zf'
        self.cal = sim.teacups(self.sys, self.exp, self.opt,
                               development=True)
        np.testing.assert_allclose(comp.a_hilbert_analytical_triplet,
                                   self.cal.spec_sim/(abs(self.cal.spec_sim.max())),  atol=1e-5, rtol=1e-6)

    def test_liouville_analytical_triplet(self):
        self.sys.precursor = 'triplet-zf'
        self.opt.space = 'liouville'
        self.cal = sim.teacups(self.sys, self.exp, self.opt,
                               development=True)
        np.testing.assert_allclose(
            comp.a_liouville_analytical_triplet, self.cal.spec_sim/(abs(self.cal.spec_sim.max())),  atol=1e-5, rtol=1e-6)


class Test_teacups_triplet:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.spin_system = 'trip'
        self.sys.precursor = 'zf'
        self.sys.width_gauss = 0
        self.sys.D_tri = 700
        self.sys.E_tri = -200
        self.sys.g_tri = [2, 2, 2]
        self.sys.decay = 1e-6
        self.sys.population = [1, 0, 0]

        self.exp.B_z = np.linspace(300, 400, 20)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 3
        self.exp.B_mw = 0.1
        self.exp.freq_mw = 9.6e9

        self.opt.grid_points = 15
        self.opt.grid = 'fibonacci'
        self.opt.space = 'hilbert'
        self.opt.mode = 'fitting'
        self.opt.eigval_mode = False
        self.opt.pop_evolution = False
        self.opt.cpu_cores = 1

    def test_value_of_spectrum(self):
        spec = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(
            spec[1]/max(abs(spec[1].real)), comp.triplet_simulation_values, atol=1e-6)


class Test_teacups_triplet_liouville:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.g_tri = [2, 2, 2]
        self.sys.g_tri_frame = [0, 0, 0]
        self.sys.decay = 1e-6
        self.sys.T_relax_1 = 1e-6
        self.sys.T_relax_2 = 1e-6
        self.sys.dynamics = None
        self.sys.width_gauss = 4
        self.sys.population = [1, 0, 0]
        self.sys.D_tri = 700
        self.sys.E_tri = 0

        self.sys.spin_system = 'trip'
        self.sys.precursor = 'zf'

        self.exp.B_z = np.linspace(275, 400, 300)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 2
        self.exp.B_mw = 0.01
        self.exp.freq_mw = 9.68e9

        self.opt.grid_points = 1
        self.opt.grid = 'fibonacci'
        self.opt.space = 'liouville'
        self.opt.mode = 'fitting'
        self.opt.hyperfine_mode = 'off'
        self.opt.pop_evolution = False
        self.opt.eigval_mode = False
        self.opt.cpu_cores = 0

    def test_number_of_peaks_liouville(self):
        spec = sim.teacups(self.sys, self.exp, self.opt)

        peaks = ssg.find_peaks(np.abs(spec[1]), 1e-6)
        peaks = peaks[0]

        assert len(peaks) == 2
        assert np.array_equal(peaks, np.array([163, 177]))

    def test_number_of_peaks_hilbert(self):
        self.opt.space = 'hilbert'
        self.sys.width_gauss = 2
        spec = sim.teacups(self.sys, self.exp, self.opt)

        peaks = ssg.find_peaks(np.abs(spec[1]), 5e-7)
        peaks = peaks[0]

        assert len(peaks) == 2
        assert np.array_equal(peaks, np.array([162, 177]))


class Test_teacups_doublet:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.spin_system = 'doub'
        self.sys.precursor = 'eigen'
        self.sys.width_gauss = 0
        self.sys.g = [1.9, 2, 2.3]
        self.sys.decay = 1e-6
        self.sys.dynamics = None
        self.sys.population = [0.6, 0.4]

        self.exp.B_z = np.linspace(290, 380, 40)
        self.exp.t_scale = [0, 2e-6]
        self.exp.t_points = 3
        self.exp.B_mw = 0.001
        self.exp.freq_mw = 9.68e9

        self.opt.grid_points = 15
        self.opt.grid = 'fibonacci'
        self.opt.space = 'hilbert'
        self.opt.mode = 'fitting'
        self.opt.hyperfine_mode = 'off'
        self.opt.eigval_mode = False
        self.opt.pop_evolution = False
        self.opt.cpu_cores = 1

    def test_value_of_spectrum(self):
        spec = sim.teacups(self.sys, self.exp, self.opt)
        np.testing.assert_allclose(
            spec[1]/max(abs(spec[1])), comp.doublet_simulation_values, rtol=4e-6)


class Test_doublet_liouville:
    def setup(self):
        self.sys = Sys()
        self.exp = Exp()
        self.opt = Opt()

        self.sys.g = [2, 2, 2]
        self.sys.g_frame = [0, 0, 0]
        self.sys.decay = 5e-6
        self.sys.T_relax_1 = 5e-6
        self.sys.T_relax_2 = 5e-6
        self.sys.dynamics = None
        self.sys.width_gauss = 4
        self.sys.population = [1, 0]
        self.sys.spin_system = 'doub'
        self.sys.precursor = 'eigen'

        self.exp.B_z = np.linspace(300, 390, 300)
        self.exp.t_scale = [0, 1.5e-6]
        self.exp.t_points = 500
        self.exp.B_mw = 0.01
        self.exp.freq_mw = 9.68e9

        self.opt.grid_points = 1
        self.opt.grid = 'fibonacci'
        self.opt.space = 'hilbert'
        self.opt.mode = 'fitting'
        self.opt.hyperfine_mode = 'off'
        self.opt.pop_evolution = False
        self.opt.eigval_mode = False
        self.opt.cpu_cores = 1

    def test_compare_maxima(self):
        cal_h = sim.teacups(self.sys, self.exp, self.opt, development=True)
        self.opt.space = 'liouville'
        cal_l = sim.teacups(self.sys, self.exp, self.opt, development=True)

        max_h = np.argmax(cal_h.spec_sim.real, axis=1)
        max_l = np.argmax(cal_l.spec_sim.real, axis=1)

        assert np.all((max_h[1:] >= 149) & (max_h[1:] <= 152))
        assert np.all((max_l[1:] >= 149) & (max_l[1:] <= 152))
