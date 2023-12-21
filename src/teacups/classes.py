import numpy as np


class Sys:
    def __init__(self):
        self.spin_system = "doub"
        self.precursor = "basis"
        self.population = [0, 1]

        self.g = [1.95, 2, 2.1]

        self.width_gauss = 3
        self.decay = 1e-6

        self.dynamics = None
        self.T_relax_1 = 1e-6
        self.T_relax_2 = 1e-6


class Exp:
    def __init__(self):
        self.B_z = np.linspace(320, 380, 600)
        self.t_scale = [0, 2e-6]
        self.t_points = 60
        self.B_mw = 0.01
        self.freq_mw = 9.75e9


class SimOpt:
    def __init__(self):
        self.grid_points = 10
        self.grid = "fibonacci"
        self.space = "hilbert"
        self.pop_evolution = False
        self.eigval_mode = False
        self.cpu_cores = 0
