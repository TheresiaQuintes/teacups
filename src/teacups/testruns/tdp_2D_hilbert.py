import teacups.classes as cl
import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("stylesheet.mplstyle")

Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# choose a triplet doublet pair spin system
Sys.spin_system = "tdp"

# set up g-tensors of the radical and the triplet
Sys.g = [2.0059, 2.0059, 2.0059]
Sys.g_tri = [2.003, 2.003, 2.003]

# define triplet ZFS
Sys.D_tri = -500
Sys.E_tri = 50

# define couplings of the radical and the triplet
Sys.J_ex = -400000

Sys.precursor = "triplet-zf"
Sys.population = [0.2, 0.205, 0.2, 0.1, 0.1]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(333, 363, 550)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid = "sophe"
SimOpt.sym = "D2h"
SimOpt.grid_points = 25
SimOpt.CUPY = False


# do the simulation
spec_sim = sim.teacups(Sys, Exp, SimOpt)
spec_sim = spec_sim.real[1] / max(abs(spec_sim.real[1]))

plt.figure()
plt.plot(Exp.B_z, spec_sim)
plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")

# plt.savefig("tdp_2D_hilbert.pdf")
