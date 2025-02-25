import teacups.classes as cl
import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("stylesheet.mplstyle")

Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# choose a triplet spin system
Sys.spin_system = "trip"

# set up g-tensors of the triplet
Sys.g_tri = [2.003, 2.003, 2.003]

# define triplet ZFS
Sys.D_tri = 700
Sys.E_tri = -100

# define initial state
Sys.precursor = "zf"
Sys.population = [1, 0, 0]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(300, 400, 1024)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 40

# do the simulation
spec_sim = sim.teacups(Sys, Exp, SimOpt)
spec_sim = spec_sim.real[1]/max(abs(spec_sim.real[1]))



plt.figure()
plt.plot(Exp.B_z, spec_sim)

plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")

# plt.savefig("triplet_2D_hilbert.pdf")
