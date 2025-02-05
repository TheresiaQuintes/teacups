import teacups.simulations as sim
import teacups.classes as cl
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
Sys.D_tri = 700

# define couplings of the radical and the triplet
Sys.J_ex = -20000

# define initial state
Sys.precursor = "eigen"
Sys.population = [0.3, 0.23, 0.2, 0.3, 0.5, 0.49]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(320, 380, 700)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 7
SimOpt.space = "hilbert"

# do the simulation
spec = sim.teacups(Sys, Exp, SimOpt)
plt.plot(Exp.B_z, spec.real[1])

SimOpt.eigval_mode = True
eigvals = sim.teacups(Sys, Exp, SimOpt)
plt.figure()
plt.plot(Exp.B_z, eigvals[:, 0])
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("E / Hz")
plt.savefig("./../_images/tutorial_teacups_2.png")

# Define relaxation operator for RQM, Rateconstants in 1/s
k_d = 0.25 / 1e-6

R = np.zeros((6, 6))
R[5, 5] = -k_d
R[4, 4] = -k_d
R[5, 2] = k_d
R[2, 5] = k_d

Sys.dynamics = R

# Change simulationoptions
SimOpt.eigval_mode = False
SimOpt.space = "liouville"
SimOpt.pop_evolution = True
Exp.t_points = 60

# Run the simulation
spec, pop_evolution = sim.teacups(Sys, Exp, SimOpt)

t = np.linspace(Exp.t_scale[0], Exp.t_scale[1], Exp.t_points)*1e6
# %% Plot population evolution
plt.figure()
plt.xlabel("$t / \mu\mathrm{s}$")
plt.ylabel("population of states")
plt.plot(t, np.array(pop_evolution).real)
plt.savefig("./../_images/tutorial_teacups_3.png")

# %% Plot single 2D
plt.figure()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.plot(Exp.B_z, spec[30].real)
plt.yticks([])
plt.savefig("./../_images/tutorial_teacups_1.png")

# %% Plot multiple 2D
plt.figure()
plt.xlabel("$B_z / \mathrm{mT}$")
for i in range(14, 30, 5):
    plt.plot(Exp.B_z, spec[i].real, label=str(np.round(t[i], 2))+" $\mu$s")

plt.yticks([])
plt.legend()
plt.savefig("./../_images/tutorial_teacups_4.png")

# %% Plot surface
plt.figure()
ax = plt.axes()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("$t / \mu\mathrm{s}$")
x, y = np.meshgrid(Exp.B_z, t)
ax.pcolormesh(x, y, spec.real, cmap="RdBu", shading="auto")
plt.savefig("./../_images/tutorial_teacups_5.png")
