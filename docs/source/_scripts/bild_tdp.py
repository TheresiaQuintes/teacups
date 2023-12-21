import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt


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
Sys.J_ex = -200000

# define initial state
Sys.precursor = "triplet-zf"
Sys.population = [0.205, 0.2, 0.3, 0.2, 0.1]

# define relaxation times and gaussian line width
Sys.T_relax_1 = 1e-6
Sys.T_relax_2 = 1e-7
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(320, 380, 700)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 50
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 5
SimOpt.space = "liouville"

# do the simulation
spec = sim.teacups(Sys, Exp, SimOpt)

# %% Plot 2D
plt.figure()
plt.plot(Exp.B_z, spec[10].real/max(abs(spec[10].real)))

# %% Plot 3D
plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(Exp.B_z, np.linspace(0, 2e-6, Exp.t_points))
ax.plot_surface(x, y, spec.real, cmap='coolwarm')

# %% Plot surface
fig, ax = plt.subplots()
ax.pcolormesh(x, y, spec.real, cmap='coolwarm', linewidth=0, antialiased=False)
plt.xlabel('$B_z$ / mT')
plt.ylabel('$t$ / s')
plt.show()
