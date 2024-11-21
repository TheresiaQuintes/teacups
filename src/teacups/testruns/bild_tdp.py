import sys
sys.path.append("../../")


import matplotlib.pyplot as plt
import teacups.classes as cl
import numpy as np
import teacups.simulations as sim


sys = cl.Sys()
exp = cl.Exp()
opt = cl.SimOpt()

sys.spin_system = "tdp"
sys.precursor = "eigen"
sys.population = [0.3, 0.23, 0.2, 0.3, 0.5, 0.48]

sys.g = [2.0059, 2.0059, 2.0059]

sys.g_tri = [2.003, 2.003, 2.003]
sys.D_tri = 700

sys.J_ex = -20000

sys.width_gauss = 1


exp.B_z = np.linspace(320, 380, 700)
exp.t_scale = [0, 2e-6]
exp.t_points = 60
exp.B_mw = 0.001
exp.freq_mw = 9.75e9

opt.grid_points = 7
opt.space = "liouville"
opt.pop_evolution = True

# %% Define relaxation operator for RQM
opt.eigval_mode = True
eigenvalues = sim.teacups(sys, exp, opt)
plt.plot(eigenvalues[:, 0])

sys.dynamics = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0., 0, 0, 0]])

eigval = eigenvalues[0, 0]
de_02 = eigval[0] - eigval[2]
de_04 = eigval[0] - eigval[4]
de_05 = eigval[0] - eigval[5]
de_12 = eigval[1] - eigval[2]
de_13 = eigval[1] - eigval[3]
de_15 = eigval[1] - eigval[5]

D_tri = sys.D_tri * 1e6

k_isc = 0.3 / 1e-11
k_e = 0
k_d = 0.25 / 1e-6

K_isc_matrix = np.zeros((6, 6))
K_isc_matrix[0, 2] = k_isc * (1 / 45 * D_tri ** 2) / de_02 ** 2
K_isc_matrix[0, 4] = k_isc * (1 / 135 * D_tri ** 2) / de_04 ** 2
K_isc_matrix[0, 5] = k_isc * (1 / 45 * D_tri ** 2) / de_05 ** 2
K_isc_matrix[1, 2] = k_isc * (1 / 45 * D_tri ** 2) / de_12 ** 2
K_isc_matrix[1, 3] = k_isc * (1 / 135 * D_tri ** 2) / de_13 ** 2
K_isc_matrix[1, 5] = k_isc * (1 / 45 * D_tri ** 2) / de_15 ** 2
K_isc_matrix[2, 0] = k_isc * (1 / 45 * D_tri ** 2) / de_02 ** 2
K_isc_matrix[2, 1] = k_isc * (1 / 45 * D_tri ** 2) / de_12 ** 2
K_isc_matrix[3, 1] = k_isc * (1 / 135 * D_tri ** 2) / de_13 ** 2
K_isc_matrix[4, 0] = k_isc * (1 / 135 * D_tri ** 2) / de_04 ** 2
K_isc_matrix[5, 0] = k_isc * (1 / 45 * D_tri ** 2) / de_05 ** 2
K_isc_matrix[5, 1] = k_isc * (1 / 45 * D_tri ** 2) / de_15 ** 2

K_pop = np.zeros((6, 6))
K_pop[0, 0] = -(k_e + k_d)
K_pop[1, 1] = -(k_e + k_d)
K_pop[2, 2] = -k_e
K_pop[3, 3] = -k_e
K_pop[4, 4] = -k_e
K_pop[5, 5] = -k_e

R = K_isc_matrix + K_pop
sys.dynamics = R[::-1, ::-1]

# %% Do Simulation
opt.eigval_mode = False
spec, population_evolution = sim.teacups(sys, exp, opt)
t = np.linspace(exp.t_scale[0], exp.t_scale[1], exp.t_points)*1e6

# %% Plot population evolution
plt.figure()
plt.xlabel("$t / \mu\mathrm{s}$")
plt.ylabel("population of states")
plt.plot(t, np.array(population_evolution).real)

# %% Plot single 2D
plt.figure()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.plot(exp.B_z, spec[30].real)

# %% Plot multiple 2D
plt.figure()
plt.xlabel("$B_z / \mathrm{mT}$")
for i in range(0, 29, 3):
    plt.plot(exp.B_z, spec[i].real)

# %% Plot surface
plt.figure()
ax = plt.axes()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("$t / \mu\mathrm{s}$")
x, y = np.meshgrid(exp.B_z, t)
ax.pcolormesh(x, y, spec.real, cmap="RdBu", shading="auto")
plt.show()
