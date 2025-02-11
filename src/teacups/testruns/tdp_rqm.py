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
Sys.population = [0.3, 0.225, 0.2, 0.3, 0.5, 0.5]

# define line width
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(320, 380, 700)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 60
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 7
SimOpt.space = "liouville"
SimOpt.pop_evolution = True

# %% set up dynamics-matrix

# determine eigenvalues
SimOpt.eigval_mode = True
eigval = sim.teacups(Sys, Exp, SimOpt)
e = np.mean(eigval[350], axis=0)

# build delta E between trip-doublet and trip-quartet states
de_53 = e[5]-e[3]
de_51 = e[5]-e[1]
de_50 = e[5]-e[0]
de_43 = e[4]-e[3]
de_42 = e[4]-e[2]
de_40 = e[4]-e[0]

# Dipolar coupling squared
D = (Sys.D_tri*1e6)**2

# set isc and doublet decay rates
k_isc = 0.3/1e-11
k_d = 0.25/1e-6

# set up dynamics matrix
R = np.zeros((6, 6))
R[5, 5] = -k_d
R[4, 4] = -k_d

R[5, 3] = k_isc/45*(D/(de_53)**2)
R[5, 1] = k_isc/135*(D/(de_51)**2)
R[5, 0] = k_isc/45*(D/(de_50)**2)
R[3, 5] = k_isc/45*(D/(de_53)**2)
R[1, 5] = k_isc/135*(D/(de_51)**2)
R[0, 5] = k_isc/45*(D/(de_50)**2)

R[4, 3] = k_isc/45*(D/(de_42)**2)
R[4, 2] = k_isc/135*(D/(de_42)**2)
R[4, 0] = k_isc/45*(D/(de_40)**2)
R[3, 4] = k_isc/45*(D/(de_43)**2)
R[2, 4] = k_isc/135*(D/(de_42)**2)
R[0, 4] = k_isc/45*(D/(de_40)**2)

Sys.dynamics = R

# %%
SimOpt.eigval_mode = False

# do simulation
spec, pop_evolution = sim.teacups(Sys, Exp, SimOpt)
spec = spec/abs(spec).real.max()

# %%
plt.figure()
plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")
plt.plot(Exp.B_z, spec[10].real, label="$t=0.3$ $\mu$s")
plt.plot(Exp.B_z, spec[30].real, label="$t=1.0$ $\mu$s")
plt.legend()
# plt.savefig("tdp_rqm_B.pdf")


plt.figure()
plt.ylabel("population")
plt.xlabel("$t$ / $\mu$s")
a = [0, 1, 2, 3, 4, 5]
labels = ["Q$_{-3/2}$", "Q$_{-1/2}$", "Q$_{+1/2}$",
          "Q$_{+3/2}$", "D$_{-1/2}$", "D$_{+1/2}$"]

time = np.linspace(Exp.t_scale[0], Exp.t_scale[1], Exp.t_points)*1e6
for val in a:
    plt.plot(time, pop_evolution[:, val].real /
             pop_evolution.real.max()*1/6, label=labels[val])
plt.legend(ncol=2)
# plt.savefig("tdp_rqm_pop.pdf")

plt.figure()
plt.plot(time, spec[:, 320].real, label="$B_z$ = 347 mT")
plt.legend()
plt.xlabel("$t$ / $\mu$s")
plt.ylabel("norm. intensity")
# plt.savefig("tdp_rqm_t.pdf")
