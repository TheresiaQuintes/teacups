import numpy as np
import matplotlib.pyplot as plt
import teacups.simulations as sim
import teacups.classes as cl

plt.style.use("stylesheet.mplstyle")


# %%
# initialize classes with default parameters
Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# set up spin System parameters
Sys.g_tri = [2.008, 2.008, 2.008]
Sys.D_tri = -898
Sys.E_tri = 161

Sys.spin_system = 'trip'
Sys.precursor = 'zf'
Sys.population = [0.05, 0, 0.95]

Sys.decay = 1e-6
Sys.dynamics = np.array([[0, 0.01e6, 0],
                         [0.01e6, 0, 0.25e6],
                         [0, 0.25e6, 0]])

Sys.width_gauss = 2

# set up Experimental parameters
Exp.B_z = np.linspace(295, 395, 256*4)
Exp.t_scale = [1.9e-6, 3.8e-6]
Exp.t_points = 476
Exp.B_mw = 0.0001

# set up simulation SimOption parameters
SimOpt.grid_points = 40
SimOpt.grid = 'sophe'
SimOpt.sym = "D2h"
SimOpt.space = 'liouville'
SimOpt.pop_evolution = True



# do simulation
spec, pop = sim.teacups(Sys, Exp, SimOpt)

#%%
sim = spec.real/max(abs(spec[25].real))
t = np.linspace(1.9e-6, 3.8e-6, 476)

plt.figure()
plt.plot(t, pop.real)
plt.xlabel("$t$")
plt.ylabel("population of states")

# plt.savefig("znp_triplet_asymmetric_pop.pdf")


#%%
plt.figure()
for i in range(0, 451, 150):
    plt.plot(Exp.B_z, sim[25+i], label=str(t[25+i]))

plt.legend()
plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")
plt.yticks([])

# plt.savefig("znp_triplet_asymmetric_spec.pdf")
