import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("stylesheet.mplstyle")

# initialize classes with default parameters
Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# set up spin System parameters
Sys.g = [2, 2, 2]
Sys.width_gauss = 3
Sys.decay = 5e-6


Sys.spin_system = 'doub'
Sys.precursor = 'eigen'
Sys.population = [0, 1]

# set up Experimental parameters
Exp.B_z = np.linspace(320, 380, 3000)
Exp.t_scale = [0, 10e-6]
Exp.t_points = 100
Exp.B_mw = 0.2

# set up simulation SimOption parameters
SimOpt.grid_points = 1
SimOpt.space = 'hilbert'

# do simulation
spec = sim.teacups(Sys, Exp, SimOpt)
spec = spec/abs(spec).real.max()


# %%

plt.figure()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("Int./a.u.")
plt.plot(Exp.B_z, spec[25].real)

# plt.savefig("doublet_transient_nutations_B.pdf")


plt.figure()
plt.ylabel("Int./a.u.")
plt.xlabel("$t / \mu\mathrm{s}$")
plt.plot(np.linspace(
    Exp.t_scale[0], Exp.t_scale[1], Exp.t_points), spec[:, 1415].real)

# plt.savefig("doublet_transient_nutations_t.pdf")
