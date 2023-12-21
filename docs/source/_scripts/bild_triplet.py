import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt


# initialize classes with default parameters
Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# set up spin System parameters
Sys.g_tri = [2, 2, 2]
Sys.D_tri = 700
Sys.E_tri = -200

Sys.spin_system = 'trip'
Sys.precursor = 'zf'
Sys.population = [0, 0, 1]

Sys.decay = 1e-6
Sys.width_gauss = 3.5

# set up Experimental parameters
Exp.B_z = np.linspace(300, 400, 1000)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 150

# set up simulation SimOption parameters
SimOpt.grid_points = 15
SimOpt.space = 'hilbert'


# do simulation
spec = sim.teacups(Sys, Exp, SimOpt)

# plot result
plt.figure()
plt.plot(Exp.B_z, spec[25].real)

plt.figure()
ax = plt.axes()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("$t / \mu\mathrm{s}$")
x, y = np.meshgrid(Exp.B_z, np.linspace(
    Exp.t_scale[0], Exp.t_scale[1], Exp.t_points))
ax.pcolormesh(x, y, spec.real, cmap="RdBu", shading="auto")
plt.show()

# SimOptional: save parameters as dictionaries
Sys_dict = vars(Sys)
Exp_dict = vars(Exp)
SimOpt_dict = vars(SimOpt)
