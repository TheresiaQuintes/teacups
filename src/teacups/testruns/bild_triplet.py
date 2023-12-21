import sys
sys.path.append("../../")

import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt


# initialize classes with default parameters
sys = cl.Sys()
exp = cl.Exp()
opt = cl.SimOpt()

# set up spin system parameters
sys.g_tri = [2, 2, 2]
sys.D_tri = 700
sys.E_tri = -200

sys.spin_system = 'trip'
sys.precursor = 'zf'
sys.population = [0, 0, 1]

sys.decay = 1e-6
sys.width_gauss = 3.5

# set up experimental parameters
exp.B_z = np.linspace(300, 400, 1000)
exp.t_scale = [0, 2e-6]
exp.t_points = 150

# set up simulation option parameters
opt.grid_points = 15
opt.space = 'hilbert'


# do simulation
spec = sim.teacups(sys, exp, opt)

# plot result
plt.figure()
plt.plot(exp.B_z, spec[25].real)

plt.figure()
ax = plt.axes()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("$t / \mu\mathrm{s}$")
x, y = np.meshgrid(exp.B_z, np.linspace(
    exp.t_scale[0], exp.t_scale[1], exp.t_points))
ax.pcolormesh(x, y, spec.real, cmap="RdBu", shading="auto")
plt.show()

# optional: save parameters as dictionaries
sys_dict = vars(sys)
exp_dict = vars(exp)
opt_dict = vars(opt)
