import teacups.classes as cl
import matplotlib.pyplot as plt
import numpy as np
import teacups.simulations as sim

plt.style.use("stylesheet.mplstyle")

# initialize classes with default parameters
sys = cl.Sys()
exp = cl.Exp()
opt = cl.SimOpt()

# set up spin system parameters
sys.g = [2, 2, 2]
sys.width_gauss = 3

sys.A1 = [150, 150, 300]
sys.I1 = 1
sys.n1 = 1
sys.A1_frame = [0, 1, 0]

# alternative hyperfine input (2x I=1/2)
# sys.A = [[[250, 250, 250], [250, 250, 250]]]
# sys.I = [[1/2, 1/2]]

sys.spin_system = "doub"
sys.precursor = "eigen"
sys.population = [1, 0]

# set up experimental parameters
exp.B_z = np.linspace(320, 380, 600)
exp.t_scale = [0, 2e-6]
exp.t_points = 2

# set up simulation option parameters
opt.grid_points = 20

# do simulation
spec = sim.teacups(sys, exp, opt)
spec = spec[1].real / max(abs(spec[1].real))

# %%
# plot result
plt.figure()
plt.plot(exp.B_z, spec)
plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")

# plt.savefig("doublet_with_hyperfines.pdf")
