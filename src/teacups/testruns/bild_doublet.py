import sys
sys.path.append("../../")

import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt
import teacups.classes as cl


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

# alternative hyperfine input
# sys.A = [[[250, 250, 250], [250, 250, 250]]]
# sys.I = [[1/2, 1/2]]

sys.spin_system = 'doub'
sys.precursor = 'basis'
sys.population = [0, 1]

# set up experimental parameters
exp.B_z = np.linspace(320, 380, 600)
exp.t_scale = [0, 2e-6]
exp.t_points = 2

# set up simulation option parameters
opt.grid_points = 15

# do simulation
spec = sim.teacups(sys, exp, opt)

# plot result
plt.figure()
plt.plot(exp.B_z, spec[1].real)
plt.show()

# optional: save parameters as dictionaries
sys_dict = vars(sys)
exp_dict = vars(exp)
opt_dict = vars(opt)
