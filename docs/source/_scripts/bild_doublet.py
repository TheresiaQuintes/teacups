import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt
import teacups.classes as cl


# initialize classes with default parameters
Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# set up spin System parameters
Sys.g = [2, 2, 2]
Sys.width_gauss = 3

Sys.A1 = [150, 150, 300]
Sys.I1 = 1
Sys.n1 = 1
Sys.A1_frame = [0, 1, 0]

# alternative hyperfine input
# Sys.A = [[[250, 250, 250], [250, 250, 250]]]
# Sys.I = [[1/2, 1/2]]

Sys.spin_system = 'doub'
Sys.precursor = 'basis'
Sys.population = [0, 1]

# set up Experimental parameters
Exp.B_z = np.linspace(320, 380, 600)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2

# set up simulation SimOption parameters
SimOpt.grid_points = 15

# do simulation
spec = sim.teacups(Sys, Exp, SimOpt)

# plot result
plt.figure()
plt.plot(Exp.B_z, spec[1].real)
plt.show()

# SimOptional: save parameters as dictionaries
Sys_dict = vars(Sys)
Exp_dict = vars(Exp)
SimOpt_dict = vars(SimOpt)
