import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("stylesheet.mplstyle")

Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()


Sys.spin_system = 'rp'
Sys.g1 = np.array([2.005, 2.005, 2.005])
Sys.g2 = np.array([2.000, 2.000, 2.000])  # - 0.003
Sys.width_gauss = 0.08
Sys.J_ex = 8
Sys.D = 0
Sys.E = 0
Sys.D = 1/3*3.3630
Sys.D_frame = [0, 1.012, -0.017]

Sys.precursor = 'triplet-pnm'
Sys.g_tri = [2.00370, 2.00285, 2.00246]
Sys.population = [1, 0, 0]
Sys.D_tri = -1.9217e+03
Sys.E_tri = 525.4678
Sys.decay = 1e-6

Exp.B_z = np.linspace(337.3, 341.05, 800)
Exp.t_scale = [0, 1e-6]
Exp.t_points = 5
Exp.B_mw = 0.00001
Exp.freq_mw = 9.5*1e9
Exp.cpu_cores = 0

SimOpt.grid_points = 2
SimOpt.sym = "D2h"
SimOpt.grid = "sophe"
SimOpt.space = 'hilbert'

# do simulation
spec = sim.teacups(Sys, Exp, SimOpt)
spec = spec[2].real/abs(spec[2]).real.max()

plt.figure()
plt.plot(Exp.B_z, spec)
plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")

# plt.savefig("rp_hf_precursor.pdf")
