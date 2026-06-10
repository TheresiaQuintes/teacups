import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("stylesheet.mplstyle")

# initialize classes with default parameters
Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()


Sys.spin_system = "rp"
Sys.g1 = [2.00304, 2.00262, 2.00232]
Sys.g1_frame = [-0.262, 0.489, 0.471]
Sys.g2 = [2.00564, 2.00494, 2.00217]

# Sys.precursor = 'singlet'
# Triplet precursor
Sys.precursor = "triplet-zf"
Sys.g_tri = [2.00370, 2.00285, 2.00246]
Sys.population = [1, 0, 0]
Sys.D_tri = -1.9217e03
Sys.E_tri = 525.4678

Sys.decay = 1e-6
Sys.T_relax_1 = 1e-6
Sys.T_relax_2 = 1e-6

Sys.D = 1 / 3 * 3.3630
Sys.D_frame = [0, 1.012, -0.017]
Sys.J_ex = 0

Sys.width_gauss = 0.08

Exp.B_z = np.linspace(350.5, 352.3, 800)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 300
Exp.B_mw = 0.03
Exp.freq_mw = 9.8562 * 1e9

SimOpt.grid_points = 20
SimOpt.space = "hilbert"
SimOpt.cpu_cores = 0

# do simulation
spec = sim.teacups(Sys, Exp, SimOpt)
spec = spec / abs(spec).real.max()

# %%
plt.figure()
plt.xlabel("$B_z / \mathrm{mT}$")
plt.ylabel("Int./a.u.")
plt.plot(Exp.B_z, spec[150].real)
plt.savefig("rp_quantum_beats_B.pdf")

plt.figure()
plt.ylabel("norm. intensity")
plt.xlabel("$t$ / $\mu$s")
plt.plot(
    np.linspace(Exp.t_scale[0], Exp.t_scale[1], Exp.t_points) * 1e6, spec[:, 269].real
)
plt.savefig("rp_quantum_beats_t.pdf")

# %% Reproduction of a worse time resolution

Sys.sigma_time = 0.1e-6
SimOpt.extend_t = True

# do simulation
spec = sim.teacups(Sys, Exp, SimOpt)
spec = spec / abs(spec).real.max()

plt.figure()
plt.ylabel("norm. intensity")
plt.xlabel("$t$ / $\mu$s")
plt.plot(Exp.t * 1e6, spec[:, 269].real)
plt.savefig("rp_no_quantum_beats_t.pdf")
