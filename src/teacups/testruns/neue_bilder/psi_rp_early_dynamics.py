import numpy as np
import matplotlib.pyplot as plt
import teacups.simulations as sim
import teacups.classes as cl

plt.style.use("stylesheet.mplstyle")

# %%
Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

Sys.spin_system = 'rp'
Sys.g1 = [2.00304, 2.00262, 2.00232]
Sys.g2 = [2.00564, 2.00494, 2.00217]
Sys.g1_frame = np.array([-15, 28, 27])*(np.pi/180)

Sys.precursor = 'singlet'

Sys.decay = 0.75e-7
Sys.T_relax_1 = 2e-6
Sys.T_relax_2 = 500e-9

Sys.D = -3.3630*3
Sys.D_frame = np.array([0, 58, -1])*(np.pi/180)
Sys.J_ex = 0

Sys.width_gauss = 0.125

Exp.B_z = np.linspace(350.55, 352.33, 500)
Exp.t_scale = [0, 100e-9]
Exp.t_points = 11
Exp.B_mw = 0.03
Exp.freq_mw = 9.8562*1e9

SimOpt.grid_points = 10
SimOpt.space = 'hilbert'

# do the simulation
spec = sim.teacups(Sys, Exp, SimOpt)
spec = spec.real/(max(abs(spec[10].real)))



# %%
m = 1
t = [0, 40, 60, 80, 100]

fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))
for m in range(1, 5):
    n = (m+1)*2
    ax1.plot(Exp.B_z, (spec[n].real)-(1.5*m-1), label=t[m])


plt.legend(loc="upper left", ncol=1)
plt.xlabel("$B_z$ / mT")
plt.ylabel("norm. intensity")
plt.yticks([])

# plt.savefig("psi_rp_early_dynamics.pdf")
