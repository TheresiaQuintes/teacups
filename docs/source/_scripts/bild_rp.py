import teacups.simulations as sim
import teacups.classes as cl
import numpy as np
import matplotlib.pyplot as plt


Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

Sys.spin_system = 'rp'
Sys.g1 = [2.00431, 2.00360, 2.00217]
Sys.g2 = [2.00370, 2.00285, 2.00246]
Sys.g2_frame = [2.21656815, 1.34390352, 4.31096325]

Sys.precursor = 'singlet'
# Triplet precursor
# Sys.precursor = 'triplet-zf'
# Sys.g_tri = [2.00370, 2.00285, 2.00246]
# Sys.population = [0.67, 0.33, 0]
# Sys.D_tri = -1.9217e+03
# Sys.E_tri = +525.4678

Sys.decay = 1e-6

Sys.D = -10.0890
Sys.D_frame = [0, 1.9198621771937625, 1.9198621771937625]
Sys.J_ex = 2.0458

Sys.width_gauss = 0.1

Exp.B_z = np.linspace(344, 347, 500)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 60
Exp.B_mw = 0.001
Exp.freq_mw = 9.68*1e9

SimOpt.grid_points = 20
SimOpt.space = 'hilbert'

# do the simulation
spec = sim.teacups(Sys, Exp, SimOpt)

# %% Plot 2D
plt.figure()
plt.plot(Exp.B_z, spec[10].real/max(abs(spec[10].real)))

# %% Plot 3D
plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(Exp.B_z, np.linspace(0, 2e-6, Exp.t_points))
ax.plot_surface(x, y, spec.real, cmap='coolwarm')

# %% Plot surface
fig, ax = plt.subplots()
ax.pcolormesh(x, y, spec.real, cmap='coolwarm', linewidth=0, antialiased=False)
plt.xlabel('$B_z$ / mT')
plt.ylabel('$t$ / s')
plt.show()
