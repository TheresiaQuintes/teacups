import matplotlib.pyplot as plt
import numpy as np
import teacups.classes as cl
import teacups.simulations as sim
import sys
sys.path.append("../../")


sys = cl.Sys()
exp = cl.Exp()
opt = cl.SimOpt()

sys.spin_system = 'rp'
sys.g1 = [2.00431, 2.00360, 2.00217]
sys.g2 = [2.00370, 2.00285, 2.00246]
sys.g2_frame = [2.21656815, 1.34390352, 4.31096325]

sys.precursor = 'singlet'
# Triplet precursor
# sys.precursor = 'triplet-zf'
# sys.g_tri = [2.00370, 2.00285, 2.00246]
# sys.population = [0.67, 0.33, 0]
# sys.D_tri = -1.9217e+03
# sys.E_tri = +525.4678

sys.decay = 1e-6

sys.D = -10.0890
sys.D_frame = [0, 1.9198621771937625, 1.9198621771937625]
sys.J_ex = 2.0458

sys.width_gauss = 0.1

exp.B_z = np.linspace(344, 347, 500)
exp.t_scale = [0, 2e-6]
exp.t_points = 60
exp.B_mw = 0.001
exp.freq_mw = 9.68*1e9

opt.grid_points = 20
opt.space = 'hilbert'

# do the simulation
spec = sim.teacups(sys, exp, opt)

# %% Plot 2D
plt.figure()
plt.plot(exp.B_z, spec[10].real/max(abs(spec[10].real)))

# %% Plot 3D
plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(exp.B_z, np.linspace(0, 2e-6, exp.t_points))
ax.plot_surface(x, y, spec.real, cmap='coolwarm')

# %% Plot surface
fig, ax = plt.subplots()
ax.pcolormesh(x, y, spec.real, cmap='coolwarm', linewidth=0, antialiased=False)
plt.xlabel('$B_z$ / mT')
plt.ylabel('$t$ / s')
plt.show()
