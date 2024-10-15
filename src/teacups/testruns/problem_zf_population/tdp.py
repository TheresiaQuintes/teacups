#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:30:11 2023

@author: quintes
"""
import teacups.classes as cl
import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt


Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# choose a triplet doublet pair spin system
Sys.spin_system = "tdp"

# set up g-tensors of the radical and the triplet
Sys.g = [2.0059, 2.0059, 2.0059]
Sys.g_tri = [2.003, 2.003, 2.003]

# define triplet ZFS
Sys.D_tri = 700
Sys.E_tri = -100

# define couplings of the radical and the triplet
Sys.J_ex = -400000

# define initial state
Sys.precursor = "eigen"
Sys.population = [0.3, 0.1, 0.1, 0.3, 0.5, 0.55]

Sys.precursor = "triplet-zf"
Sys.population = [0.21, 0.2, 0.1, 0.1, 0.2]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(320, 380, 700)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 20
SimOpt.CUPY = False


# do the simulation
cal = sim.teacups(Sys, Exp, SimOpt, development=True)

spec_sim = cal.spec_sim

# %%
M = np.loadtxt("M.txt")
n = 1
plt.figure()
plt.plot(Exp.B_z, spec_sim.real[n]/max(abs(spec_sim.real[n])))
plt.plot(M[0], M[1])

# # %%
# for J_ex in [0, -10, -50, -100, -200, -500, -1000, -2000, -3000, -4000, -5000, -5500, -5800, -6000, -6300, -6500, -7000, -7500, -8000, -10000, -15000, -20000]:
#     print(J_ex)
#     Sys.J_ex = J_ex
#     spec_sim = sim.teacups(Sys, Exp, SimOpt)
#     spec = spec_sim[1].real
#     spec /= max(abs(spec))
#     np.savetxt("sim_zf_tea_"+str(J_ex)+".txt", spec)
# # %%

# for J_ex in [0, -10, -50, -100, -200, -500, -1000, -2000, -3000, -4000, -5000, -5500, -5800, -6000, -6300, -6500, -7000, -7500, -8000, -10000, -15000, -20000]:
#     tea = np.loadtxt("sim_zf_tea_"+str(J_ex)+".txt")
#     easy = np.loadtxt("sim_zf_"+str(J_ex)+".txt")
#     plt.figure()
#     plt.plot(Exp.B_z, tea, label="teacups")
#     plt.plot(Exp.B_z, easy[1], label="easyspin")
#     plt.xlabel("$B_z$ / mT")
#     plt.legend()
#     plt.savefig("sim_zf_J"+str(J_ex)+".png", dpi=200)
