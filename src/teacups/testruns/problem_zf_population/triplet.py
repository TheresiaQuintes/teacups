#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:48:38 2023

@author: quintes
"""
import teacups.classes as cl
import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt


Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# choose a triplet spin system
Sys.spin_system = "trip"

# set up g-tensors of the triplet
Sys.g_tri = [2.003, 2.003, 2.003]

# define triplet ZFS
Sys.D_tri = -700
Sys.E_tri = 100

# define initial state
Sys.precursor = "zf"
Sys.population = [0.3, 0.2, 0.1]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
Exp.B_z = np.linspace(320, 380, 600)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 25

# do the simulation
cal = sim.teacups(Sys, Exp, SimOpt, development=True)
spec_sim = cal.spec_sim
# %%
M = np.loadtxt("M_tri.txt")
plt.figure()
plt.plot(Exp.B_z, spec_sim.real[1]/max(abs(spec_sim.real[1])))
plt.plot(M[0], M[1])
