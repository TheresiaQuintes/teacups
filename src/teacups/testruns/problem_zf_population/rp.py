#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:21:45 2024

@author: theresia
"""

import teacups.classes as cl
import teacups.simulations as sim
import numpy as np
import matplotlib.pyplot as plt


Sys = cl.Sys()
Exp = cl.Exp()
SimOpt = cl.SimOpt()

# choose a triplet spin system
Sys.spin_system = "rp"

# set up g-tensors
Sys.g1 = [2.0020, 2.0020, 2.0020]
Sys.g2 = [1.9999, 1.9999, 1.9999]

# define interactions
Sys.J_ex = 2.5
Sys.D = 3/2*10

# define initial state
Sys.precursor = "singlet"

Sys.precursor = 'triplet-zf'
Sys.population = [0.3, 0., 0.7]
Sys.g_tri = [2.003, 2.003, 2.003]
Sys.D_tri = -700
Sys.E_tri = 100

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 0.05

# experimental setup
Exp.B_z = np.linspace(346.5, 349.5, 600)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 50
Exp.B_mw = 0.00001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt.grid_points = 23

# do the simulation
cal = sim.teacups(Sys, Exp, SimOpt, development=True)
spec_sim = cal.spec_sim
# %%
M = np.loadtxt("M_rp.txt")
plt.figure()
plt.plot(Exp.B_z, spec_sim.real[6]/max(abs(spec_sim.real[6])))
plt.plot(M[0], M[1])
