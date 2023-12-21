#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:48:38 2023

@author: quintes
"""
import easypairspin.epr_setup as epr
import easypairspin.plotting as plotting
from easypairspin.easypairspin import easypairspin
import numpy as np
import matplotlib.pyplot as plt


Sys = epr.Spinsystem()

# choose a triplet spin system
Sys.spin_system = "trip"

# set up g-tensors of the triplet
Sys.g_tri = [2.003, 2.003, 2.003]

# define triplet ZFS
Sys.D_tri = 700
Sys.E_tri = -100

# define initial state
Sys.precursor = "zf"
Sys.population = [0.3, 0.2, 0.1]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
exp_mag_field = np.linspace(320, 380, 600)
Exp = epr.Experimental(exp_mag_field)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt = epr.SimulationOptions()
SimOpt.routine = 'teacups'
SimOpt.grid_points = 25
SimOpt.grid = "fibonacci"
SimOpt.space = "hilbert"

# do the simulation
easypairspin(Sys, Exp, SimOpt)
#fig_1 = plotting.plot_2D(Exp.B_z, Exp.spec_sim.real[1], xlabel='$B_0$/mT',
#                         ylabel='', labels='static spectrum')

M = np.loadtxt("M_tri.txt")
plt.figure()
plt.plot(Exp.B_z, Exp.spec_sim.real[1])
plt.plot(M[0], M[1])
