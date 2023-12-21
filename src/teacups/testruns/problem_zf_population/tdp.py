#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:30:11 2023

@author: quintes
"""
import easypairspin.epr_setup as epr
import easypairspin.plotting as plotting
from easypairspin.easypairspin import easypairspin
import numpy as np
import matplotlib.pyplot as plt


Sys = epr.Spinsystem()

# choose a triplet doublet pair spin system
Sys.spin_system = "tdp"

# set up g-tensors of the radical and the triplet
Sys.g = [2.0059, 2.0059, 2.0059]
Sys.g_tri = [2.003, 2.003, 2.003]

# define triplet ZFS
Sys.D_tri = 700

# define couplings of the radical and the triplet
Sys.J_ex = -20000

# define initial state
#Sys.precursor = "eigen"
#Sys.population = [0.3, 0.2, 0.2, 0.3, 0.5, 0.5]

Sys.precursor = "triplet-zf"
Sys.population = [0.1, 0.2, 0.3, 0.2, 0.1]

# define decay time and line width
Sys.decay = 1e-6
Sys.width_gauss = 1

# experimental setup
exp_mag_field = np.linspace(320, 380, 700)
Exp = epr.Experimental(exp_mag_field)
Exp.t_scale = [0, 2e-6]
Exp.t_points = 2
Exp.B_mw = 0.001
Exp.freq_mw = 9.75e9

# simulation options
SimOpt = epr.SimulationOptions()
SimOpt.routine = 'teacups'
SimOpt.grid_points = 10
SimOpt.grid = "fibonacci"
SimOpt.space = "hilbert"

# do the simulation
easypairspin(Sys, Exp, SimOpt)
#fig_1 = plotting.plot_2D(Exp.B_z, Exp.spec_sim.real[1], xlabel='$B_0$/mT',
#                         ylabel='', labels='static spectrum')

M = np.loadtxt("M.txt")
plt.figure()
plt.plot(Exp.B_z, Exp.spec_sim.real[1])
plt.plot(M[0], M[1])
