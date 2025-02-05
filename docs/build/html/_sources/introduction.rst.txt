============
Introduction
============

..    include:: <isonum.txt>

What is teacups?
================

Teacups stands for *Time resolved EPR: Algebraic Calculations for Unequally Populated Spin Systems*. It is a Python3 package for the simulation of timeresolved, transient EPR spectra of spinporalized spin species.

What are its features?
======================


* Until now four spin systems can be calculated
	* Doublets ``'doub'``
	* Triplets ``'trip'``
	* Spin correlated radical pairs ``'rp'``
	* Triplet-Doublet-Pairs ``'tdp'``
	
* Calculations can be done in
	* Hilbert space |rarr| fast calculations, coherent dynamics
	* Lioville space |rarr| slower calculations, include incoherent dynamics

* Relaxation behaviour can be simulated
	* phenomenological using relaxation times T\ :sub:`1`\  and  T\ :sub:`2`\
	* explicit by defining transitions between the systems eigenstates

* The following results can be gained:
	* 3D-EPR-spectrum (intensity dependent on time and magnetic field)
	* Population evolution (evolution of the populations of the eigenstates)
	* Eigenvalues of the spin system




Requirements
============

You will need the following installed on your system to run a TEACUPS-simulation:

* Python3
* numpy
* scipy
* matplotlib
* tqdm


Literature
==========
To get an overview about teacups and its theoretical background you can consult this :download:`paper <./_static/teacups_main.pdf>` and the :download:`SI <./_static/teacups_SI.pdf>`. 

All Simulationarguments are summarized in the :download:`quick reference <./../quickreference/quickreference.pdf>`.
