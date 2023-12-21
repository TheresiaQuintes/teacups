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
	* Hilbert space |rarr| fast calculations
	* Lioville space |rarr| include dynamics

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

