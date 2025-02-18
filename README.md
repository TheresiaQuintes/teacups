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
	* Hilbert space ↔️ fast calculations, coherent dynamics
	* Lioville space ↔️ slower calculations, include incoherent dynamics

* Relaxation behaviour can be simulated
	* phenomenological using relaxation times T_1 and  T_2
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

So far, the programme package has been run under an Ubuntu Linux distribution.


Download teacups
================

You can download **teacups** via git::

      git clone https://gitlab.physchem.uni-freiburg.de/koesters/teacups.git

Documentation
=============
The full documentation including advice on installation and usage of the package can be found by opening `teacups/docs/build/html/index.html` in your browser.

