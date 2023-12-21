..    include:: <isonum.txt>

===========================
Simulation Options (SimOpt)
===========================
In the class ``SimOpt`` all parameters that set options for the simulation are set as attributes. This can be done in the script by typing::
	
	Sys.Attribute = Value

In the following all possible attributes are named. For each attribute an explanation, cases where it is needed and an example are given. You can get a quick reference file with all attributes :download:`here </quickreference.pdf>`.

If you use the class provided by teacups.classes, the (recommended) default attributes are defined, so that you only have to define the values that you would like to change  from default in your skript::
 
	>>> import teacups.classes as cl
	
	>>> SimOpt = cl.SimOpt()
	
	>>> vars(SimOpt)
	{'grid_points': 10,
	 'grid': 'fibonacci',
	 'space': 'hilbert',
	 'pop_evolution': False,
	 'eigval_mode': False,
	 'cpu_cores': 0}


Grid
----

SimOpt.grid_points
^^^^^^^^^^^^^^^^^^
* Number of points between theta = 0 and theta = pi/2 on the orientational sphere
	* Given as a integer
	* From this value the grid with points distributed equally on a hemisphere is created
	* Typical values range from 1-20 points dependent on the anisotropy of the system, the more anisotrope a system is, the more points are needed
* Obligatory for all simulations
* e.g.::
	
	SimOpt.grid_points = 7


SimOpt.grid
^^^^^^^^^^^
.. warning::
   The sophe-grid uses interpolation between the angle points. This does not work properly for narrow spectra. It is useful for fast calculation of very broad spectra with a small number of angle points to get a quick overview. To get a smooth spectrum the fibonacci grid is still recommended.

* Type of the grid that is chosen to distribute points equally on a hemisphere
	* Set it to a string, either 'fibonacci' or 'sophe'
* Obligatory for all simulations
* e.g.::
	
	SimOpt.grid = 'fibonacci'

SimOpt.width_intp
^^^^^^^^^^^^^^^^^
* If the grid is a sophe grid points between the grid points will be interpolated. To get the desired peaks a interpolation width in mT has to be set
	* Width in mT
* Obligatory, if ``SimOpt.grid = 'sophe'``
* e.g.::
	
	SimOpt.widthintp = 1  # mT

Simulation modes
----------------

SimOpt.space
^^^^^^^^^^^^
* Choose the space for the simulation
	* Either ``'hilbert'`` or ``'liouville'``
* Obligatory for all simulations
* e.g.::
	
	SimOpt.space = 'hilbert'

.. note::
   For quick simulations you should use the Hilbert space. It is recommended to do a Hilbert space simulation for your system first to determine the minimal sufficient number of grid-points and magnetic field/time-points for your spin system to get a smooth graph. If you are interested in relaxation and intend to use ``Sys.T_relax_1/2`` or ``Sys.dynamics`` you have to choose the liouville space. This will be more expensive but you can take all (other) parameters from the initial Hilbert space simulation.


SimOpt.pop_evolution
^^^^^^^^^^^^^^^^^^^^
* Decides wether the population evolution shall be calculated
	* Set it to ``True`` if you want to get a population evolution and to ``False`` in all other cases
	* Attention: If it is set to ``True`` the simulation has two outputs: The spectrum and the population evolution.
* Obligatory for all Liouville space simulations
* e.g.::

	SimOpt.space = 'liouville'
	
	SimOpt.pop_evolution = False
	spec = teacups(Sys, Exp, SimOpt)
	
	SimOpt.pop_evolution = True
	spec, population_evolution = teacups(Sys, Exp, SimOpt)

SimOpt.eigval_mode
^^^^^^^^^^^^^^^^^^
* Calculate only the eigenvalues of the spin system dependent on the magnetic field
	* Set to ``True`` if you do like get only the eigenvalues and stop the simulation after. Set to ``False`` if you would like to run the full simulation.
* Obligatory for all simulations
* e.g.::
	
	SimOpt.eigval_mode = True
	eigvals = teacups(Sys, Exp, SimOpt)

Calculation settings
--------------------

SimOpt.cpu_cores
^^^^^^^^^^^^^^^^
* Number of CPU cores used for time propagation
	* Positive Integer
	* Maximum: The number of CPU cores of your machine
	* If set to 0: All available cores will be used
* Obligatory for all simulations
* e.g.::
	
	# do not use multiprocessing for the time propagation
	SimOpt.cpu_cores = 1
	
	# use all available cores
	SimOpt.cpu_cores = 0
