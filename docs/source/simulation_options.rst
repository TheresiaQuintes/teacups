..    include:: <isonum.txt>

===========================
Simulation Options (SimOpt)
===========================
In the class ``SimOpt`` all parameters that define the options for the simulation are set as attributes. This can be done in the script by typing::
	
	SimOpt.Attribute = Value

If the class is initialized by::

	import teacups.classes as cl
	SimOpt = cl.SimOpt()

the following attributes are preallocated and can be used as a starting point, which should be adapted by the user:

* grid_points: 15
* grid: "fibonacci"
* theta: [np.pi]
* phi: [0]
* sym: "D2h"
* space: "hilbert"
* pop_evolution: False
* eigval_mode: False
* cpu_cores: 1
* CUPY: False
* extend_t: False

In the following all possible attributes are named. For each attribute an explanation, cases where it is needed and an example are given. You can get a quick reference file with all attributes :download:`here <./../quickreference/quickreference.pdf>`.

Grid
----

SimOpt.grid
^^^^^^^^^^^
* Type of the grid that is chosen to distribute points equally on a hemisphere
	* Set it to a string, either 'fibonacci' or 'sophe' (add SimOpt.grid_points)
	* Set it to 'single' if you don't want to simulate a full hemisphere but your own chosen angle points (add SimOpt.theta and SimOpt.phi)
* Obligatory for all simulations
* e.g.::
	
	SimOpt.grid = 'fibonacci'

.. hint:: If you know the point group of your spin system it is more efficient to choose ``SimOpt.grid = 'sophe'``. You have to add ``SimOpt.sym``.

SimOpt.grid_points
^^^^^^^^^^^^^^^^^^
* Number of points between theta = 0 and theta = pi/2 on the orientational sphere
	* Integer
	* From this value the grid with points distributed equally on a hemisphere is created
	* Typical values range from 1-30 points dependent on the anisotropy of the system, the more anisotrope a system is, the more points are needed
* Obligatory for all simulations with 'fibonacci' or 'sophe' grid
* e.g.::
	
	SimOpt.grid = 'fibonacci'
	SimOpt.grid_points = 7

SimOpt.theta / SimOpt.phi
^^^^^^^^^^^^^^^^^^^^^^^^^
* Lists with angles to choose one or multiple special orientations
	* Both lists have to have the same length
	* Angles are set in rad
* Obligatory for all simulations with SimOpt.grid = 'single'
* e.g.::
	
	SimOpt.grid = 'single'
	SimOpt.theta = [0, 0.5]
	SimOpt.phi = [0, np.pi/2]

SimOpt.sym
^^^^^^^^^^
* Point group of the spin system
	* String
	* The following symmetries are possible: "C1", "Ci", "C2h", "S6", "C4h", "C6h", "D2h", "Th", "D3d", "D4h", "Oh", "D6h", "Dooh", "O3"
	* Many spin systems have the symmetry "D2h"
* Obligatory for all simulations with SimOpt.grid = 'sophe'
* e.g.::
	
	SimOpt.grid = "sophe"
	SimOpt.sym = "D2h"

Simulation modes
----------------

SimOpt.space
^^^^^^^^^^^^
* Choose the space for the simulation
	* Either ``'hilbert'`` or ``'liouville'``
	* Choose the Hilbert-space if you are interested in coherent dynamics
	* Choose the Liouville-space if you would like to include incoherent dynamics
* Obligatory for all simulations
* e.g.::
	
	SimOpt.space = 'hilbert'

.. hint::
   As a starting point you should do a Hilbert space simulation (which is less expensive) to determine the minimal sufficient number of grid-points and magnetic field/time-points to get a satisfying smooth spectrum. If you are then intrested in incoherent dynamics you have to switch to Liouville space but are now able to take over all other parameters from the Hilbert space simulation.


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
	* Set to ``True`` if you do like to get only the eigenvalues and stop the simulation after. Set to ``False`` if you would like to run the full simulation.
* Obligatory for all simulations
* e.g.::
	
	SimOpt.eigval_mode = True
	eigvals = teacups(Sys, Exp, SimOpt)

* The shape of the returned eigval-array is::
	B-points x grid-points x eigenvalues

SimOpt.extend_t
^^^^^^^^^^^^^^^
* If Sys.sigma_time is set the signal can be extended to negative values to see the full baseline. Set extend_t to True:
* e.g.::
	
	Sys.sigma_time = 0.1e-6
	SimOpt.extend_t = True
* The new t-axis is saved in ``Exp.t``

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

SimOpt.CUPY
^^^^^^^^^^^
* For some time consuming functionalities GPU-support is availabe
	* Set to True, if you want to calculate some parts on your (nvidia) GPU
	* Set to False, if you do not want to calculate on your GPU
* Obligatory for all simulations
* e.g.::
	
	SimOpt.CUPY = False

