..    include:: <isonum.txt>

==================
Experimental (Exp)
==================
In the class ``Exp`` all parameters that describe the experimental conditions are set as attributes. This can be done in the script by typing::
	
	Exp.Attribute = Value

In the following all possible attributes are named. For each attribute an explanation, cases where it is needed and an example are given. You can get a quick reference file with all attributes :download:`here </quickreference.pdf>`.
	 

Magnetic fields
---------------

Exp.B_z
^^^^^^^
* Static magnetic field vector along the *z*-axis
	* Give all points of the *B*-axis for which the spectrum shall be calculated
	* Create a numpy.ndarray
	* Values in mT
* Obligatory for all simulations
* e.g.::
	
	import numpy as np
	
	# field ranges from 340mT -> 350mT and has 200 points
	Exp.B_z = np.linspace(340, 350, 200)

Exp.B_mw
^^^^^^^^
* Amplitude of the microwave field (along the *x*-axis)
	* Float
	* Given in mT
* Obligatory for all simulations
* e.g.::
	
	Exp.B_mw = 0.001  # mT

Exp.freq_mw
^^^^^^^^^^^
* Frequency of the microwave field
	* Float
	* Given in Hz
* Obligatory for all simulations
* e.g.::
	
	# X-band spectrum
	Exp.freq_mw = 9.75e9  # Hz

Time
----

Exp.t_scale
^^^^^^^^^^^
* Borders of the time range
	* Give upper and lower border in a list
	* Given in s
	* Normally this should start at 0s and ends at your desired time
* Obligatory for all simulations
* e.g.::

	Exp.t_scale = [0, 2e-6]  # s


Exp.t_points
^^^^^^^^^^^^
* Number of time points, that shall be calculated
	* Integer
* Obligatory for all simulations
* e.g.::
	
	Exp.t_points = 100


