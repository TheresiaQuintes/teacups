=========
Developer
=========
.. note::
   If you are a developer and would like to add your own features to teacups this secion of the documentation may be useful. It presents all modules and their functions.

TEACUPS consists of 13 Python modules, which contain classes and functions. They can be roughly sorted into three main categories, namely specific simulation modules, general modules, and structuring modules. 

Specific simulation modules
---------------------------
These modules include all functions that are specific for the simulation of transient EPR spectra. The main file simulations.py calls all functions in the right order to obtain the wanted spectrum. The other modules are used to set up and transform the explicit matrices as desired by the :download:`theory </teacups_paper.pdf>`. The functions of the specific simulation modules use a four-class system as input and output, which consists of ``Sys``, ``Exp``, ``SimOpt`` and ``Cal``. All information needed for the simulation is provided and sorted in these four classes as attributes:
* ``Sys`` contains all spin system parameters including relaxation times.
* ``Exp`` contains all experimental parameters.
* ``SimOpt`` contains simulations options, e.g. the choice of the space, the number of grid points, and the simulation mode.
* ``Cal`` is at the beginning an empty class, but it is filled with the results of the calculations during the simulation. These results are fed to further functions later.


creators.py
^^^^^^^^^^^
In the creators-module functions for creating matrix-objects like tensors, detection operator or spinoperator can be found.

.. automodule:: teacups.creators
   :members:

density_matrices.py
^^^^^^^^^^^^^^^^^^^
For each spin-system a function that sets up an initial density matrix is provided in this module.

.. automodule:: teacups.density_matrices
   :members:

hamiltonians.py
^^^^^^^^^^^^^^^
The module hamiltonians contains all functions that are concerned with setting up any hamiltonian for the provided spin systems.

.. automodule:: teacups.hamiltonians
   :members:
   
hyperfine.py
^^^^^^^^^^^^
Functions for setting up hyperfine-tensors and the hyperfine-hamiltonian for a system of one or two spins. Further a function, that calculates the trEPR signal with all hyperfine-interactions can be found here.

.. automodule:: teacups.hyperfine
   :members:
   
input_handler.py
^^^^^^^^^^^^^^^^
The functions of this module deal with the input parameters. Parameters are scaled and transformed to other types for further calculations. The basic class Calculations is set up here.

.. automodule:: teacups.input_handler
   :members:
   
relaxation.py
^^^^^^^^^^^^^
Functions for the definition of relaxation superoperators are provided in this module.

.. automodule:: teacups.relaxation
   :members:
   
signals_and_processing.py
^^^^^^^^^^^^^^^^^^^^^^^^^
Timeevoultion and detection functions can be found in this module.

.. automodule:: teacups.signals_and_processing
   :members:
   
simulations.py
^^^^^^^^^^^^^^
Here the main simulation function, that calls all other functions in the neccessary order, can be found.

.. automodule:: teacups.simulations
   :members:
            

General modules
---------------
These modules contain functions that are generally usable (and could be used outside of *teacups*). Included are, amongst other things, the setup of a grid on a sphere, the Gaussian convolution of a spectrum, or the general setup of a relaxation matrix/ Hamiltonian. The functions take parameters as input and return the desired output. They do not use the four-class system, which makes them more flexible.

convolution.py
^^^^^^^^^^^^^^
The functions in this module are written to convolve a calculated 2D-spectrum along the B-axis. Gaussian convolution or isotrope hyperfine convolution are provided.

.. automodule:: teacups.convolution
   :members:

grid.py
^^^^^^^
Functions for the calculation of a given number of points equally distributed on a fibonacci sphere or using a sophe grid and coordinate transformation from cartesian to spherical coordinates are provided here.

.. automodule:: teacups.grid
   :members:

orientation_dependent_ham.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The tensor rotation function and the creation of linear and bilinear Hamiltonians can be found here.

.. automodule:: teacups.orientation_dependent_ham
   :members:
            

Structuring modules
-------------------
All matrices of the specific simulation modules are objects of the classes defined in the structuring modules: matrix_tools and multioperator_tools. Both classes define matrices including methods like rotation to a number of angle combinations, the setup of a Hamiltonian etc. An object of class *Multioperator* has an attribute called ``B_angle_matrix``. This is a three-dimensional NumPy array containing the values of the operator for a number of magnetic field points and a number of angle points. Its dimension is ``len(B) x len(angles) x spinsystem dimension x spinsystem dimension``. The specific simulation modules often use objects of Multioperator-type, which has the advantage, that calculations can be done for all field points and angle points at once without using loops. This makes the code much faster.

matrix_tools.py
^^^^^^^^^^^^^^^
This module provides classes for basic matrices that are needed for calculations of EPR spectra like a Hamiltonian, tensor or a spinoperator.

.. automodule:: teacups.matrix_tools
   :members:

multioperator_tools.py
^^^^^^^^^^^^^^^^^^^^^^
The Multioperator class in introduced here together with all its methods. This class is usable for fast array calculations as loops are not longer necessary. A Multioperator consists of a 3D-array: The outest dimension is the magnetic field, followed by the angle combinations. In the innest dimension quadratic operators can be found. This are e.g. Hamiltonians. The Multioperaor provides the same methods as the classes from matrix_tools but they a applied for all quadratic matrices in the Multioperator.

.. automodule:: teacups.multioperator_tools
   :members:
