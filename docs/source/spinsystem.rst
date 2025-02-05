..    include:: <isonum.txt>

=================
Spin System (Sys)
=================
In the class ``Sys`` all parameters that describe the fundamental spin system are set as attributes. This can be done in the script by typing::
	
	Sys.Attribute = Value

If the class is initialized by::

	import teacups.classes as cl
	Sys = cl.SpinSystem()

the following attributes are preallocated (but can be newly assigned by the user):

* spin_system: "doub"
* precursor: "basis"
* population: [0, 1]
* g: [1.95, 2, 2.1]
* width_gauss: 3
* decay: 1e-6
* dynamics: None
* T_relax_1: 1e-6
* T_relax_2: 1e-6

In the following all possible attributes are named. For each attribute an explanation, cases where it is needed and an example are given. You can get a quick reference file with all attributes :download:`here <./../quickreference/quickreference.pdf>`.


Polarisation
------------

Sys.spin_system
^^^^^^^^^^^^^^^
* Defines the type of the spin system, set it to
	* ``'doub'`` for simulating a doublet.
	* ``'trip'`` for simulating a triplet.
	* ``'rp'`` for simulating a radical pair.
	* ``'tdp'`` for simulating a triplet-doublet pair.
* Obligatory for all simulations
* e.g.::

	 Sys.spin_system = 'doub'

Sys.precursor
^^^^^^^^^^^^^
* Defines the initial polarisation of the spin system. Dependent on the chosen system different initial polarisations are available. The precursor attribute defines the basis chosen for the initial polarisation. Set Sys.precursor to
	- ``'eigen'`` to give the polarisation vector in the eigenbasis of the system. |rarr| possible for all spin systems
	- ``'zf'`` to give the poalrisation vector in the basis of the zero field states. |rarr| possible for triplet spin systems
	- ``'singlet'`` to initialize a pure singlet polarisation. |rarr| possible for radical pair spin systems
	- ``'triplet-zf'`` to give the polarisation vector as a trplet precursor in the zero field basis |rarr| possible for radical pairs (with triplet precursor) and triplet doublet pairs (consisting of the triplet precursor and a radical precursor
	
* Obligatory for all simulations
* e.g.::

	 Sys.precursor = 'eigen'

.. hint:: Some bases are especially recommended for special spin systems:
	
	- for ``'doub'`` use ``'eigen'`` for the population of the alpha and beta states in ascending energetic order 
	- for ``'trip'`` use ``'zf'`` for ISC triplets (populate :math:`T_x`, :math:`T_y` and :math:`T_z` in ascending energetic order) and ``'eigen'`` for recombination triplets (populate :math:`T_-`, :math:`T_0` and :math:`T_+` in ascending energetic order)
	- for ``'rp'`` use ``'singlet'`` for a singlet precursor and ``'triplet-zf'`` for a triplet precursor
	- for ``'tdp'`` use ``'triplet-zf'`` to populate the zerofield levels of an ISC-triplet perecursor and a radical precursor and ``'eigen'`` to populate the highfield levels of the coupled system
	
Sys.population
^^^^^^^^^^^^^^
* Set the initial population vector in the basis that is defined in Sys.precursor.
	* List of floats
	* In general the populations are given in ascending order of the energy levels.
* Obligatory for all simulations
* e.g.::
	
	# doublet, eigen: populate the alpha and beta spinstates
	Sys.population = [0.5, 0.505]
	
	# triplet, zf: populate the three zerofield states of a triplet Tx, Ty and Tz
	Sys.population = [0, 1, 0]
	
	# radical pair, triplet-zf: populate the three zerofield states of the triplet-precursor
	Sys.population = [1, 1, 0]
	
	# radical pair, singlet
	# no Sys.population is needed
	
	# triplet doublet pair, eigen: populate the six eigenstates of the triplet doublet pair
	Sys.population = [0.3, 0, 0, 0.3, 0.1, 0.1]  
	
	# triplet doublet pair, triplet-zf:
	# populate the two doublet precursor levels alpha and beta
	# and the three triplet precursor zero field levels
	# [alpha, beta, Tx, Ty, Tz]
	Sys.population = [0.2, 0.2, 0.3, 0.2, 0.1]


g-tensors
---------
.. math:: 
   \mathbf{g} &= \begin{pmatrix} g_{xx} & 0 & 0\\ 0 & g_{yy} & 0 \\ 0 & 0 & g_{zz} \end{pmatrix} \\
   &\text{}\\
      \mathbf{rot}_{z,y',z''} &=  \begin{pmatrix}
                                \cos \phi \cos \vartheta \cos \psi - \sin \phi \sin \psi & -\cos \phi \cos \vartheta \sin \psi -\sin \phi \cos \psi & \cos \phi \sin \vartheta \\
                                \sin \phi \cos \vartheta \cos \psi + \cos \phi \sin \psi & -\sin \phi \cos \vartheta \sin \psi + \cos \phi \cos \psi & \sin \phi \sin \vartheta \\
                                -\sin \vartheta \cos\psi & \sin \vartheta \sin \psi & \cos \vartheta \\
             \end{pmatrix}
	   	   
Sys.g
^^^^^
* Principal elements of the g-tensor of a doublet spin
	* List of floats with three members (:math:`g_{xx}`, :math:`g_{yy}`, :math:`g_{zz}`)
	
* Obligatory for doublet systems and triplet doublet pairs
* e.g.::
	
	Sys.g = [2.001, 2.005, 2.003] 

Sys.g_tri
^^^^^^^^^
* Diagonal elements of the g-tensor of a triplet spin
	* List of floats with three members (:math:`g_{xx}`, :math:`g_{yy}`, :math:`g_{zz}`)
	  
* Obligatory for triplet systems and triplet doublet pairs and radical pairs with triplet precursor
* e.g.::
	
	Sys.g_tri = [2.01, 2.05, 1.99] 
		
Sys.g1
^^^^^^
* Diagonal elements of the g1-tensor of a radical pair
	* List of floats with three members (:math:`g_{xx}`, :math:`g_{yy}`, :math:`g_{zz}`)
	* Acceptor
* Obligatory for radical pair simulation
* e.g.::
	
	Sys.g1 = [2.001, 2.005, 2.003] 
   
Sys.g2
^^^^^^
* Diagonal elements of the g2-tensor of a radical pair
	* List of floats with three members (:math:`g_{xx}`, :math:`g_{yy}`, :math:`g_{zz}`)
	* Donor
* Obligatory for radical pair simulation
* e.g.::
	
	Sys.g2 = [2.001, 2.005, 2.003] 

Sys.g_frame
^^^^^^^^^^^
* Three euler angles to rotate the doublet g-tensor from molecular to laboratory frame
	* List of floats with three members (:math:`\phi`, :math:`\vartheta`, :math:`\psi`)
* Optional for doublet and triplet doublet pair systems
* e.g.::
	
	Sys.g_frame = [0, np.pi/2, 0]  # rad

Sys.g_tri_frame
^^^^^^^^^^^^^^^
* Three euler angles to rotate the triplet g-tensor from molecular to laboratory frame
	* List of floats with three members (:math:`\phi`, :math:`\vartheta`, :math:`\psi`)
* Optional for triplet systems, triplet doublet pairs and radical pairs with triplet precursor
* e.g.::
	
	Sys.g_tri_frame = [0, np.pi/2, 0]  # rad
		
Sys.g1_frame
^^^^^^^^^^^^
* Three euler angles to rotate the g1-tensor of a radical pair from molecular to laboratory frame
	* List of floats with three members (:math:`\phi`, :math:`\vartheta`, :math:`\psi`)
* Optional for radical pair simulation
* e.g.::
	
	Sys.g_1_frame = [0, np.pi/2, 0]  # rad
   
Sys.g2_frame
^^^^^^^^^^^^
* Three euler angles to rotate the g2-tensor of a radical pair from molecular to laboratory frame
	* List of floats with three members (:math:`\phi`, :math:`\vartheta`, :math:`\psi`)
* Optional for radical pair simulation
* e.g.::
	
	Sys.g_2_frame = [0, np.pi/2, 0]  # rad

Spin-spin interactions
----------------------
Sys.D
^^^^^
* Dipolar coupling constant *D* for two interacting spin species
	* Float in MHz
	* The dipolar coupling Hamiltonian is defined as:
	.. math::
	   \hat{H}_{\mathrm{dip}} &= \mathbf{S}_1 \mathbf{D} \mathbf{S}_2  \\
	   &\text{ }\\
	   \mathbf{D} &= \begin{pmatrix} -1/3 D+E & 0 & 0\\ 0 & -1/3 D-E & 0 \\ 0 & 0 & 2/3 D \end{pmatrix}	   
* Optional for radical pairs and triplet doublet pairs
* e.g.::
	
	Sys.D = 300  # MHz

Sys.E
^^^^^
* Dipolar coupling constant *E* for two interacting spin species
	* Float in MHz
	* The dipolar coupling Hamiltonian is defined as:
	.. math::
	   \hat{H}_{\mathrm{dip}} &= \mathbf{S}_1 \mathbf{D} \mathbf{S}_2  \\
	   &\text{ }\\
	   \mathbf{D} &= \begin{pmatrix} -1/3 D+E & 0 & 0\\ 0 & -1/3 D-E & 0 \\ 0 & 0 & 2/3 D \end{pmatrix}	   
* Optional for radical pairs and triplet doublet pairs
* e.g.::
	
	Sys.E = -100  # MHz

Sys.D_frame
^^^^^^^^^^^
* Three euler angles to rotate the D-tensor for dipolar couplings from molecular to laboratory frame
	* List of floats with three members (phi, theta, psi)
* Optional for radical pair and tripled douplet pair simulations if a dipolar coupling is given
* e.g.::
	
	Sys.D_frame = [0, np.pi/2, 0]  # rad
	
Sys.J_ex
^^^^^^^^
* Exchange coupling constant *J* for two interacting spin species
	* Float in MHz
	* For radical pairs the exchange coupling Hamiltonian is set up as:
	.. math::
	   \hat{H}_{\mathrm{ex}} = -J\left(1/2+2\mathbf{S}_1\mathbf{S}_2\right)
	* For triplet doublet pairs the exchange coupling Hamiltonian is set up as:
	.. math::
	   \hat{H}_{\mathrm{ex}} = J\mathbf{S}_1\mathbf{S}_2
* Optional for radical pairs and triplet doublet pairs
* e.g.::
	
	Sys.J = -20000  # MHz

Sys.D_tri
^^^^^^^^^
* Zero field splitting constant *D* for triplets
	* Float in MHz
	* The ZFS Hamiltonian is defined as:
	.. math::
	   \hat{H}_{\mathrm{ZFS}} &= \mathbf{S} \mathbf{D} \mathbf{S} \\
	   &\text{ }\\
	   \mathbf{D} &= \begin{pmatrix} -1/3 D+E & 0 & 0\\ 0 & -1/3 D-E & 0 \\ 0 & 0 & 2/3 D \end{pmatrix}	   
* Optional for triplets, radical pairs with triplet precursor and triplet doublet pairs
* e.g.::
	
	Sys.D_tri = 700  # MHz

Sys.E_tri
^^^^^^^^^
* Zero field splitting constant *E* for triplets
	* Float in MHz
	* The ZFS Hamiltonian is defined as:
	.. math::
	   \hat{H}_{\mathrm{ZFS}} &= \mathbf{S} \mathbf{D} \mathbf{S}  \\
	   &\text{ }\\
	   \mathbf{D} &= \begin{pmatrix} -1/3 D+E & 0 & 0\\ 0 & -1/3 D-E & 0 \\ 0 & 0 & 2/3 D \end{pmatrix}	   
* Optional for triplets, radical pairs with triplet precursor and triplet doublet pairs
* e.g.::
	
	Sys.E_tri = -100  # MHz

Sys.D_tri_frame
^^^^^^^^^^^^^^^
* Three euler angles to rotate the ZFS tensor of triplets from molecular to laboratory frame
	* List of floats with three members (phi, theta, psi)
* Optional for triplets, radical pairs with triplet precursor and triplet doublet pairs if a ZFS is defined
* e.g.::
	
	Sys.D_tri_frame = [0, np.pi/2, 0]  # rad


Hyperfine interaction
---------------------
.. note::
   Hyperfine interactions for coupled systems do not work properly until now.
   
Sys.Ai
^^^^^^
* Diagonal elements of a hyperfine tensor of nucleus i
	* i ranges from 1-5
	* List of floats with three members (Ax, Ay and Az)
* Optional for doublet and triplet simulations
* e.g.::
	
	Sys.A1 = [100, 100, 200]

Sys.Ai_frame
^^^^^^^^^^^^
* Three euler angles to rotate the hyperfine tensor of nucleus i from molecular to laboratory frame
	* i ranges from 1-5
	* List of floats with three members (phi, theta, psi)
* Optional for doublets and triplets where a hyperfine tensor A is given
* e.g.::
	
	Sys.A_1_frame = [0, np.pi/2, 0]  # rad

Sys.Ii
^^^^^^
* Spin quantum number of nucleus (-group) i
	* integer or half
	* i ranges from 1-5
* Optional for doublets and triplets where a hyperfine tensor A is given
* e.g.::

	Sys.I1 = 1/2

Sys.ni
^^^^^^
* Number of nuclei in core group i
	* integer
	* i ranges from 1-5
* Optional for doublets and triplets where a hyperfine tensor A is given
* e.g.::

	Sys.n1 = 3
	
Relaxation
----------
Sys.decay
^^^^^^^^^
* After a simulation in Hilbert space the resulting spectrum is convoluted with an exponential function to simulate an exponential decay of the signal. The exponential decay constant tau (which has no physical meaning) can be set in ``Sys.decay``.
	* Given as a float in s
	* The convolution is done by:
	.. math::
	   S_{\mathrm{decay}} = S * \exp\left[-\tau t\right]
* Obligatory for all simulations in Hilbert space
* e.g.::
	
	Sys.decay = 1e-6  # s

Sys.T_relax_1
^^^^^^^^^^^^^
* This is the longitudinal relaxation time for the simulation of a phenomenological relaxation.
	* Given as a float in s
	* Describes the relaxation time that drives the populations to equilibrium
* Optional for all simulations in Liouville space (choose ``Sys.dynamics = None`` if you want to use ``Sys.T_relax_1`` and ``Sys.T_relax_2``)
* e.g.::
	
	Sys.dynamics = None
	Sys.T_relax_1 = 1e-6  # s
	Sys.T_relax_2 = 1e-7  # s

Sys.T_relax_2
^^^^^^^^^^^^^
* This is the transversal relaxation time for the simulation of a phenomenological relaxation.
	* Given as a float in s
	* Describes the relaxation time that describes the decay of the coherences
* Optional for all simulations in Liouville space (choose ``Sys.dynamics = None`` if you want to use ``Sys.T_relax_1`` and ``Sys.T_relax_2``)
* e.g.::
	
	Sys.dynamics = None
	Sys.T_relax_1 = 1e-6  # s
	Sys.T_relax_2 = 1e-7  # s

Sys.dynamics
^^^^^^^^^^^^
* Set your own relaxation matrix for transitions between the systems eigenstates
	* The rateconstants are given in 1/s
	* The matrix has the dimension of the spin system (if the spin system e.g. has 4 eigenfunctions, its dimension is 4x4)
	* The diagonal elements describe the decay of populations to states "outside" of the spin system
	* The off-diagonal elements describe transitions between the eigenstates
	* The order of the eigenstates is ascending
* Optional for all simulations in Liouville space
* e.g.::

	# System with three eigenstates
	# Population of eigenstate 1 (lowest energy) decays with -1/5s^-1
	# Transition between state 1 <-> 3 with a rate of 1/2s^-1
	Sys.dynamics = np.array([[-1/5, 0, 1/2],
	                         [   0, 0,   0],
	                         [ 1/2, 0,   0]])

Line width
----------
Sys.width_gauss
^^^^^^^^^^^^^^^
* After the spectrum is built it is convoluted with a gaussian function. The width of this function can be set in Sys.width_gauss.
	* Float in mT
* Obligatory for all simulations
* e.g.::
	
	Sys.width_gauss = 1  # mT	            
