===============
Getting started
===============


Download teacups
================

You can download **teacups** via git::

   git clone 'https://gitlab.physchem.uni-freiburg.de/koesters/teacups.git'


Installation
============

.. note::

   Anaconda is used for development and so it is highly recommended. Virtualenv might 
   be possible, but there is no guarantee for full functionality!


#. Open a Conda-Shell

#. Setup a new conda environment::

      conda create -n teacups
   
   .. note::
   
      You can also use another environment name or use an existing environment. It is
      recommended to use a new environment to avoid side effects.

#. Activate the conda environment::

      conda activate teacups

#. Install necessary packages::

      conda install numpy scipy matplotlib tqdm

#. Add the cloned directory to your pythonpath::
    
      .../teacups/src/
	
   .. note::
   
      You can add the path e.g. in an IDE like Spyder in which you want to run the code.

