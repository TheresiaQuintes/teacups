import numpy as np
import itertools as it

import teacups.signals_and_processing as sap
import teacups.input_handler as inputs
import teacups.hyperfine as hf
import teacups.matrix_tools as mt
import teacups.convolution as co
import teacups.creators as cr
import teacups.hamiltonians as ham
import teacups.density_matrices as dm

import matplotlib.pyplot as plt


def teacups(Sys: object, Exp: object, SimOpt: object, development=False
            ) -> 'np.ndarray':
    """
    Simulate a 2D transient EPR spectrum of a spinpolarized spin system.

    Parameters
    ----------
    Sys : object
        Spinsystem object, that contains parameters of the spin system.
        Attributes can be given to this object in the following syntax:
        sys.attribute = ?.
    Exp : object
        Experiment object, that contains parameters of the experiment.
        Attributes can be given to this object in the following syntax:
        sys.attribute = ?.
    SimOpt : object
        Simulationoptions object, that contains simulation option parameters.
        Attributes can be given to this objects in the following
        syntax: sys.attribute = ?.
    development : boolean, optional
        If development is set to True, the whole Calculation object
        (initialized and filled during the simulation) will be given back
        including all interim results, instead of the signal-array.

    Returns
    -------
    intensity : np.ndarray
        2D Array containing the simulated intensities in arbitrary units. One
        dimension is the magnetic field, the other is the time.
    pop_evolution : np.ndarray
        Population_evolution is optionally given back if SimOpt.pop_evolution
        is True and SimOpt.space is 'liouville'.

    """

    sys, exp, opt, cal = inputs.input_object_handler(Sys, Exp, SimOpt)

    inputs.hyperfine_converter(sys)

    # scale input parameters
    inputs.scale_inputs(sys, exp, opt)

    # create spaces
    inputs.predefinitions(sys, exp, cal)

    # initialize spinsystem
    inputs.initialize_spin_system(sys)

    inputs.create_grid(opt, cal)

    # create spin operator and detection operator
    cr.set_up_spinoperator(sys, cal)
    cr.set_up_observable(sys, opt, cal)

    # set up tensors
    cr.set_up_tensors(sys, cal)

    # set up hamiltonian

    if sys.spin_system == 'rp':
        cal.ham_sys = ham.set_up_rp_hamiltonian(sys, exp, opt, cal)
        ham_mw = ham.set_up_mw_hamiltonian(sys, exp, opt, cal)
        cal.ham = cal.ham_sys + ham_mw

    elif sys.spin_system == 'trip':
        # set up secular high-field triplet hamiltonian
        cal.ham_sys = ham.set_up_triplet_hamiltonian(exp, opt, cal)
        cal.ham_mw = ham.set_up_mw_hamiltonian(sys, exp, opt, cal)
        cal.ham = cal.ham_sys + cal.ham_mw

        # set up (non-secular) zero-field and high-field Hamiltonian without mw
        # interaction
        cal.ham_tri_zf = ham.set_up_triplet_zero_field_hamiltonian(
            exp, opt, cal)
        cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            exp, opt, cal)

    elif sys.spin_system == 'doub':
        cal.ham_sys = ham.set_up_doublet_hamiltonian(exp, opt, cal)
        cal.ham_mw = ham.set_up_mw_hamiltonian(sys, exp, opt, cal)
        cal.ham = cal.ham_sys + cal.ham_mw

    elif sys.spin_system == 'tdp':
        cal.ham_sys = ham.set_up_tdp_hamiltonian(sys, exp, opt, cal)
        cal.ham_mw = ham.set_up_mw_hamiltonian(sys, exp, opt, cal)
        cal.ham = cal.ham_sys + cal.ham_mw

    # Stop simulation routine if only the eigenvalues are desired
    if opt.eigval_mode is True:
        eigval, eigvec = np.linalg.eigh(cal.ham_sys)
        return eigval

    # set up initial density matrix
    if sys.precursor == 'triplet-zf' or sys.precursor == 'triplet-eigen':
        cal.s_tri = mt.Spinoperator(1)
        cal.ham_tri_hf = ham.set_up_triplet_high_field_hamiltonian(
            exp, opt, cal)
        cal.ham_tri_zf = ham.set_up_triplet_zero_field_hamiltonian(
            exp, opt, cal)

    dm.set_up_density_matrix(sys, exp, opt, cal)

    # clear memory
    keys = vars(cal).copy()
    for key in keys:
        if key.endswith('_tensor'):
            delattr(cal, key)

    # calculate eigenvalues and eigenvectors of the spinsystem if needed
    if opt.space == 'liouville':
        cal.eigval, cal.eigvec = np.linalg.eigh(cal.ham_sys)

    # clear memory
    delattr(cal, 'ham_sys')

    # build the signal including the hyperfine interactions if any nuclei are
    # given
    if list(it.chain(*sys.I)):
        hf.set_up_hyperfine_tensors(sys, cal)

        cal.ham_hf = hf.create_hf_hamiltonian(sys.s, sys.I, cal.A_tensor)

        # create signal
        hf.make_signal_with_hyperfine(sys, exp, opt, cal)

    # build signal if no coupling nuclei are given
    else:
        ham.set_up_commutator_superoperator(sys, opt, cal)

        print('starting the propagation...')
        sap.propagation(sys, opt, cal)

        # clear memory
        keys = vars(cal).copy()
        for key in keys:
            if key.startswith('ham'):
                delattr(cal, key)

        print('start making the signal...')
        sap.make_signal(exp, opt, cal)

        sap.powder_average(exp, opt, cal)

    # do Voigt convolution
    cal.spec_sim = co.voigt_convolution(sys.width_gauss, cal.spec_sim)

    # calculate decay of the signal in hilbert space
    if SimOpt.space == 'hilbert':
        sap.signal_hilbert_decay(sys, cal)

    # returns
    if development:
        return cal
    else:
        if SimOpt.pop_evolution is True and SimOpt.space == 'liouville':
            return cal.spec_sim, cal.pop_evolution
        else:
            return cal.spec_sim
