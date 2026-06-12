import numpy as np
import itertools as it

import teacups.signals_and_processing as sap
import teacups.input_handler as inputs
import teacups.hyperfine as hf
import teacups.convolution as co
import teacups.creators as cr
import teacups.hamiltonians as ham
import teacups.density_matrices as dm


def teacups(Sys: object, Exp: object, SimOpt: object) -> "np.ndarray":
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

    Returns
    -------
    intensity : np.ndarray
        2D Array containing the simulated intensities in arbitrary units. One
        dimension is the magnetic field, the other is the time.
    pop_evolution : np.ndarray
        Population_evolution is optionally given back if SimOpt.pop_evolution
        is True and SimOpt.space is 'liouville'. 2D Array that contains the
        simulated populations for the first orientation and the first magnetic
        field point for every time point in arbitrary units.
    eigval: np.ndarray
        2D Array optionally given back if SimOpt.eigval_mode is True.
        Ends the calculation without continuing to intensities or time
        evolutions. Contains the eigenvalues of the system hamiltonian in Hz.
        One dimension is the magnetic field, then grid points.

    """

    sys, exp, opt, cal = inputs.input_object_handler(Sys, Exp, SimOpt)

    inputs.hyperfine_converter(sys)

    # scale input parameters
    inputs.scale_inputs(sys, exp, opt)

    # create spaces
    inputs.predefinitions(sys, exp, cal)

    # initialize spinsystem
    inputs.initialize_spin_system(sys)

    # initialize the grid
    inputs.create_grid(opt, cal)

    # split the grid if not enough memory available
    inputs.split_grid(sys, exp, opt, cal)

    # do simulation for each part of the grid
    for chunk in range(len(cal.phi_split)):
        print(chunk)
        cal.phi = cal.phi_split[chunk]
        cal.theta = cal.theta_split[chunk]
        if opt.grid == "sophe":
            cal.weights = cal.weights_split[chunk]
        opt.grid_points = len(cal.phi)

        # create spin operator and detection operator
        cr.set_up_spinoperator(sys, cal)
        cr.set_up_observable(sys, opt, cal)

        # set up tensors
        cr.set_up_tensors(sys, cal)

        # set up spin system hamiltonian
        if sys.spin_system == "rp":
            cal.ham_sys = ham.set_up_rp_hamiltonian(sys, exp, opt, cal)
        elif sys.spin_system == "trip":
            cal.ham_sys = ham.set_up_triplet_hamiltonian(exp, opt, cal)
        elif sys.spin_system == "doub":
            cal.ham_sys = ham.set_up_doublet_hamiltonian(exp, opt, cal)
        elif sys.spin_system == "tdp":
            cal.ham_sys = ham.set_up_tdp_hamiltonian(sys, exp, opt, cal)

        # add microwave interaction
        cal.ham = cal.ham_sys + ham.set_up_mw_hamiltonian(sys, exp, opt, cal)

        # Stop simulation routine if only the eigenvalues are desired
        if opt.eigval_mode is True:
            eigval, eigvec = np.linalg.eigh(cal.ham_sys)
            return eigval

        # set up initial density matrix
        dm.set_up_density_matrix(sys, exp, opt, cal)

        # clear memory
        keys = vars(cal).copy()
        for key in keys:
            if key.endswith("_tensor"):
                delattr(cal, key)
        delattr(cal, "g_iso")

        # calculate eigenvalues and eigenvectors of the spinsystem if needed
        if opt.space == "liouville":
            cal.eigvec = np.linalg.eigh(cal.ham_sys)[1]

        # clear memory
        delattr(cal, "ham_sys")

        # build the signal including the hyperfine interactions if any nuclei
        # are given
        if list(it.chain(*sys.I)):
            hf.set_up_hyperfine_tensors(sys, cal)

            cal.ham_hf = hf.create_hf_hamiltonian(sys.s, sys.I, cal.A_tensor)

            # create signal
            hf.make_signal_with_hyperfine(sys, exp, opt, cal)

        # build signal if no coupling nuclei are given
        else:
            # build commutator superoperator (in Liouville space)
            ham.set_up_commutator_superoperator(sys, opt, cal)

            # build propagation operator
            print("starting the propagation...")
            sap.propagation(sys, opt, cal)

            # clear memory
            delattr(cal, "ham")
            if opt.space == "liouville":
                delattr(cal, "ham_superop")

            # time propagation
            print("start making the signal...")
            sap.make_signal(exp, opt, cal)

            # clear memory
            delattr(cal, "theta")
            delattr(cal, "phi")
            delattr(cal, "s")
            delattr(cal, "observable")
            delattr(cal, "rho")
            delattr(cal, "propagation")
            if opt.space == "liouville":
                delattr(cal, "eigvec")

            # build powder average
            sap.powder_average(opt, cal)

    # do Voigt convolution
    cal.spec_sim = co.voigt_convolution(
        sys.sigma_time, sys.width_gauss, cal.spec_sim, opt.extend_t
    )

    cal.t = co.extend_time_axis(cal.t, cal.spec_sim)
    Exp.t = cal.t

    # calculate decay of the signal in hilbert space
    if SimOpt.space == "hilbert":
        sap.signal_hilbert_decay(sys, cal)

    # return result
    if SimOpt.pop_evolution is True and SimOpt.space == "liouville":
        return cal.spec_sim, cal.pop_evolution
    else:
        return cal.spec_sim
