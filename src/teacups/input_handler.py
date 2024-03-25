import numpy as np
from copy import deepcopy
import teacups.grid as gri


COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


class Calculations:
    """
    A Calculation object contains nothing, all attributes are empty. They are
    filled by several functions during the simulation.

    Attributes
    ----------
    cal.g1_tensor : object
        g1-tensor of the radical pair. This is a Tensor object.
        Can be set up by creators.set_up_tensors.
    cal.g2_tensor : object
        g2-tensor of the radical pair. This is a Tensor object.
        Can be set up by creators.set_up_tensors.
    cal.D_tensor : object
        D-tensor of coupled electrons. This is a Tensor object.
        Can be set up by creators.set_up_tensors.
    cal.g_tri_tensor : object
        g-tensor of the triplet precursor. This is a Tensor object.
        Can be set up by creators.set_up_tensors.
    cal.D_tri_tensor : object
        D-Tensor of the triplet precursor. This is a Tensor object.
        Can be set up by creators.set_up_tensors.
    cal.g_tensor : object
        g-Tensor of a radical. This is a Tensor object.
        Can be set up by creators.set_up_tensors.
    cal.g_iso : float
        Isotropic g-value.
        Can be set up by creators.set_up_tensors.
    cal.s : object
        Spinoperator of the spin system. This is a Spinoperator object.
        Can be set up by creators.set_up_spinoperator.
    cal.observable : np.ndarray
        Observable operator (S_y-operator in ST-basis) either as a matrix
        (calculations in hilbert-space) or as a vector (calculations in
        liouville space).
        Can be set up by creators.set_up_observable.
    cal.rho : np.ndarray
        The density matrix array in the basis of the systems Hamiltonian.
        Can be set up by density_matrices.set_up_density_matrix.
    cal.ham_mw : np.ndarray
        Microwave coupling hamiltonian in the rotating frame.
        Can be set up by hamiltonians.set_up_mw_hamiltonian.
    cal.ham_sys : np.ndarray
        Spin system hamiltonian. Can be set up by diverse functions from the
        hamiltonian module.
    cal.ham_sys : np.ndarray
        Full interaction hamiltonian. Sum of all hamiltonians.
    cal.ham_superop : np.ndarray
        The attribute will be built, if opt.space is set to liouville. This is
        the commutator superoperator. It can be set up by
        hamiltonians.set_up_commutator_superoperator.
    cal.A_tensor : list
        List of tensor objects one for each hyperfine tensor given in sys.
        Can be created b hyperfine.set_up_hyperfine_tensors.
    cal.spec_sim : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z. It can be set up by the make_signal function from
        hyperfine or signals_and_processing modules.
    cal.t : np.ndarray
        Linear spaced 1D-array with time points at which the spectrum
        will be simulated. Can be created by input_handler.predifinitions.
    cal.phi : np.ndarray
        Array with the phi values for each angle point. Can be created by
        input_handler.create_grid.
    cal.theta : np.ndarray
        Array with the theta values for each angle point. Can be created by
        input_handler.create_grid.
    cal.propagation : np.ndarray
        Time propagation operator. Can be set up by
        signals_and_processing.propagation.
    cal.signal : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z and all orientation points.
        It can be created by signals_and_processing.make_signal.
    cal.eigval : np.ndarray
        Eigenvalues of the spin system.
    cal.eigvec : np.ndarray
        Corresponding eigenvectors of the spin system.
    cal.pop_evolution : np.ndarray
        Population evolution of the spin system.


    """

    def __init__(self):
        self.g1_tensor = None
        self.g2_tensor = None
        self.D_tensor = None
        self.g_tri_tensor = None
        self.D_tri_tensor = None
        self.g_tensor = None
        self.g_iso = None

        self.s = None

        self.observable = None
        self.rho = None

        self.ham_mw = None
        self.ham_sys = None
        self.h = None
        self.ham_superop = None

        self.A_tensor = None

        self.spec_sim = None
        self.t = None

        self.theta = None
        self.phi = None

        self.propagation = None
        self.signal = None

        self.eigval = None
        self.eigvec = None

        self.pop_evolution = None


def input_object_handler(Sys: object, Exp: object, Opt: object
                         ) -> tuple[object, object, object, object]:
    """
    Build copys of the three input objects, that contain all simulation
    parameters. Create a Calculation object.

    Parameters
    ----------
    Sys : object
        This is a Spinsystem object, that should contain all spinsystem
        parameters needed for a simulation.
    Exp : object
        This is an Experiment object, that should contain all experimental
        parameters needed for a simulation.
    Opt : object
        This is an Options object, that should contain all simulation option
        parameters needed for a simulation.

    Returns
    -------
    sys : object
        Copy of Sys.
    exp : object
        Copy of Exp.
    opt : object
        Copy of Opt.
    cal : object
        Freshly initialized object from the class Calculations. This is needed
        for saving results during the simulation.

    """
    sys = deepcopy(Sys)
    exp = deepcopy(Exp)
    opt = deepcopy(Opt)
    cal = Calculations()
    return sys, exp, opt, cal


def scale_inputs(sys: object, exp: object, opt: object) -> None:
    """
    Change the unit of the attributes of the input to Hz or T.

    Parameters
    ----------
    sys : object
        Parameters of the spinsystem. All coupling parameters are
        changed to Hz.
    exp : object
        Experimental parameters. The microwave frequency is change to Hz,
        all magnetic parameters are changed to T.
    opt : object
        Simulation options object. If the sophe grid is chosen, the gaussian
        line width for interpolation is scaled.
    """

    # MHz -> Hz
    try:
        sys.J_ex *= 1e6
    except AttributeError:
        pass
    try:
        sys.D *= 1e6
    except AttributeError:
        pass
    try:
        sys.E *= 1e6
    except AttributeError:
        pass
    try:
        sys.D_tri *= 1e6
    except AttributeError:
        pass
    try:
        sys.E_tri *= 1e6
    except AttributeError:
        pass
    try:
        sys.A = [np.array(a)*1e6 for a in sys.A]
    except AttributeError:
        pass

    # mT -> T
    exp.B_z /= 1e3
    exp.B_mw /= 1e3
    sys.width_gauss /= 1e3

    # gaussian line width -> standard deviation
    std = sys.width_gauss/(2*np.sqrt(2*np.log(2)))
    # standard deviation has to be converted to the width of the magnetic field
    # array
    conversion = (exp.B_z.max() - exp.B_z.min()) / exp.B_z.shape[0]
    std = std/conversion
    sys.width_gauss = std

    if opt.grid == 'sophe':
        std = opt.width_intp/(2*np.sqrt(2*np.log(2)))
        conversion = (exp.B_z.max() - exp.B_z.min()) / exp.B_z.shape[0]
        std = std/conversion
        opt.width_intp = std

    return None


def predefinitions(sys, exp: object, cal: object) -> None:
    """
    Predefine arrays needed for the calculations: A linear spaced t-axis,
    an empty signal-array and a hyperfine intensity attribute.

    Parameters
    ----------
    exp : object
        Contains experimental parameters. This function uses exp.B_z,
        exp.t_scale and  exp.t_points.

    cal : object
        Container for calculated spaces.

    Attributes
    ----------
    cal.t : np.ndarray
        Linear spaced 1D-array with time points at which the spectrum
        will be simulated.
    cal.spec_sim : np.ndarray
        Array contains only zeros and will be filled later. The shape is
        nPoints x tPoints.
    cal.intensity : float
        The attribute intensity is created and set to one. It is changed if
        hyperfinecouplings are added analytically. Otherwise the intensity
        factor remains one.

    Returns
    -------
    None.

    """

    cal.t = np.linspace(exp.t_scale[0], exp.t_scale[1], exp.t_points,
                        dtype=FLOAT_TYPE)
    cal.spec_sim = np.zeros((exp.t_points, len(exp.B_z)), dtype=COMPLEX_TYPE)

    return None


def initialize_spin_system(sys: object) -> None:
    """
    Set up the spin quantum numer s for a given spinsystem.

    Parameters
    ----------
    sys : object
        Object of class spin system. The needed attribute spin_system is a
        string describing the type of the spin system. This is either 'rp'
        for a radical pair, 'doub' for a doublet or 'trip' for a triplet.

    Attributes
    ----------
    sys.s : list
        This list contains the spin_quantum numbers of the spin system.

    Returns
    -------
    None

    """
    if sys.spin_system == 'rp':
        sys.s = [1/2, 1/2]
    elif sys.spin_system == 'doub':
        sys.s = [1/2]
    elif sys.spin_system == 'trip':
        sys.s = [1]
    elif sys.spin_system == 'tdp':
        sys.s = [1/2, 1]

    return None


def create_grid(opt: object, cal: object) -> None:
    """
    Create the orientational grid and set up pairs of theta and phi for each
    angle point. The variable grid_points is changed from the input value
    (Number of points between theta=0 and theta=pi/2) to the number of angle
    points. A fibonacci grid or a sophe grid either is built, or a grid
    consisting of user-chosen angle points.

    Parameters
    ----------
    opt : object
        Object of class Options. The attribute opt.grid sets the chosen
        grid. It may be 'sophe' or 'fibonacci' either.
        It has to contain the attribute opt.grid_points which is the number of
        points between theta=0 and theat=pi/2 on the orientational sphere that
        shall be calculated.
        If opt.grid is set to 'single' user-chosen orientations can be
        calculated. In that case opt.theta and opt.phi have to be given as
        lists containing the desired angle inputs.
    cal : object
        Container for calculated values.

    Attributes
    ----------
    cal.phi : np.ndarray
        Array with the phi values for each angle point.
    cal.theta : np.ndarray
        Array with the theta values for each angle point.
    opt.grid_points : int
        Number of angle points.

    Returns
    -------
    None

    """
    if opt.grid == 'sophe':
        cal.phi, cal.theta, w = gri.sophe_grid(opt.grid_points)
        opt.grid_points = len(cal.phi)
        opt.width_intp /= 1e3

    elif opt.grid == 'fibonacci':
        cal.theta, cal.phi = gri.get_theta_phi(
            int(opt.grid_points + 4*opt.grid_points*(opt.grid_points-1)/2))
        opt.grid_points = len(cal.phi)

    elif opt.grid == 'single':
        cal.theta = np.array(opt.theta)
        cal.phi = np.array(opt.phi)
        opt.grid_points = len(cal.phi)
    return


def hyperfine_converter(sys: object) -> None:
    """
    Allow flexible input of hyperfine parameters. Function has the following
    possibilities for handling hyperfine parameters:

    - No hyperfines are given: Only sys.I is set as an empty list. No further
    hyperfine correlated attributes are added to the sys object.

    - Hyperfines are given in the spinsystem attributes sys.I, sys.A and
    sys.A_frame as lists (like the lists like needed by
    set_up_hyperfine_tensors and create_hf_hamiltonian): Function just returns;
    it does not change the attributes.

    - Hyperfines are given as numbered attributes of sys, e.g. sys.I1, sys.I2
    etc., the spin_system is not 'rp': Attributes sys.A, sys.I and sys.A_frame
    are build as lists (like needed by set_up_hyperfine_tensors and
    create_hf_hamiltonian) and filled with all n numbered attributes. If no
    sys.A_frame_i is given the values in sys.A_frame are set to [0, 0, 0]. The
    attribute sys.n_i determines the number of times the type of core is added
    to sys.A/I/A_frame.

    - Hyperfines are given as numbered attributes of sys, e.g. sys.I1, sys.I2
    etc. and the spin_system is 'rp': The attributes are created as if
    spin_sytem would not be 'rp' but the resulting lists contain two instead
    of one list of hyperfine parameters, one for each electron. The place where
    the numbered elements shall be placed is defined by the attributes
    donor_list and acceptor list.

    Parameters
    ----------
    sys : object
        Spin system object. May contain parameters for the hyperfine coupling.

    Returns
    -------
    None

    Examples
    --------
    >>> sys = Sys()
    >>> sys.spin_system = 'doub'
    >>> sys.I1 = 1/2
    >>> sys.n1 = 2
    >>> sys.A1 = [2, 2, 2]
    >>> hyperfine_converter(sys)
    >>> sys.A
    [[[2, 2, 2], [2, 2, 2]]]
    >>> sys.I
    [[1/2, 1/2]]
    >>> sys.A_frame
    [[[0, 0, 0], [0, 0, 0]]]

    >>> sys.spin_system = 'rp'
    >>> sys.I1 = 1/2
    >>> sys.n1 = 1
    >>> sys.A1 = [1, 2, 3]
    >>> sys.I2 = 1
    >>> sys.n2 = 1
    >>> sys.A2 = [4, 5, 6]
    >>> sys.A2_frame = [7, 7, 7]
    >>> sys.acceptor_list = [1]
    >>> sys.donor_list = [2]
    >>> hyperfine_converter(sys)
    >>> sys.I
    [[1/2], [1]]
    >>> sys.A
    [[[1, 2, 3]], [[4, 5, 6]]]
    >>> sys.A_frame
    [[[0, 0, 0]], [[7, 7, 7]]]

    """
    all_hyperfines = [nucspin for nucspin in vars(sys).keys()
                      if nucspin.startswith("I")]

    if not all_hyperfines:
        sys.I = []
        return

    else:
        if 'I' in all_hyperfines:
            return
        else:
            if not sys.spin_system == 'rp':
                A = []
                A_frame = []
                I = []

                for nuc in all_hyperfines:
                    num = nuc[-1]
                    for n in range(vars(sys)["n"+str(num)]):
                        A.append(vars(sys)["A"+str(num)])
                        I.append(vars(sys)["I"+str(num)])
                        try:
                            A_frame.append(vars(sys)["A"+str(num)+"_frame"])
                        except KeyError:
                            A_frame.append([0, 0, 0])

                sys.A = [A]
                sys.I = [I]
                sys.A_frame = [A_frame]
                return

            else:
                A_d = []
                A_a = []
                A_d_frame = []
                A_a_frame = []
                I_d = []
                I_a = []
                for d in sys.donor_list:
                    for n in range(vars(sys)["n"+str(d)]):
                        A_d.append(vars(sys)["A"+str(d)])
                        I_d.append(vars(sys)["I"+str(d)])
                        try:
                            A_d_frame.append(vars(sys)["A"+str(d)+"_frame"])
                        except KeyError:
                            A_d_frame.append([0, 0, 0])

                for a in sys.acceptor_list:
                    for n in range(vars(sys)["n"+str(a)]):
                        A_a.append(vars(sys)["A"+str(a)])
                        I_a.append(vars(sys)["I"+str(a)])
                        try:
                            A_a_frame.append(vars(sys)["A"+str(a)+"_frame"])
                        except KeyError:
                            A_a_frame.append([0, 0, 0])

                sys.A = [A_a, A_d]
                sys.I = [I_a, I_d]
                sys.A_frame = [A_a_frame, A_d_frame]
                return
