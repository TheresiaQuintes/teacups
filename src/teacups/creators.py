import teacups.matrix_tools as mt
import numpy as np
COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


def create_tensor(diag: list, phi: 'np.ndarray', theta: 'np.ndarray',
                  first_rotation=list()) -> object:
    """
    Setup any tensor. An object of class mt.Tensor is built. Diagonal elements
    are filled into the tensor in its diagonal coordinate frame. If
    first_rotation is not empty, an initial rotation to laboratory frame will
    be carried out. Afterwards the tensor.multirot attribute is built by using
    the tensor.multirotation function with the angle-arrays phi and theta.

    Parameters
    ----------
    diag : list
        A list of the three diagonal elements of the tensor.
    phi : np.ndarray
        Array with the phi values for each angle point.
    theta: np.ndarray
        Array with the theta values for each angle point.
    first_rotation : list
        If a first rotation into laboratory frame is wished, first_rotation
        must be set to a list of the three euler angles phi, theta and psi
        around which the tensor is rotated. If it is empty, no rotation will be
        carried out. Default is an empty list.

    Returns
    -------
    tensor : object
        Tensor object with initialized tensor and multirot.

    """
    tensor = mt.Tensor(np.array(diag))

    if len(first_rotation) == 0:
        pass
    else:
        angle_list = np.array(first_rotation, dtype=FLOAT_TYPE)
        tensor.rotation(angle_list[0], angle_list[1], angle_list[2])
        tensor.matrix = tensor.rot.astype(FLOAT_TYPE)

    tensor.multirotation(phi, theta)
    tensor.multirot = tensor.multirot.astype(FLOAT_TYPE)
    return tensor


def create_zfs_tensor_diagonals(D: float, E: float) -> 'np.ndarray':
    """
    Create an array with three diagonal elements of the zero-field splitting
    (ZFS) tensor. Calculate this values from the zero-field-splitting
    parameters D and E by:

    .. math::
       -1/3*D+E; -1/3*D-E; 2/3*D.

    Parameters
    ----------
    D : float
        Zero-field-splitting parameter D.
    E : float
        Zero-field-splitting parameter E.

    Returns
    -------
    diag : np.ndarray
        The three diagonal elements of a ZFS tensor (-1/3*D+E, -1/3*D-E, 2/3*D).
        This is a 1D-array.

    """
    diag = np.array([-1/3*D+E, -1/3*D-E, 2/3*D], dtype=FLOAT_TYPE)
    return diag


def create_dipol_tensor_diagonals(D: float, E: float) -> 'np.ndarray':
    """
    Create an array with three diagonal elements of the dipolar interaction
    tensor of two spin species. Calculate this values from the
    axial dipolar coupling $D$ and the rhombic dipolar coupling $E$ by:

    .. math::
       D+E; D-E; -2D.

    Parameters
    ----------
    D : float
        Axial dipolar coupling.
    E : float
        Rhombic dipolar coupling.

    Returns
    -------
    diag : np.ndarray
        The three diagonal elements of a dipol tensor (D+E, -D-E, -2D).
        This is a 1D-array.

    """
    diag = np.array([D+E, D-E, -2*D], dtype=FLOAT_TYPE)
    return diag


def set_up_tensors(sys: object, cal: object) -> None:
    """
    Set up all wished tensors for further calculations. The tensors are set up
    for all orientations and are saved as tensor objects.
    Furthermore, if the frame argument in sys is given, the tensors
    will be rotated to an initial frame.

    Parameters
    ----------
    sys : object
        Contains parameters concerning the spin system. This function can use
        the following arguments and returns the tensors:
        g1, g2 (g-tensors of a radical pair) + g1_frame/g2_frame [lists]
        g (g-tensor of a radical) + g_frame [lists]
        g_tri (g-tensor of a triplet) + g_tri_frame [lists]
        D, E (dipole coupling parameters of a coupled spin system) [floats]
        + D_frame [list], if D, E are not given, they will be set to zero
        D_tri, E_tri (ZFS parameters of a triplet) [floats]
        + D_tr_frame [list], if D, E are not given, they will be set
        to zero
    cal : object
        Container for calculated tensors. The attributes cal.phi and cal.theta
        (arrays with the angle points on a sphere for rotations) are needed.

    Attributes
    ----------
    cal.g1_tensor : object
        g1-tensor of the radical pair. This is a Tensor object. (optional)
    cal.g2_tensor : object
        g2-tensor of the radical pair. This is a Tensor object. (optional)
    cal.D_tensor : object
        D-tensor of coupled electrons. This is a Tensor object. (optional)
    cal.g_tri_tensor : object
        g-tensor of the triplet precursor. This is a Tensor object.
    cal.D_tri_tensor : object
        D-Tensor of the triplet precursor. This is a Tensor object.
    cal.g_tensor : object
        g-Tensor of a radical. This is a Tensor object. (optional)
    cal.g_iso : float
        Isotropic g-value.

    Returns
    -------
    None.

    """
    if not hasattr(sys, 'D'):
        sys.D = 0
    if not hasattr(sys, 'E'):
        sys.E = 0
    if not hasattr(sys, 'D_tri'):
        sys.D_tri = 0
    if not hasattr(sys, 'E_tri'):
        sys.E_tri = 0.01

    for attr in vars(sys):
        if attr == 'g':
            if hasattr(sys, 'g_frame'):
                cal.g_tensor = create_tensor(sys.g, cal.phi, cal.theta,
                                             first_rotation=sys.g_frame)
            else:
                cal.g_tensor = create_tensor(sys.g, cal.phi, cal.theta)
            cal.g_iso = 1/3*np.sum(sys.g)

        elif attr == 'g1':
            if hasattr(sys, 'g1_frame'):
                cal.g1_tensor = create_tensor(sys.g1, cal.phi, cal.theta,
                                              sys.g1_frame)
            else:
                cal.g1_tensor = create_tensor(sys.g1, cal.phi, cal.theta)

        elif attr == 'g2':
            if hasattr(sys, 'g2_frame'):
                cal.g2_tensor = create_tensor(sys.g2, cal.phi, cal.theta,
                                              sys.g2_frame)
            else:
                cal.g2_tensor = create_tensor(sys.g2, cal.phi, cal.theta)
            cal.g_iso = 1/2*(1/3*np.sum(sys.g1) + 1/3*np.sum(sys.g2))

        elif attr == 'g_tri':
            if hasattr(sys, 'g_tri_frame'):
                cal.g_tri_tensor = create_tensor(
                    sys.g_tri, cal.phi, cal.theta, sys.g_tri_frame)
            else:
                cal.g_tri_tensor = create_tensor(sys.g_tri, cal.phi, cal.theta)
            if hasattr(sys, 'g2'):
                pass
            elif hasattr(sys, 'g'):
                cal.g_iso = 1/2*(1/3*np.sum(sys.g_tri)+1/3*np.sum(sys.g))
            else:
                cal.g_iso = 1/3*np.sum(sys.g_tri)

        elif attr == 'D_tri':
            diag = create_zfs_tensor_diagonals(sys.D_tri, sys.E_tri)
            if hasattr(sys, 'D_tri_frame'):
                cal.D_tri_tensor = create_tensor(
                    diag, cal.phi, cal.theta, sys.D_tri_frame)
            else:
                cal.D_tri_tensor = create_tensor(diag, cal.phi, cal.theta)

        elif attr == 'D':
            diag = create_dipol_tensor_diagonals(sys.D, sys.E)
            if hasattr(sys, 'D_frame'):
                cal.D_tensor = create_tensor(
                    diag, cal.phi, cal.theta, sys.D_frame)

            else:
                cal.D_tensor = create_tensor(diag, cal.phi, cal.theta)

    return None


def set_up_spinoperator(sys: object, cal: object) -> None:
    """
    Set up the spin matrix operators for the spin system.

    Parameters
    ----------
    sys : object
        Contains parameters of the spin system. This function needs sys.s which
        is a list of the quantum numbers of the spin system.
    cal : object
        Container for calculated spinoperators. This object may be empty.

    Attributes
    ----------
    cal.s : object
        Spinoperator of the spin system. This is a Spinoperator object.

    Returns
    -------
    None.

    """
    try:
        len(sys.s)
    except TypeError:
        sys.s = [sys.s]

    if len(sys.s) == 1:
        S = mt.Spinoperator(sys.s[0])
    elif len(sys.s) == 2:
        S = mt.Spinoperator(sys.s[0], sys.s[1])
        S.matrix = S.matrix + S.matrix_coupling_spins[0]

    cal.s = S

    return None


def set_up_observable(sys: object, opt: object, cal: object) -> None:
    """
    Set up the observable operator for an EPR-Experiment in the singlet-triplet
    basis of the radical pair. Observation direction is y-direction (in case
    of the static magnetic field set in z-direction). The total spin operator
    of a radical pair is taken and transforem to ST-basis. For calculations in
    hilbert-space the operator is given back as a 4x4-matrix. For calculations
    in liouville-space it is given back as a 16x1 vector.

    Parameters
    ----------
    opt : object
        Contains simulation option parameters. This function uses opt.space
        (set either to 'hilbert' or 'liouville'), which defines the space in
        which the calculation is carried out.
    cal : object
        Container for calculated results during the simulation. This function
        uses the total spinoperator cal.s (e.g. built by the function
        set_up_radical_pair_spinoperators).

    Attributes
    ----------
    cal.observable : np.ndarray
        Observable operator (S_y-operator in ST-basis) either as a matrix
        (calculations in hilbert-space) or as a vector (calculations in
        liouville space).

    Returns
    -------
    None.

    """

    obs = mt.Operator(cal.s.dimension)
    obs.matrix = cal.s.get('y')

    if sys.spin_system == 'rp':
        st_transformation = np.array([[1, 0, 0, 0],
                                      [0, np.sqrt(1/2), np.sqrt(1/2), 0],
                                      [0, -np.sqrt(1/2), np.sqrt(1/2), 0],
                                      [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        obs.matrix = st_transformation.T@obs.matrix@st_transformation

    obs.matrix = np.conjugate(obs.matrix.T)

    if opt.space == 'hilbert':
        cal.observable = obs.matrix
    elif opt.space == 'liouville':
        obs.build_vector()
        cal.observable = obs.vector
    else:
        print('opt.space has to be either lioville or hilbert')

    return None
