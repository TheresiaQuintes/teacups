import numpy as np
import teacups.multioperator_tools as mut
import teacups.matrix_tools as mt
import teacups.relaxation as rlx

import scipy.constants as const
MU_B = const.physical_constants['Bohr magneton in Hz/T'][0]
COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


def set_up_mw_hamiltonian(sys: object, exp: object, opt: object, cal: object
                          ) -> 'np.ndarray':
    """
    Calculate a Hamiltonian matrix operator for the coupling of a spin system
    with a weak microwave field in x-direction. The Hamiltonian is calculated
    for all orientations and magnetic field points and returned as a
    B_angle_matrix attribute from the class Multioperator.
    Secular approximation is presumed. By a shift of the lamor frequencies the
    Hamitlonian is transformed to the rotating frame. Therefore, the microwave
    Hamiltonian is not time dependent. In case the spin system is a radical
    pair the Hamiltonian is transferred to the singlet triplet basis.
    Otherwise it is left in the product basis.

    Parameters
    ----------
    sys : object
        Contains spinsystem parameters. This function uses the attributes
        sys.spin_system.
    exp : object
        Contains experimental parameters. This function uses the attributes
        exp.B_mw (strength of microwave field) and exp.freq_mw
        (microwave frequency).
    opt : object
        Contains simulation options. This function uses opt.grid_points.
    cal : object
        Container for results of calculations during the simulation. This
        function uses cal.g_iso (e.g. from the function set_up_tensors) and
        the spinoperator object cal.s.

    Returns
    -------
    ham_mw : np.ndarray
        Microwave coupling hamiltonian in the rotating frame. This is a
        B_angle_matrix attribute from the class Multioperator.

    """
    if sys.spin_system == 'rp':
        sqrt_2 = 1/np.sqrt(2)
        T = np.array([[1, 0, 0, 0], [0, sqrt_2, sqrt_2, 0],
                      [0, -sqrt_2, sqrt_2, 0], [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        sx = T.T @ cal.s.get('x') @ T
        sx /= sqrt_2
        sz = T.T @ cal.s.get('z') @ T

    else:
        sx = cal.s.get('x')
        sz = cal.s.get('z')

    a = -1*sz*exp.freq_mw
    b = cal.g_iso*exp.B_mw*MU_B*sx

    ham = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
    ham.matrix = a+b
    ham.matrix_changed()

    if sys.spin_system == 'rp':
        ham.B_angle_matrix *= 2*np.pi

    return ham.B_angle_matrix


def set_up_doublet_hamiltonian(exp: object, opt: object, cal: object
                               ) -> 'np.ndarray':
    """
    Calculate a doublet Hamiltonian. The high-field-Hamiltonian is set up as
    a multioperator object for each orientation and each magnetic field point.
    The Zeeman-interaction is included. The spinfunctions alpha and beta are
    used as the basis set.
    As the Hamiltonian is set up in the rotating frane, secular approximation
    is applied.

    Parameters
    ----------
    exp : object
        Contains experimental parameters. This functions uses the attribute
        exp.B_z (magnetic field array).
    opt: object
        Contains simulation option parameters. This function uses the attribute
        opt.grid_points, which is the number of angle combinations that shall
        be caluclated for the powder average.
    cal : object
        Container for results of calculations during the simulation. This
        function uses the tensor object cal.g_tensor, which may be created by
        the function set_up_tensors and cal.s (spinoperator of the doublet).

    Returns
    -------
    ham : np.ndarray
        Doublet high field hamiltonian. This is a B_angle_matrix attribute from
        the class Multioperator.

    """
    ham = mut.Multioperator(cal.s, opt.grid_points, exp.B_z*MU_B)
    B_z = exp.B_z*MU_B
    B_z = B_z[:, np.newaxis]

    ham.B_angle_matrix[:, :, 0, 0] = 0.5*B_z * \
        cal.g_tensor.multirot[:, 2, 2]
    ham.B_angle_matrix[:, :, 1, 1] = -0.5*B_z * \
        cal.g_tensor.multirot[:, 2, 2]
    ham = ham.B_angle_matrix

    return ham


def set_up_triplet_hamiltonian(exp: object, opt: object, cal: object
                               ) -> 'np.ndarray':
    """
    Calculate the triplet Hamiltonian. The high-field-Hamiltonian is set up as
    a multioperator object for each grid point and for each magnetic field
    point. Interactions included are: Zeeman-interaction and dipolar
    interaction (=ZFS). The spinfunctions (-1, 0, +1) are used as a basis set.
    As the Hamiltonian is set up in the rotating frame, secular approximation
    is applied:

    .. math::
        H = D_{zz} * (S_z^2 - 1/3*S^2) + H_\mathrm{Zeeman}

    Parameters
    ----------
    exp : object
        Contains experimental parameters. This function uses the attribute
        exp.B_z (magnetic field array).
    opt : object
        Contains simulation options. This function uses opt.grid_points.
    cal : object
        Container for results of calculations during the simulation. This
        function uses the tensor objects cal.g_tri_tensor and cal.D_tri_tensor.
        They may be created by the function set_up_tensors. Further cal.s
        (spinoperator of the triplet) is used.

    Returns
    -------
    ham : np.ndarray
        Triplet high field hamiltonian. This is a B_angle_matrix attribute from
        the class Multioperator.

    """
    ham = mut.Multioperator(cal.s, opt.grid_points, exp.B_z*MU_B)

    B_z = exp.B_z*MU_B
    B_z = B_z[:, np.newaxis]

    ham.B_angle_matrix[:, :, 0, 0] = B_z*cal.g_tri_tensor.multirot[:, 2, 2] \
        + (1/2)*cal.D_tri_tensor.multirot[:, 2, 2]
    ham.B_angle_matrix[:, :, 1, 1] = -1*cal.D_tri_tensor.multirot[:, 2, 2]
    ham.B_angle_matrix[:, :, 2, 2] = -B_z*cal.g_tri_tensor.multirot[:, 2, 2]\
        + (1/2)*cal.D_tri_tensor.multirot[:, 2, 2]

    ham = ham.B_angle_matrix

    return ham


def set_up_triplet_high_field_hamiltonian(exp: object, opt: object,
                                          cal: object) -> 'np.ndarray':
    """
    Calculate the hamiltonian of a triplet state in high magnetic field in
    the D-tensor main-axis system (Tx, Ty, Tz). It is built as the sum of
    the ZFS and the Zeeman interaction.

    The triplet hamiltonian is calculated for different B_z-values and all
    combinations of the euler angels phi and theta (grid_points points
    distributed on the fibonacci sphere). All hamiltonians are returned in
    cal.ham_tri wich is an object of the class Multioperator.

    Parameters
    ----------
    exp : object
        Contains experimental parameters. This function uses exp.B_z
    opt : object
        Contains simulation option parameters. This function uses
        opt.grid_points.
    cal : object
        Container for results of calculations during the simulation. This
        function uses the attributes cal.D_tri_tensor and cal.g_tri_tensor
        (e.g.built by the function set_up_tensors).

    Returns
    -------
    ham_tri_hf : np.ndarray
        Hamiltonian of a triplet precursor of a radical pair in high field.
        This is a B_angle_matrix attribute from the class Multioperator.

    """
    s_tri = mt.Spinoperator(1)

    # ZFS-Hamiltonian
    ham_d = mut.Multioperator(s_tri, opt.grid_points, exp.B_z*MU_B)
    ham_d.create_bilinear_operator(cal.D_tri_tensor, s_tri)
    ham_d.angle_matrix_changed()

    # The eigenbasis of the ZFS-Hamiltonian is the xyz-Basis
    eig_d, vec_d = np.linalg.eigh(ham_d.B_angle_matrix)

    # Calculate high-field Hamiltonian
    ham_tri_hf = mut.Multioperator(s_tri, opt.grid_points, exp.B_z*MU_B)
    ham_tri_hf.zeeman_coupling(cal.g_tri_tensor)
    ham_tri_hf.B_angle_matrix -= ham_d.B_angle_matrix

    # Transfer high-field Hamiltonian to xyz-Basis
    ham_tri_hf.B_angle_matrix = np.conj(
        np.transpose(vec_d, (0, 1, 3, 2))) @ ham_tri_hf.B_angle_matrix @ vec_d
    ham_tri_hf = ham_tri_hf.B_angle_matrix

    return ham_tri_hf


def set_up_rp_hamiltonian(sys: object, exp: object, opt: object, cal: object
                          ) -> 'np.ndarray':
    """
    Calculate a Hamiltonian matrix operator for a radical pair. Coupling
    with the static magnetic field is included as well as interactions of the
    two spins. The Hamiltonian is calculated for all orientations and magnetic
    field points in the singlet-triplet-basis and returned as a
    B_angle_matrix attribute from the class Multioperator. Secular
    approximation is presumed.

    Parameters
    ----------
    sys : object
        Contains spinsystem parameters. This function uses the attributes
        sys.g1, sys.g2 and optional sys.J_ex (if given).
    exp : object
        Contains experimental parameters. This function uses the attribute
        exp.B_z (magnetic field array).
    opt : object
        Contains simulation options. This function uses opt.grid_points.
    cal : object
        Container for results of calculations during the simulation. This
        function uses cal.g1_tensor and cal.g2_tensor and optional cal.D_tensor
        if it is given (e.g. from set_up_tensors).

    Returns
    -------
    ham_rp : np.ndarray
        Radical pair hamiltonian in the singlet triplet basis. This is a
        B_angle_matrix attribute from the class Multioperator.

    """
    sum_g = 1/2*(cal.g1_tensor.multirot[:, 2, 2] +
                 cal.g2_tensor.multirot[:, 2, 2])
    difference_g = 1/2*(cal.g1_tensor.multirot[:, 2, 2] -
                        cal.g2_tensor.multirot[:, 2, 2])

    if hasattr(cal, 'D_tensor'):
        D = cal.D_tensor.multirot[:, 2, 2] * 2*np.pi
    else:
        D = 0
    if hasattr(sys, 'J_ex'):
        J_ex = sys.J_ex*2*np.pi
    else:
        J_ex = 0

    B_z = 2*np.pi*MU_B*exp.B_z
    B_z = B_z[:, np.newaxis]

    omega = B_z*sum_g
    delta_omega = B_z*difference_g

    ham = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
    ham.B_angle_matrix[:, :, 0, 0] = omega - J_ex - 1/2*D
    ham.B_angle_matrix[:, :, 1, 1] = J_ex
    ham.B_angle_matrix[:, :, 1, 2] = delta_omega
    ham.B_angle_matrix[:, :, 2, 1] = delta_omega
    ham.B_angle_matrix[:, :, 2, 2] = -J_ex + D
    ham.B_angle_matrix[:, :, 3, 3] = -omega - J_ex - 1/2*D

    ham_rp = ham.B_angle_matrix

    return ham_rp


def set_up_tdp_hamiltonian(sys: object, exp: object, opt: object, cal: object
                           ) -> 'np.ndarray':
    r"""
    Calculate a Hamiltonian matrix operator for a coupled triplet doublet pair.
    Coupling with the static magnetic field of the triplet and the doublet is
    included each as well as the ZFS of the triplet and the interactions of the
    two spins. The Hamiltonian is calculated for all orientations and magnetic
    field points in the product basis and returned as a
    B_angle_matrix attribute from the class Multioperator. Secular
    approximation is presumed.

    Parameters
    ----------
    sys : object
        Contains spinsystem parameters. This function uses the attributes
        sys.g_tri and sys.J_ex.
    exp : object
        Contains experimental parameters. This function uses the attribute
        exp.B_z (magnetic field array).
    opt : object
        Contains simulation options. This function uses opt.grid_points.
    cal : object
        Container for results of calculations during the simulation. This
        function uses cal.g_tensor and cal.g_tri_tensor, cal.D_tri_tensor and
        cal.D_tensor (e.g. from set_up_tensors). Further cal.s (spinoperator
        for a triplet-doublet-system is needed)

    Returns
    -------
    ham_tdp : np.ndarray
        Triplet-doublet-pair hamiltonian in the shape of a B_angle_matrix from
        the class Multioperator. The hamiltonian is given in the product basis
        of doublet and triplet:
        \|a, +1\>, \|a, 0\>, \|a, -1\>, \|b, +1\>, \|b, 0\>, \|b, -1\>.

    """
    s = cal.s

    cal.s = mt.Spinoperator(1)
    ham_tri = set_up_triplet_hamiltonian(exp, opt, cal)

    cal.s = mt.Spinoperator(1/2)
    ham_doub = set_up_doublet_hamiltonian(exp, opt, cal)

    cal.s = s
    ham_tdp = np.zeros((len(exp.B_z), opt.grid_points, 6, 6),
                       dtype=COMPLEX_TYPE)
    ham_tdp = np.kron(ham_doub, np.eye(3, dtype=FLOAT_TYPE)) \
        + np.kron(np.eye(2, dtype=FLOAT_TYPE), ham_tri)

    cal.D_tensor.multirot /= 2
    wt = 1/np.sqrt(2)
    ham_tdp[:, :, 0, 0] += cal.D_tensor.multirot[:, 2, 2] + 1/2*sys.J_ex
    ham_tdp[:, :, 1, 3] += -wt*cal.D_tensor.multirot[:, 2, 2] + wt*sys.J_ex
    ham_tdp[:, :, 2, 2] += -cal.D_tensor.multirot[:, 2, 2] - 1/2*sys.J_ex
    ham_tdp[:, :, 2, 4] += -wt*cal.D_tensor.multirot[:, 2, 2] + wt*sys.J_ex
    ham_tdp[:, :, 3, 1] += -wt*cal.D_tensor.multirot[:, 2, 2] + wt*sys.J_ex
    ham_tdp[:, :, 3, 3] += -cal.D_tensor.multirot[:, 2, 2] - 1/2*sys.J_ex
    ham_tdp[:, :, 4, 2] += -wt*cal.D_tensor.multirot[:, 2, 2] + wt*sys.J_ex
    ham_tdp[:, :, 5, 5] += cal.D_tensor.multirot[:, 2, 2] + 1/2*sys.J_ex

    return ham_tdp


def set_up_tdp_full_high_field_hamiltonian(sys: object, exp: object,
                                           opt: object, cal: object
                                           ) -> 'np.ndarray':
    r"""
    Calculate a Hamiltonian matrix operator for a coupled triplet doublet pair.
    Coupling with the static magnetic field of the triplet and the doublet is
    included each as well as the ZFS of the triplet and the interactions of the
    two spins. The Hamiltonian is calculated for all orientations and magnetic
    field points in the xyz-Basis (basis in which the ZFS-Hamiltonian of the
    triplet is diagonal) and returned as a B_angle_matrix
    attribute from the class Multioperator. The interaction matrices are
    calculated by direct prducts of spin matrices and interaction tensors and
    no secular approximation is applied.

    Parameters
    ----------
    sys : object
        Contains spinsystem parameters. This function uses the attributes
        sys.J_ex.
    exp : object
        Contains experimental parameters. This function uses the attribute
        exp.B_z (magnetic field array).
    opt : object
        Contains simulation options. This function uses opt.grid_points.
    cal : object
        Container for results of calculations during the simulation. This
        function uses cal.g_tensor and cal.g_tri_tensor, cal.D_tri_tensor and
        cal.D_tensor (e.g. from set_up_tensors).

    Returns
    -------
    ham_hf : np.ndarray
        Full triplet-doublet-pair high-field hamiltonian in the shape of a
        B_angle_matrix from the class Multioperator. The hamiltonian is given
        in the xyz-Basis of doublet and triplet:
        \|a, x\>, \|b, x\>, \|a, y\>, \|b, y\>, \|a, z\>, \|b, z\>.

    """
    setup_s = mt.Spinoperator(0.5, 1)
    S_doub = mt.Spinoperator(0.5, 1)
    S_trip = mt.Spinoperator(0.5, 1)

    S_doub.matrix = setup_s.matrix
    S_trip.matrix = setup_s.matrix_coupling_spins[0]

    # ZFS
    h_zfs = mut.Multioperator(S_trip, opt.grid_points, exp.B_z*MU_B)
    h_zfs.create_bilinear_operator(cal.D_tri_tensor, S_trip)
    h_zfs.angle_matrix_changed()

    # zeeman interactions
    h_zeeman_doub = mut.Multioperator(
        S_doub, opt.grid_points, exp.B_z*MU_B)
    h_zeeman_doub.zeeman_coupling(cal.g_tensor)

    h_zeeman_trip = mut.Multioperator(
        S_trip, opt.grid_points, exp.B_z*MU_B)
    h_zeeman_trip.zeeman_coupling(cal.g_tri_tensor)

    # dipolar interaction
    h_dip = mut.Multioperator(S_doub, opt.grid_points, exp.B_z*MU_B)
    h_dip.create_bilinear_operator(cal.D_tensor, S_trip)
    h_dip.angle_matrix_changed()

    # exchange interaction
    h_ex = mut.Multioperator(S_doub, opt.grid_points, exp.B_z*MU_B)
    h_ex.exchange_coupling(-1/2*sys.J_ex, S_trip)

    # full high field hamiltonian
    ham_hf = h_zeeman_doub.B_angle_matrix + h_zeeman_trip.B_angle_matrix +\
        h_zfs.B_angle_matrix + h_dip.B_angle_matrix + h_ex.B_angle_matrix


    # the eigenbasis of ham_zfs is xx yy zz (= xyz-basis)
    eig_zfs, vec_zfs = np.linalg.eigh(h_zfs.B_angle_matrix)

    # transform high field hamiltonian to xyz-basis
    ham_hf = np.conj(np.transpose(
        vec_zfs, (0, 1, 3, 2))) @ ham_hf @ vec_zfs


    ham_hf = ham_hf.astype(COMPLEX_TYPE)

    return ham_hf


def set_up_commutator_superoperator(sys: object, opt: object, cal: object
                                    ) -> None:
    """
    Build a commutator superoperator from a hilbert space hamiltonian for
    further calculations in liouville space.

    Parameters
    ----------
    sys : object
        Contains information on the spin system. This function uses information
        on the relaxation times, i.e. sys.dynamics or sys.T_relax_1 and
        sys.T_relax_2.
    opt : object
        Contains simulation option parameters. This function uses the
        attribute opt.space. If it is set to liouville the commutator
        superoperator will be calculated. Else the function will do nothing.
    cal : object
        Container for results of calculations during the simulation. This
        function uses the attribute cal.ham, a hamiltonian in hilbert space
        (B_angle_matrix is required, can be calculated e.g. by the function
        set_up_rp_hamiltonian). The hamiltonian is used to set up the
        commutator superoperator.

    Attributes
    ----------
    cal.ham_superop : np.ndarray
        The attribute will be built, if opt.space is set to liouville. This is
        the commutator superoperator given as a B_angle_matrix attribute from
        the class Multioperator.

    Returns
    -------
    None.

    """
    if opt.space == 'hilbert':
        pass
    elif opt.space == 'liouville':
        ham_adj = np.transpose(np.conjugate(cal.ham), [0, 1, 3, 2])
        ham_superop = np.kron(np.eye(
            cal.ham.shape[-1], dtype=FLOAT_TYPE), cal.ham[:, :] ) - \
            np.kron(ham_adj[:, :], np.eye(cal.ham.shape[-1], dtype=FLOAT_TYPE))

        """ (how to build the commutator superoperator)
        https://physics.stackexchange.com/questions/163546/finding-the-matrix-representation-of-a-superoperator
        https://arxiv.org/pdf/1510.08634
        """
        relax = rlx.create_relaxation_superoperator(sys, cal)

        ham_superop = ham_superop + 1j * relax
        cal.ham_superop = ham_superop
    else:
        print('opt.space has to be either hilbert or liouville')

    return None
