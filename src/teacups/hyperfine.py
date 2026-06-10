import numpy as np
import teacups.creators as cr
import teacups.multioperator_tools as mut
import teacups.matrix_tools as mt
import teacups.signals_and_processing as sap
import teacups.hamiltonians as ham

import scipy.constants as const

MU_B = const.physical_constants["Bohr magneton in Hz/T"][0]


def set_up_hyperfine_tensors(sys: object, cal: object) -> None:
    """
    Set up a set of hyperfine tensors, optionally rotate them to an initial
    frame and rotate them to a set of angles theta and phi. The function
    cr.create_tensor is used.

    Parameters
    ----------
    sys : object
        Contains parameters concerning the spin system. This function takes
        use of the attribute sys.A which contains a list of lists with the
        three diagonal elements af a hyperfine-tensor each. If sys.A_frame
        is given (list of same dimension as sys.A) the tensors are rotated
        to the given initial frame.
    cal : object
        Contains calculated results during the simulation. Cal.phi and cal.
        theta are needed, which contain all angle points.

    Attributes
    ----------
    cal.A_tensor : list
        List of tensor objects one for each hyperfine tensor given in sys.

    Returns
    -------
    None.

    """
    cal.A_tensor = []
    if hasattr(sys, "A_frame"):
        for spin_count, spin in enumerate(sys.A):
            A_s = []
            for nuc_count, nuc in enumerate(spin):
                A_tensor = cr.create_tensor(
                    sys.A[spin_count][nuc_count],
                    cal.phi,
                    cal.theta,
                    sys.A_frame[spin_count][nuc_count],
                )
                A_s.append(A_tensor)
            cal.A_tensor.append(A_s)
    else:
        for spin_count, spin in enumerate(sys.A):
            A_s = []
            for nuc_count, nuc in enumerate(spin):
                A_tensor = cr.create_tensor(
                    sys.A[spin_count][nuc_count], cal.phi, cal.theta
                )
                A_s.append(A_tensor)
            cal.A_tensor.append(A_s)

    return None


def create_hf_hamiltonian(spins: list, coupling_nucs: list, hf_tensors: list) -> list:
    """
    Calculate the diagonal elements of a hyperfine interaction hamiltonian
    of any number of spins with any number of nuclei. Input parameters are
    given as lists, their dimensions are described below. The interaction
    hamiltonian is calculated for all grid_points and for all combinations
    of spin and nuceus. The hamiltonian is set up for each spin with the
    given coupling nuclei and returned in a list with the length of the number
    of spins. The spin-hamiltonians lengths are given as the product of the
    multiplicities of the spin and the coupling nuclei.

    Parameters
    ----------
    spins : list
        List containing the spin quantum numbers of the spins in the system as
        floats.
    coupling_nucs : list
        List containing lists of the spin quantum numbers of the coupling
        nuclei. In the coupling_nucs list has to be a list of floats for
        each spin. The quantum numbers of the nuclei are placed into the
        lists, which are refered to their spin. E.g. if a nucleus couples
        to the second spin in the spins-list its quantum number is set into
        the second list in coupling_nucs-list.
    hf_tensors : list
        List containing lists of the hyperfine tensors. The list is set up
        analogously to the coupling_nucs list. For each nucleus a tensor is
        placed at the same position of the hf_tensors-list. The tensors in
        the list are objects of class Tensor and have the multirot attribute
        initialized.

    Returns
    -------
    list
        For each spin an array is returned in this list. The array contains
        the diagonal elements of the hyperfine interaction in the product
        basis. The shape of the arrays is
        gridpoints x (multiplicity_spin*product of
        multiplicity_of_all_coupling_nucs).

    Examples
    --------

    >>> A = mt.Tensor([1, 1, 1])
    >>> A.multirotation(1)
    >>> create_hf_hamiltonian([1/2], [[1/2]], [[A]])
    [array([[ 0.25+0.j, -0.25+0.j, -0.25+0.j,  0.25+0.j]])]

    >>> create_hf_hamiltonian([1/2, 1/2], [[1/2],[1]], [[A],[A]])
    [array([[ 0.25+0.j, -0.25+0.j, -0.25+0.j,  0.25+0.j]]),
     array([[ 0.5+0.j,  0. +0.j, -0.5+0.j, -0.5+0.j,  0. +0.j,  0.5+0.j]])]

    >>> h = create_hf_hamiltonian([1/2, 1], [[1/2, 1/2],[]], [[A, A],[]])
    >>> h
    [array([[ 0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j, -0.5+0.j,  0. +0.j,
              0. +0.j,  0.5+0.j]]),
     array([[0.+0.j, 0.+0.j, 0.+0.j]])]
    >>> h[0].shape
    (1, 8)

    """
    ham_hf = []
    for n, s in enumerate(spins):
        spin_matrices = mt.Spinoperator(s, coupling_nucs[n])
        ham_hf_for_each_spin = np.zeros(
            (
                list(np.concatenate(hf_tensors).flat)[0].multirot.shape[0],
                spin_matrices.dimension,
                spin_matrices.dimension,
            ),
            dtype=np.complex128,
        )

        for nuc in range(spin_matrices.matrix_coupling_spins.shape[0]):
            ham_hf_tmp = mut.Multioperator(
                spin_matrices, hf_tensors[n][nuc].multirot.shape[0], [0]
            )
            try:
                s2 = mt.Spinoperator(s, coupling_nucs[n])
                s2.matrix = spin_matrices.matrix_coupling_spins[nuc]
                ham_hf_tmp.create_bilinear_operator(hf_tensors[n][nuc], s2)
            except IndexError:
                pass
            ham_hf_for_each_spin += ham_hf_tmp.angle_matrix

        ham_hf_for_each_spin = ham_hf_for_each_spin[
            :, range(spin_matrices.dimension), range(spin_matrices.dimension)
        ]
        ham_hf.append(ham_hf_for_each_spin)

    return ham_hf


def make_signal_with_hyperfine(
    sys: object, exp: object, opt: object, cal: object
) -> None:
    """
    Calculate the timeresolved signal of a trEPR-experiment a system consisting
    of one or two spins including hyperfine interactions. The Hamiltonian
    is set up for all combinations of magnetic quantum numbers of the spins
    and the coupling nucs. The propagaion and the signal are calculated for
    each orientation. Therefore, all attributes needed in sap.propagation and
    sap.make_signal are needed in this function. Further attributes are listed
    below.

    Parameters
    ----------
    sys : object
        Object of class spinsystem. This function uses sys.s (list of spin
        quantum numbers of the spins) and sys.spin_system (string with the
        name of the spin system).
    exp : object
        Experimental parameters object.
    opt : object
        Simulation options object.
    cal : object
        Object of class Calculations. Results during the simulation are saved
        here. This function uses from previous functions the following
        attributes: cal.ham (hamiltonian of the system excluding the
        hyperfines) cal.ham_hf (hyperfine hamiltonian, e.g. built by
        create_hf_hamiltonian)

    Attributes
    ----------
    cal.spec_sim : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z. So the shape is len(t)xlen(B).

    Returns
    -------
    None

    """
    dims = (np.array(sys.s)) * 2 + 1

    ham_total = cal.ham
    dim_total = ham_total.shape[-1]

    # system consisting of 2 spins
    if len(sys.s) == 2:
        dimensions = [int(dim_total / dims[1]), int(dim_total / dims[0])]
        combinations = [
            int(cal.ham_hf[0].shape[-1] / dimensions[0]),
            int(cal.ham_hf[1].shape[-1] / dimensions[1]),
        ]

        ham_hf = hyperfine_of_coupled_system(
            cal.ham_hf, combinations, dimensions, dim_total
        )

        for m_a in range(combinations[0]):
            for m_b in range(combinations[1]):
                cal.ham = ham_total.copy()
                ham_hf_tmp = ham_hf[0][m_a] + ham_hf[1][m_b]

                if sys.spin_system == "rp":
                    ham_hf_tmp *= 2 * np.pi
                    cal.ham[:, :, 0, 0] -= ham_hf_tmp[:, 0]
                    cal.ham[:, :, 1, 2] -= ham_hf_tmp[:, 1]
                    cal.ham[:, :, 2, 1] -= ham_hf_tmp[:, 1]
                    cal.ham[:, :, 3, 3] -= -ham_hf_tmp[:, 0]
                else:
                    cal.ham[:, :, range(dim_total), range(dim_total)] += ham_hf_tmp

                ham.set_up_commutator_superoperator(sys, opt, cal)

                print(
                    "start propagation for combination "
                    + str(m_a + 1)
                    + "/"
                    + str(combinations[0])
                    + " "
                    + str(m_b + 1)
                    + "/"
                    + str(combinations[1])
                )
                sap.propagation(sys, opt, cal)

                print("start making the signal...")
                sap.make_signal(exp, opt, cal)

                sap.powder_average(exp, opt, cal)

    # system consisting of 1 spin
    elif len(sys.s) == 1:
        dim_hf = cal.ham_hf[0].shape[-1]
        combinations = int(dim_hf / dim_total)

        for m_i in range(combinations):
            cal.ham = ham_total.copy()
            cal.ham[:, :, range(dim_total), range(dim_total)] += cal.ham_hf[0][
                :, m_i::combinations
            ]

            ham.set_up_commutator_superoperator(sys, opt, cal)
            print(
                "start propagation for combination "
                + str(m_i + 1)
                + "/"
                + str(combinations)
            )
            sap.propagation(sys, opt, cal)

            print("start making the signal...")
            sap.make_signal(exp, opt, cal)

            sap.powder_average(opt, cal)

    return None


def hyperfine_of_coupled_system(
    hams_hf: list, combinations: list, dimensions: list, dim_total: float
) -> list:
    """
    Calculate the hyperfine hamiltonian of a spin system of two coupled spins.
    For each spin M_i hamiltonians are calculated, where M_i is the the total
    number of magnetic quantum numbers. E.g. if one nucleus with spin 1/2 is
    coupled M_i is 2, if one nucleus with spin 1/2 and one nucleus with spin 1
    are coupled to the spin M_i is 2*3 = 6. A list with the diagonal elements
    of the two hamiltonians is returned. The shape of each element of the list
    is M_i x grid_points x total_dimension_of_spinsystem.

    Parameters
    ----------
    hams_hf : list
        List with two hyperfine hamiltonians (one for each spin). The
        hamiltonians are B_angle_matrices as returned by the function
        create_hf_hamiltonian.
    combinations : list
        List of two integers, one for each spin. This is the number of
        combinations of the magnetic quantum numbers of spin and coupling
        nuclei. The number can be calculated by:
        hf_hamiltonian_shape/multiplicity_of_spin
    dimensions : list
        List of integers, one for each spin. They give the multiplicity of
        each spin. E.g. if S = 1, dimension = 2*1+2 = 3.
    dim_total : float
        Total dimension of the spinsystem. E.g if s_1 = 1/2, s_2 = 1 the
        total dimension is 2*3=6.

    Returns
    -------
    list
        List with two hyperfine interaction hamiltonians. Their shape is
        M_i x grid_points x total_dimension_of_spinsystem each.

    Examples
    --------

    >>> A = mt.Tensor([1, 1, 1])
    >>> A.multirotation(1)
    >>> h = create_hf_hamiltonian([1/2, 1], [[1/2, 1/2],[]], [[A, A],[]])
    >>> h
    [array([[ 0.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j, -0.5+0.j,  0. +0.j,
              0. +0.j,  0.5+0.j]]),
     array([[0.+0.j, 0.+0.j, 0.+0.j]])]
    >>>  hyperfine_of_coupled_system(h, [4, 1], [2, 3], 6)
    [array([[[ 0.5+0.j,  0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j]],
            [[ 0. +0.j,  0. +0.j,  0. +0.j,  0. +0.j,  0. +0.j,  0. +0.j]],
            [[ 0. +0.j,  0. +0.j,  0. +0.j,  0. +0.j,  0. +0.j,  0. +0.j]],
            [[-0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j]]]),
     array([[[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]])]

    """
    grid_points = hams_hf[0].shape[0]
    ham_a = np.zeros((combinations[0], grid_points, dim_total), dtype=complex)
    ham_b = np.zeros((combinations[1], grid_points, dim_total), dtype=complex)

    for a in range(combinations[0]):
        ham_tmp = np.zeros((grid_points, dimensions[0], dimensions[0]), dtype=complex)
        ham_tmp[:, range(dimensions[0]), range(dimensions[0])] = hams_hf[0][
            :, a :: combinations[0]
        ]
        ham_tmp = np.kron(ham_tmp, np.eye(dimensions[1]))
        ham_a[a] = ham_tmp[:, range(dim_total), range(dim_total)]

    for b in range(combinations[1]):
        ham_tmp = np.zeros((grid_points, dimensions[1], dimensions[1]), dtype=complex)
        ham_tmp[:, range(dimensions[1]), range(dimensions[1])] = hams_hf[1][
            :, b :: combinations[1]
        ]
        ham_tmp = np.kron(np.eye(dimensions[0]), ham_tmp)
        ham_b[b] = ham_tmp[:, range(dim_total), range(dim_total)]

    return [ham_a, ham_b]
