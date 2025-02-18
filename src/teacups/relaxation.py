import numpy as np

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


def superoperator_population_relaxation(k_matrix):
    """
    Calculate a relaxation superoperator for transitions between the states.
    A matrix k_matrix can be given containing the relaxation rates for
    the transitions between the populations. The decay of the populations
    is calculated as a sum of the transitions to other states. Diagonal
    elements of the k_matrix give decay rates of populations to other states
    (which are not described by the matrix, e.g. a ground state). Attention:
    If diagonal elements are defined the superoperator changes the trace of
    the density matrix and it does not stay 1.

    Parameters
    ----------
    k_matrix : np.ndarray
        Matrix containing the transition rates between the states. E.g.:

        ::

                         a b c
                       a 4 1 2
            k_matrix = b 1   3
                       c 2 3

        for the transition sceme:

        ::

            a <-> b: 1
            a <-> c: 2
            b <-> c: 3

    Returns
    -------
    R : np.ndarray
        Relaxation superoperator. Its dimension is the dimension of T1_matrix
        squared. E.g.:

        ::

              aa  ab  ac  ba  bb  bc  ca  cb  cc
           aa 1                1               2
           ab
           ac
           ba
           bb  1              -4               3
           bc
           ca
           cb
           cc  2               3              -5

    """
    dimension = k_matrix.shape[0]
    dimension_sqr = dimension**2

    R = np.zeros((dimension_sqr, dimension_sqr), dtype=FLOAT_TYPE)
    R_diag = np.zeros((dimension_sqr, dimension_sqr), dtype=FLOAT_TYPE)

    for n in range(dimension):
        for m in range(dimension):
            R[n+n*dimension, m+m*dimension] = k_matrix[n, m]

        for n, el in enumerate(R):
            R_diag[n, n] = -np.sum(el)

        R += R_diag

    R[range(0, dimension_sqr, dimension+1),
      range(0, dimension_sqr, dimension+1)] += np.diag(k_matrix)

    return R


def superoperator_coherence_relaxation(k, dimension):
    """
    Calculate a superoperator for the decay of the coherences of a matrix.
    The coherences will decay with the rate constant T2

    Parameters
    ----------
    k : float
        Rate constant of the decay of the coherences.
    dimension : int
        Hilbert space dimension of the matrices. E.g. for a triplet
        dimension = 3.

    Returns
    -------
    R : np.ndarray
        Relaxationsuperoperator for the coherences. All diagonal elements that
        are assignable to the coherences are set to the negative value of k.
        E.g.:

        ::

              aa ab ba bb
           aa
           ab    -k
           ba      -k
           bb

    """
    R = np.eye(dimension**2, dtype=FLOAT_TYPE)
    R *= -k

    R[range(0, dimension**2, dimension+1),
      range(0, dimension**2, dimension+1)] = 0

    return R


def phenomenological_relaxation_superoperator(T_relax_1: float, T_relax_2:
                                              float, dimension: float
                                              ) -> 'np.ndarray':
    """
    Create the phenomenological relaxation superoperator for a spin system
    using the relaxation times T1 (longitudinal) and T2 (transversal).
    The superoperator matrix is set up in the eigenbasis of the system with
    ascending order of the eigenvalues.

    Parameters
    ----------
    T_relax_1 : float
        Longitudinal relaxation time describing spin-lattice relaxation.
    T_relax_2 : float
        Transversal relaxation time describing FID.
    dimension : float
        Hilbert space dimension of the matrices. E.g. for a triplet
        dimension = 3.

    Returns
    -------
    relaxation_superop : numpy.array
        Relaxation superoperator matrix in the systems eigenbasis.
        Its shape is dimension**2 x dimension**2.

    """
    T_relax_1 = T_relax_1**(-1)
    T_relax_2 = T_relax_2**(-1)

    T1_matrix = T_relax_1 * (np.ones((dimension, dimension))-np.eye(dimension))

    R_pop = superoperator_population_relaxation(T1_matrix)
    R_coh = superoperator_coherence_relaxation(T_relax_2, dimension)

    R = R_pop + R_coh
    return R


def relaxation_operator_to_hamiltonian_basis(relaxation_superoperator:
                                             'np.ndarray', eigvec: 'np.ndarray'
                                             ) -> 'np.ndarray':
    """
    Transform the relaxation superoperator from the eigenbasis of the
    Hamiltonian to the basis of the Hamiltonian that is used during all
    calculations. The Hamiltonian is B-angle-matrix-style. After transformation
    the relaxaton superoperator has this shape too, as basistransformation
    is done for every B- and angle-point.

    The matrix containing the eigenvectors of the system is defined by:

    .. math::
        H_\mathrm{diag} = v^{-1} H v

    In superoperator space the transformationmatrix for the identical
    basistransformation can be set up as

    .. math::
        v_\mathrm{super} = v \otimes v

    so that a basistransformation of a superoperator R can be done by:

    .. math::
        R_\mathrm{H basis} = v_\mathrm{super} \cdot R_\mathrm{diag}\
            \cdot v_\mathrm{super}^{-1}.

    Parameters
    ----------
    relaxation_superoperator : np.ndarray
        Quadratic superoperator containing the relaxation rates between the
        eigenstates of the spinsystem in ascending order from left to right.
    eigvec : np.ndarray
        Eigenvectors of the spin systems Hamiltonian that can be used for
        basistransformation between the eigenstates and the main basis for
        further calculations. The shape of this array is
        B_points x angle_points x dim x dim.

    Returns
    -------
    relaxation : np.ndarray
        Relaxationsuperoperator in the same basis as the systems Hamiltonian.
        It has the shape of a B-angle-matrix now.

    """
    dim = (eigvec.shape[0], eigvec.shape[1], eigvec.shape[2]**2,
           eigvec.shape[3]**2)
    R = np.zeros(dim, dtype=COMPLEX_TYPE)
    for b in range(dim[0]):
        for a in range(dim[1]):
            R[b, a] = np.kron(eigvec[b, a], eigvec[b, a])

    relaxation = R\
        @ relaxation_superoperator @ np.conj(np.transpose(R, (0, 1, 3, 2)))

    return relaxation


def create_relaxation_superoperator(sys: object, cal: object) -> 'np.ndarray':
    """
    Calculate a relaxation superoperator matrix dependend on the user input.
    If a relaxation time is given the phenomenological relaxation superoperator
    is returned. If a matrix is given in the attribute sys.dynamics this matrix
    is taken to set up a superoperator describing any dynamic system.

    Parameters
    ----------
    sys : Spin system object.
        Contains information about the relaxation process. If it has an
        attribute sys.T_relax_1 (and sys.T_relax_2) these are two floats that
        are the longitudinal (and transversal) relaxation time in seconds.
        If sys has an attribute called sys.dynamics relaxation times are
        ignored and the matrix provided in sys.dynamics is used to set up an
        relaxation superoperator. The matrix has to have the shape nxn where
        n is the dimension of the systems spinoperator. The basis are the
        eigenfunctions in ascending order. Elements of the matrix are rate
        constants (1/s) of the transtions between the eigenstates.
    cal : Object containing results during calculations.
        This function uses the spin attribute cal.s and the systems
        eigenvectors cal.eigvec.

    Raises
    ------
    AttributeError
        An attribute error is raised if neither sys.T_relax_1 nor sys.dynamics
        is given.

    Returns
    -------
    relax : np.ndarray
        Relaxation superoperator, dimension n**2 x n**2. The relaxation
        superoperator is given back in the basis of the systems Hamiltonian.

    """
    if sys.dynamics is not None:
        relax = superoperator_population_relaxation(sys.dynamics)
    elif hasattr(sys, 'T_relax_1'):
        relax = phenomenological_relaxation_superoperator(sys.T_relax_1,
                                                          sys.T_relax_2,
                                                          cal.s.dimension)
    else:
        raise AttributeError('The dynamics of the spin system have do be \
                             either defined by the matrix sys.dynamics or \
                                 by the phenomenological relaxation times \
                                     sys.T_relax_1 and sys.T_relax_2.')

    relax = relaxation_operator_to_hamiltonian_basis(relax, cal.eigvec)

    return relax
