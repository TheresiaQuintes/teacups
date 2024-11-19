import numpy as np


def tensor_rotation(tensor: 'np.ndarray', phi: float, theta: float,
                    psi=0.0) -> 'np.ndarray':
    """
    Euler transformation of a given tensor using y-convention.

    Euler matrix is set up with the given angles in the multiplicated form.
    Unitytransformation is carried out.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor in main axes system which will be rotated by euler
        transformation. The shape has to be 3x3.
    phi : float
        First euler angle given in rad.
    theta : float
        Second euler angle given in rad.
    psi : float, optional
        Third euler angle given in rad. The default is 0.

    Returns
    -------
    rotated_tensor : np.ndarray
        Tensor after euler transformation.

    Examples
    --------
    >>> tensor_rotation(np.array(([1, 2, 3], [0, 0, 0], [0, 0, 0])), 1, 2)
    array([[ 0.821379  , -0.05376802, -0.17383894],
           [ 3.07396785, -0.20122401, -0.65058311],
           [-1.79474586,  0.11748527,  0.37984501]])
    """

    # Allocations
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    costhet = np.cos(theta)
    sinthet = np.sin(theta)
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    eulermatrix = np.zeros((3, 3))

    # Set up the full 3-dimensional Euler matrix
    eulermatrix[0][0] = cosphi*costhet*cospsi - sinphi*sinpsi
    eulermatrix[0][1] = -cosphi*costhet*sinpsi - sinphi*cospsi
    eulermatrix[0][2] = cosphi*sinthet
    eulermatrix[1][0] = sinphi*costhet*cospsi + cosphi*sinpsi
    eulermatrix[1][1] = - sinphi*costhet*sinpsi + cosphi*cospsi
    eulermatrix[1][2] = sinphi*sinthet
    eulermatrix[2][0] = -sinthet*cospsi
    eulermatrix[2][1] = sinthet*sinpsi
    eulermatrix[2][2] = costhet

    # Final two sided matrix multiplication (similarity transformation)
    eulermatrix_transpose = eulermatrix.T
    rotated_tensor = eulermatrix_transpose@(tensor@eulermatrix)
    rotated_tensor = rotated_tensor.astype(tensor.dtype)
    return rotated_tensor


def create_linear_hamiltonian(tensor: 'np.ndarray', spinop: 'np.ndarray',
                              z=True) -> 'np.ndarray':
    """
    Create a linear interaction Hamiltonian (e.g. Zeeman interaction) between
    a spin vector S and an interaction matrix tensor.

    Choose between either the x- and y- components of a third vector beeing
    multiplied (e.g. static magnetic field vector) set to zero (perpendicular
    magnetic field) or the y- and z- components set to zero (parallel
    magnetic field).

    Parameters
    ----------
    tensor : np.ndarray
        Interaction matrix between spin vector and magnetic field.
        The shape is 3x3.
    spinop : np.ndarray
        Spin vector operator. Contains nxn spin-operator-matrices:
        [S_x, S_y, S_z].
    z : bool, optional
        Gives choice which component of the third vector is not set to zero.
        The default is True.

    Returns
    -------
    ham : np.ndarray
        Linear interaction Hamiltonian. The shape is nxn
        H = S*tensor*magnetic-field-vector.

    Examples
    --------
    >>> sigma_x = np.array([[0, 0.5], [0.5, 0]])
    >>> sigma_y = np.array([[0, -1j/2], [1j/2, 0]])
    >>> sigma_z = np.array([[0.5, 0], [0, -0.5]])
    >>> unit = np.eye(2)
    >>> S_x = np.kron(sigma_x, unit)
    >>> S_y = np.kron(sigma_y, unit)
    >>> S_z = np.kron(sigma_z, unit)
    >>> S = np.array([S_x, S_y, S_z])
    >>> g = np.array([[2.0, 0, 0], [0, 2.01, 0], [0, 0, 2.02]])
    >>> create_linear_hamiltonian(g, S)
    array([[ 1.01+0.j,  0.  +0.j,  0.  +0.j,  0.  +0.j],
           [ 0.  +0.j,  1.01+0.j,  0.  +0.j,  0.  +0.j],
           [ 0.  +0.j,  0.  +0.j, -1.01+0.j,  0.  +0.j],
           [ 0.  +0.j,  0.  +0.j,  0.  +0.j, -1.01+0.j]])
    >>> create_linear_hamiltonian(g, S, z=False)
    array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
    """

    if z is True:
        # Set up the Hamiltonian for perpendicular magnetic field
        ham = spinop[0]*tensor[0, 2] + spinop[1]*tensor[1, 2]\
            + spinop[2]*tensor[2, 2]
    else:
        # Set up the Hamiltonian for parallel magnetic field
        ham = spinop[0]*tensor[0, 0] + spinop[1]*tensor[1, 0]\
            + spinop[2]*tensor[2, 0]
    return ham


def create_bilinear_hamiltonian(spinop1: 'np.ndarray', tensor: 'np.ndarray',
                                spinop2: 'np.ndarray') -> 'np.ndarray':
    """
    Create a bilinear interaction Hamiltonian between two spin vectors,
    spinop1 and spinop2, and an interaction matrix tensor.

    Parameters
    ----------
    spinop1 : np.ndarray
        Spin vector operator of the first interacting Spin.
        Contains nxn spin-operator-matrices: [S_x, S_y, S_z].
    tensor : np.ndarray
        3x3 - Interaction matrix between the two spin vectors.
    spinop2 : np.ndarray
        Spin vector operator of the second interacting spin.
        Contains nxn spin-operator-matrices: [S_x, S_y, S_z].

    Returns
    -------
    ham : np.ndarray
        Bilinear interaction Hamiltonian. H = spinop1*tensor*spinop2.
        The shape will be nxn.

    Examples
    --------
    >>> sigma_x = np.array([[0, 0.5], [0.5, 0]])
    >>> sigma_y = np.array([[0, -1j/2], [1j/2, 0]])
    >>> sigma_z = np.array([[0.5, 0], [0, -0.5]])
    >>> unit = np.eye(2)
    >>> S_x = np.kron(sigma_x, unit)
    >>> S_y = np.kron(sigma_y, unit)
    >>> S_z = np.kron(sigma_z, unit)
    >>> S = np.array([S_x, S_y, S_z])
    >>> D = np.array([[0.5, 0, 0], [0, 1.5, 0], [0, 0, -2]])
    >>> create_bilinear_hamiltonian(S, D, S)
    array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """

    ham = 0

    for i in range(0, 3):
        for j in range(0, 3):
            ham = np.add(ham, spinop1[i] @ (tensor[i, j] * spinop2[j]))
    return ham
