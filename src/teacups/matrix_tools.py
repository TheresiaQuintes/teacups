import numpy as np
import teacups.orientation_dependent_ham as odh

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


class Matrix:
    """
    A Matrix-object contains the dimension and the matrix of a square matrix.

    Parameters
    ----------
    dimension : int
        Dimension of matrix which is generated.

    Attributes
    ----------
    matrix : np.ndarray
        Array containing only zeros. The shape of the array is
        dimension x dimension.

    dimension: int
        Dimension of matrix.

    Raises
    ------
    ValueError
        ValueError is raised, if the dimension is no integer.

    Examples
    --------
    >>> m = Matrix(2)
    >>> m.matrix
    array([[0., 0.],
           [0., 0.]])
    >>> m.dimension
    2

    """

    def __init__(self, dimension: int):
        if type(dimension) is not int:
            raise ValueError("Only use integer values for dimension!")
        else:
            self.matrix = np.zeros((dimension, dimension), dtype=COMPLEX_TYPE)
            self.dimension = dimension

    def scalar(self, multipliers: "np.ndarray") -> None:
        """
        Build scalar product of the matrix attribute with the product of all
        multipliers.

        Parameters
        ----------
        multipliers : np.ndarray
            Array containing all scalars which shall be multiplied with each
            other and with the matrix. The shape of the array is n x 1.

        Attributes
        ----------
        matrix : np.ndarray
            Matrix attribute is changed by being multiplied with the scalars.
            The shape of the matrix is still dimension x dimension.

        Returns
        -------
        None.

        Examples
        --------
        >>> m = Matrix(2)
        >>> m.matrix[0, 0] = 2
        >>> m.matrix[1, 1] = 3
        >>> multipliers = np.array([2, 3, 4])
        >>> m.matrix
        array([[2., 0.],
               [0., 3.]])
        >>> m.scalar(multipliers)
        >>> m.matrix
        array([[48.,  0.],
               [ 0., 72.]])

        """
        self.matrix *= np.prod(multipliers)

        return None

    def product(self, scd_matrix: "np.ndarray", left=False) -> None:
        """
        Build matrix product of the matrix attribute and a second matrix
        being multiplied either from right (default) or from left side.

        Parameters
        ----------
        scd_matrix : np.ndarray
            Matrix of same dimension as matrix attribute, which shall be
            multiplied with the matrix attribute.
        left : boolean, optional
            If left is set to True the matrix attribute is multiplied from
            the left side. If left is set to False it is multiplied from
            the right. The default is False.

        Raises
        ------
        IndexError
            IndexError is raised if the dimension of the second matrix is not
            the same as the dimension of matrix attribute.

        Attributes
        ----------
        matrix : np.ndarray
            Matrix attribute is changed to the multiplication product. The
            dimension is not changed.

        Returns
        -------
        None.

        Examples
        --------
        >>> a = np.arange(1, 5).reshape((2, 2))
        >>> m = Matrix(2)
        >>> m.matrix = a
        >>> m.matrix
        array([[1, 2],
               [3, 4]])
        >>> m.product(m.matrix)
        >>> m.matrix
        array([[ 7, 10],
               [15, 22]])

        """
        if scd_matrix.shape != self.matrix.shape:
            raise IndexError(
                "Multiplied matrix must have same shape as" + "matrix attribute."
            )
        elif not left:
            self.matrix = self.matrix @ scd_matrix
        else:
            self.matrix = scd_matrix @ self.matrix

        return None

    def basis_transformation(
        self, trans: "np.ndarray", inverse_left=True, orthonormal=False
    ) -> None:
        """
        Change the basis of matrix. trans is the transformation matrix:

        .. math::
            M = T^{-1} \cdot M \cdot T.

        Parameters
        ----------
        trans : np.ndarray
            Transformation matrix containing arrays of the new basis. The
            array has to have the same shape as the matrix attribute.
        inverse_left : boolean, optional
            Defines the direction of the basis transformation. If set to
            True the inverse of the transformation matrix is multiplied
            from the left side. If set to false, the inverse of the
            transformation matrix is multiplied from the right side and the
            direction of the basistransformation is reversed.
            The default is True.
        orthonormal : boolean, optional
            If orthonormal is set to True the basis transforation is done by
            using the adjungate instead of the inverse transformation matrix
            in calculation. This is possible if old and new basis consist only
            of orthonormal vectors. The default is False.

        Attributes
        ----------
        matrix : np.ndarray
            Matrix in new basis system.

        Returns
        -------
        None.

        Examples
        --------
        >>> m = Matrix(2)
        >>> m.matrix = np.arange(1, 5).reshape((2,2))
        >>> trans = np.array([[0, 1/2], [-1/2, 0]])
        >>> m.basis_transformation(trans)
        >>> m.matrix
        array([[ 4., -3.],
               [-2.,  1.]])

        """
        if trans.shape != self.matrix.shape:
            raise IndexError(
                "Transformationmatrix must have same shape as" + "matrix attribute."
            )

        if inverse_left is True:
            if orthonormal is True:
                self.matrix = np.conj(trans.T) @ self.matrix @ trans
            else:
                self.matrix = np.linalg.inv(trans) @ self.matrix @ trans
        else:
            if orthonormal is True:
                self.matrix = trans @ self.matrix @ np.conj(trans.T)
            else:
                self.matrix = trans @ self.matrix @ np.linalg.inv(trans)

        return None


class Tensor(Matrix):
    """
    An object from class Tensor is a special Matrix object. The matrix
    attribute contains a diagonal 3x3 matrix. The diagonal elements
    are given when setting up the object. The class Tensor provides functions
    for the rotation of a tensor to other frames.

    Parameters
    ----------
    diagonal : np.ndarray
        1D-List (or array) containing exactly 3 diagonal elements of the
        tensor. All other values in the tensor matrix will be set to zero.

    Attributes
    ----------
    matrix : np.ndarray
        The matrix attribute is a diagonal 3x3-matrix which contains
        the diagonal elements given in "diagonal".
    dimension: int
        The dimension of the matrix attribute is 3.


    Examples
    --------
    >>> a = np.arange(1, 4)
    >>> t = tensor(a)
    >>> t.matrix
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    >>> t.dimension
    3

    """

    def __init__(self, diagonal: list):
        if len(diagonal) != 3:
            raise IndexError("Tensor needs exactly three diagonal elements")

        self.matrix = np.diag(np.array(diagonal, dtype=FLOAT_TYPE))
        self.dimension = 3

    def rotation(self, phi: float, theta: float, psi=0.0) -> None:
        """
        Rotate the tensor attribute by Euler rotation. The result is given to
        the attribute rot, while tensor remains unchanged.

        Parameters
        ----------
        phi : float
            First euler angle given in rad.
        theta : float
            Second euler angle given in rad.
        psi : float, optional
            Third euler angle given in rad. The default is 0.

        Raises
        ------
        ValueError
            If dimension of tensor is not 3x3 ValueError will be raised, as
            the Euler transformation is not possible.

        Returns
        -------
        None.

        Attributes
        ----------
        self.rot : np.ndarray
            Contains elements of the rotated tensor. The dimension is 3 x 3.

        Examples
        --------
        >>> a = np.arange(1, 4)
        >>> t = tensor(a)
        >>> t.rotation(1, 1)
        >>> t.rot
        array([[ 2.62285229,  0.24564775, -0.58737276],
               [ 0.24564775,  1.29192658,  0.3825737 ],
               [-0.58737276,  0.3825737 ,  2.08522113]])

        """
        self.rot = odh.tensor_rotation(self.matrix, phi, theta, psi)
        self.rot = self.rot.astype(FLOAT_TYPE)

        return None

    def multirotation(self, phi: "np.ndarray", theta: "np.ndarray") -> None:
        """
        Rotate the tensor attribute to all combinations of theta and
        phi. The  two arrays contain angles for each point that shall be
        calculated. All results are saved in a new multirot attribute.

        Parameters
        ----------
        phi : np.ndarray
            Array with the phi values for each angle point.
        theta : np.ndarray
            Array with the theta values for each angle point.

        Attributes
        ----------
        multirot : np.ndarray
            Matrix containing the rotated tensor for each angle combination.
            The matrix has the dimension nKnots x 3 x 3.

        Returns
        -------
        None.

        """
        self.multirot = np.zeros((len(phi), 3, 3), dtype=FLOAT_TYPE)

        for i in range(0, len(theta)):
            self.rotation(phi[i], theta[i])
            self.multirot[i] = self.rot

        return None


class Operator(Matrix):
    """
    An Operator object is an object of class matrix, which has two additional
    attributes: The vector attribute and the superop attribute.

    Parameters
    ----------
    dimension : int
        Dimension of matrix which is generated.

    Attributes
    ----------
    matrix : np.ndarray
        Array containing only zeros. The shape of the array is
        dimension x dimension.
    dimension : int
        Dimension of the matrix.
    vector : np.ndarray
        Flattened matrix. The new dimension is 1 x dimension^2.
    superop : None
        Attribute can be filled by the function build_superoperator.

    """

    def __init__(self, dimension: int):
        Matrix.__init__(self, dimension)
        self.vector = self.build_vector()
        self.superop = self.build_superoperator()

    def build_vector(self) -> None:
        """
        Changing the dimension of the matrix attribute to a 1d vector
        using .flatten()-function for numpy arrays. Lines are simply written
        in only one long line.

        Attributes
        ----------
        vector : np.ndarray
            Vector array containing all values of the matrix in one line.
            The new dimension is 1 x dimension^2

        Returns
        -------
        None.

        Examples
        --------
        >>> o = Operator(2)
        >>> o.matrix[1, 0] = 1
        >>> o.matrix
        array([[0., 0.],
               [1., 0.]])
        >>> o.build_vector()
        >>> o.vector
        array([0., 0., 1., 0.])

        """
        self.vector = self.matrix.flatten()

        return None

    def build_superoperator(self, swap=False) -> None:
        """
        Double the dimension of the matrix operator (in case of use in a higher
        dimensional space). Tensor product of operator with unit matrix of the
        same dimension is built using the numpy.kron function.

        Parameters
        ----------
        swap : Boolean, optional
            If swap is False the tensor product is built wiht matrix on left
            and unit matrix on right site. If swap is set to any other value
            matrix and unit matrix are swapped. The default is False.

        Attributes
        ----------
        superop :  numpy array
            Result of tensor product of matrix and unit matrix. The dimension
            is 2*dimension x 2*dimension.

        Returns
        -------
        None.

        Examples
        --------
        >>> o = Operator(2)
        >>> o.matrix[0, 0] = 1
        >>> o.build_superoperator()
        >>> o.superop
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        >>> o.build_superoperator(swap=True)
        >>> o.superop
        array([[1., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 0.]])

        """
        eye = np.eye(self.dimension, dtype=self.matrix.dtype)
        if not swap:
            self.superop = np.kron(self.matrix, eye)
        else:
            self.superop = np.kron(eye, self.matrix)

        return None


class Spinoperator(Operator):
    """
    By creating a spinoperator object the pauli matrices for a spin (with any
    spin quantum number) are calculated. Optionally the spin is coupled to any
    number of further arbitrary spins (e.g. nuclear spins). In this case the
    Pauli matrices of the coupled spins will be calculated, too.

    If no nuclear spin quantum number(s) is (are) provided, the Pauli matrices
    are only set up for the uncoupled spin.

    If the uncoupled spin is not provided, an electron spin of s = 1/2 is used.
    The Pauli matrices of the coupled spins are ordered according to the
    corresponding input array as the tensor products are calculated in exactly
    this order.

    Parameters
    ----------
    spin : float, optional
        Spin quantum number for the spin of the Spinoperator object.
        The default is 1/2.
    coupling_spins : np.ndarray, optional
        Spin quantum number or an array of spin quantum numbers of further
        coupling spins. The default is None.

    Attributes
    ----------
    matrix : np.ndarray
        This array contains three nxn-matrices. The first one is the
        x-spinoperator, second is y-operator and third is z-operator.
    dimension : int
        Hilbert-space-dimension of single spin-operators of each cartesian
        direction.
    matrix_coupling_spins : np.ndarray
        If coupling spins are provided this matrix contains the three Pauli
        matrices for all coupling spins in the order of the input array.
    vector : np.ndarray
        Flattened matrix attribute.

    Examples
    --------
    >>> s = Spinoperator(1/2)
    >>> s.matrix
    array([[[ 0. +0.j ,  0.5+0.j ],
            [ 0.5+0.j ,  0. +0.j ]],
    <BLANKLINE>
           [[ 0. +0.j ,  0. -0.5j],
            [ 0. +0.5j,  0. +0.j ]],
    <BLANKLINE>
           [[ 0.5+0.j ,  0. +0.j ],
            [ 0. +0.j , -0.5+0.j ]]])
    >>> s.dimension
    2

    >>> s = spinoperator(1/2, 1/2)
    >>> print(s.matrix)
    [[[[ 0. +0.j   0. +0.j   0.5+0.j   0. +0.j ]
    [ 0. +0.j   0. +0.j   0. +0.j   0.5+0.j ]
    [ 0.5+0.j   0. +0.j   0. +0.j   0. +0.j ]
    [ 0. +0.j   0.5+0.j   0. +0.j   0. +0.j ]]
    [[ 0. +0.j   0. +0.j   0. -0.5j  0. +0.j ]
    [ 0. +0.j   0. +0.j   0. +0.j   0. -0.5j]
    [ 0. +0.5j  0. +0.j   0. +0.j   0. +0.j ]e vector attribute and the superop attrib
    [ 0. +0.j   0. +0.5j  0. +0.j   0. +0.j ]]
    [[ 0.5+0.j   0. +0.j   0. +0.j   0. +0.j ]
    [ 0. +0.j   0.5+0.j   0. +0.j   0. +0.j ]
    [ 0. +0.j   0. +0.j  -0.5+0.j  -0. +0.j ]
    [ 0. +0.j   0. +0.j  -0. +0.j  -0.5+0.j ]]]]
    >>> print(s.matrix_coupling_spins)
    [[[[ 0. +0.j   0.5+0.j   0. +0.j   0. +0.j ]
    [ 0.5+0.j   0. +0.j   0. +0.j   0. +0.j ]
    [ 0. +0.j   0. +0.j   0. +0.j   0.5+0.j ]
    [ 0. +0.j   0. +0.j   0.5+0.j   0. +0.j ]]
    [[ 0. +0.j   0. -0.5j  0. +0.j   0. +0.j ]
    [ 0. +0.5j  0. +0.j   0. +0.j   0. +0.j ]
    [ 0. +0.j   0. +0.j   0. +0.j   0. -0.5j]
    [ 0. +0.j   0. +0.j   0. +0.5j  0. +0.j ]]
    [[ 0.5+0.j   0. +0.j   0. +0.j   0. +0.j ]
    [ 0. +0.j  -0.5+0.j   0. +0.j  -0. +0.j ]
    [ 0. +0.j   0. +0.j   0.5+0.j   0. +0.j ]
    [ 0. +0.j  -0. +0.j   0. +0.j  -0.5+0.j ]]]]
    """

    def __init__(self, spin=1 / 2, coupling_spins=None):
        """©Stephan Rein, modified by Theresia Quintes"""

        dim_spin = int(2 * spin + 1)

        # Pauli matrices for spin without further coupling spins
        if coupling_spins is None:
            pauli_matrices = self.pauli_matrices(spin)
            self.dimension = dim_spin
            self.matrix = np.array(pauli_matrices, dtype=COMPLEX_TYPE)
            self.build_vector()
            self.build_superoperator()

            return

        # Pauli matrices for spin together with coupling spins
        else:
            # Find number of couplings
            # TODO in try except
            if isinstance(coupling_spins, float) or isinstance(coupling_spins, int):
                number_of_couplings = 1
            else:
                number_of_couplings = len(coupling_spins)
                if number_of_couplings == 1:
                    coupling_spins = coupling_spins[0]
                else:
                    coupling_spins = np.array(coupling_spins)

            # Calculate dimensions
            # TODO  np.array(, dtype=np.int64)?
            dim_couplings = 2 * coupling_spins + 1
            dim_couplings_total = int(np.prod(dim_couplings))
            dim_total = int(dim_spin * dim_couplings_total)

            # Allocate empty arrays
            pauli_coupling_spins = np.zeros(
                (number_of_couplings, 3, dim_total, dim_total), dtype=COMPLEX_TYPE
            )
            pauli_spin = np.zeros((3, dim_total, dim_total), dtype=COMPLEX_TYPE)

            # Pauli matrix for spin
            pauli_uncoupled = self.pauli_matrices(spin)
            pauli_spin = np.kron(
                pauli_uncoupled, np.eye(dim_couplings_total, dtype=FLOAT_TYPE)
            )

            # Pauli matrices for coupling spins
            for n in range(0, number_of_couplings):
                if number_of_couplings == 1:
                    pauli_uncoupled = self.pauli_matrices(coupling_spins)
                    pauli_coupling_spins[n] = np.kron(
                        np.eye(dim_spin, dtype=FLOAT_TYPE), pauli_uncoupled
                    )
                else:
                    pauli_uncoupled = self.pauli_matrices(coupling_spins[n])
                    pauli_tmp = pauli_uncoupled
                    for n_coup in range(0, number_of_couplings):
                        dim = int(dim_couplings[n_coup])
                        if n_coup != n:
                            if n_coup < n:
                                pauli_tmp = np.kron(
                                    np.eye(dim, dtype=FLOAT_TYPE), pauli_tmp
                                )
                            else:
                                pauli_tmp = np.kron(pauli_tmp, np.eye(dim))
                    pauli_coupling_spins[n] = np.kron(
                        np.eye(dim_spin, dtype=FLOAT_TYPE), pauli_tmp
                    )

            self.matrix = pauli_spin
            self.matrix_coupling_spins = pauli_coupling_spins
            self.dimension = dim_total
            self.build_vector()
            self.build_superoperator()

            return

    def pauli_matrices(self, spin: float):
        """
        Create the three Pauli matrices for each cartesian direction for a
        given spin quantum number.

        Parameters
        ----------
        spin : float
            Spin quantum number of spin. Matrix operators are set up for this
            spin quantum number.

        Returns
        -------
        sigma_x : np.ndarray
            Array contains the x-Pauli-matrix.
        sigma_y : np.ndarray
            Array contains the y-Pauli-matrix.
        sigma_z : np.ndarray
            Array contains the z-Pauli-matrix.

        Examples
        --------
        >>> s = Spinoperator(1/2)
        >>> s.pauli_matrices(1/2)
        (array([[0. +0.j, 0.5+0.j],
                [0.5+0.j, 0. +0.j]]),
         array([[0.-0.j , 0.-0.5j],
                [0.+0.5j, 0.-0.j ]]),
         array([[ 0.5+0.j,  0. +0.j],
                [ 0. +0.j, -0.5+0.j]]))

        Notes
        -----
        ©Stephan Rein, modified by Theresia Quintes
        """
        dimension = int(2 * spin + 1)
        dimension = np.array(dimension).astype(int)

        # Preallocate empty arrays
        off_diagonal_elements = np.zeros(dimension - 1, dtype=COMPLEX_TYPE)
        sigma_x = np.zeros((dimension, dimension), dtype=COMPLEX_TYPE)
        sigma_y = np.zeros((dimension, dimension), dtype=COMPLEX_TYPE)
        sigma_z = np.zeros((dimension, dimension), dtype=COMPLEX_TYPE)

        # Calculate off-diagonal elements for sigma_x & sigma_y matrices
        for i in range(0, dimension - 1):
            off_diagonal_elements[i] = np.sqrt((i + 1) * (dimension - 1 - i))

        # Fill Pauli-matrices
        for i in range(0, dimension):
            sigma_z[i, i] = dimension / 2.0 - i - 0.5
            if i + 1 <= dimension - 1:
                sigma_x[i, i + 1] = 0.5 * off_diagonal_elements[i]
                sigma_y[i, i + 1] = 0.5 * 1j * off_diagonal_elements[i]

        sigma_x = sigma_x + np.transpose(np.conjugate(sigma_x))
        sigma_y = sigma_y + np.transpose(np.conjugate(sigma_y))
        sigma_y = np.conjugate(sigma_y)

        return sigma_x, sigma_y, sigma_z

    def get(self, coordinate: str) -> "np.ndarray":
        """
        Get the spin operator matrices for each cartesian direction out of the
        matrix attribute.

        Parameters
        ----------
        coordinate : str
            Coordinate can be either 'x', 'y' or 'z'. The appropriate
            spin operator is chosen.

        Raises
        ------
        ValueError
            If an other character is filld in for coordinate.

        Returns
        -------
        matrix[i] : np.ndarray
            Cartesian operator of S which is chosen by coordinate.

        Examples
        --------
        >>> s = Spinoperator(1/2)
        >>> s.get('x')
        array([[0. +0.j, 0.5+0.j],
               [0.5+0.j, 0. +0.j]])

        """
        # TODO dictionary?
        # BUG coordinate.lower() verwenden
        if coordinate == "x":
            return self.matrix[0]
        elif coordinate == "y":
            return self.matrix[1]
        elif coordinate == "z":
            return self.matrix[2]
        else:
            raise ValueError("Only 'x', 'y' and 'z' are allowed coordinates.")


class Hamiltonian(Operator):
    """
    A Hamiltonian object is the same as an Operator object, but some functions
    are added.

    Parameters
    ----------
    dimension : int
        Dimension of matrix which is generated.

    Attributes
    ----------
    matrix : np.ndarray
        Array containing only zeros. The shape of the array is
        dimension x dimension.
    dimension : int
        Dimension of the matrix.
    vector : np.ndarray
        Flattened matrix. The new dimension is 1 x dimension^2.
    superop : None
        Attribute can be filled by the function build_superoperator.

    """

    def exchange_coupling(
        self, spinoperator1: object, J_ex: float, spinoperator2: object
    ) -> None:
        """
        Calculate interaction hamiltonian of the exchangecoupling of two spins.

        Parameters
        ----------
        spinoperator1 : object
            Spin vector operator of the first interacting spin. This is a
            Spinoperator object.
        J_ex : float
            Exchange coupling constant.
        spinoperator2 : object
            Spin vector operator of the second interacting spin. This is a
            Spinoperator object.

        Attributes
        ----------
        matrix : np.ndarray
            Hamiltonian of exchange coupling in matrix representation.

        Returns
        -------
        None.

        Examples
        --------
        >>> h = Hamiltonian(2)
        >>> j = 5.2
        >>> s = Spinoperator(1/2)
        >>> h.exchange_coupling(s, j, s)
        >>> h.matrix
        array([[-10.4+0.j,  -0. +0.j],
               [ -0. +0.j, -10.4+0.j]])

        """
        if spinoperator1.dimension != spinoperator2.dimension:
            raise IndexError("Spinoperatordimensions must not differ.")

        self.matrix = J_ex * (
            spinoperator1.get("x") @ spinoperator2.get("x")
            + spinoperator1.get("y") @ spinoperator2.get("y")
            + spinoperator1.get("z") @ spinoperator2.get("z")
        )
        self.matrix = self.matrix.astype(COMPLEX_TYPE)

        self.build_vector()
        self.build_superoperator()

        return None

    def microwave_coupling(
        self, omega_mw: float, omega_nut: float, spinoperator: object
    ) -> None:
        """
        Calculate the interaction hamiltonian of a spinoperator with a magnetic
        microwave-field in the rotating frame representation. Therefore, the
        hamiltonian is set up as the sum of the interaction hamiltonian with
        the microwave-field and the offset-hamiltonian which describes the
        shift of all lamor frequencies in rotating frame: H = H_mw - H_off.

        Parameters
        ----------
        omega_mw : float
            Frequency of the microwave-field. Used as rotation frequency of
            the rotating frame.
        omega_nut : float
            Nutation frequency of precession of the spin around microwave-
            field. Calculatable by omega_nut = 1/2*g*mu_B*B_x (all scalars).
        spinoperator : object
            Spin vector operator of the interacting spin. This is a
            Spinoperator object.

        Attributes
        ----------
        matrix : np.ndarray
            Hamiltonian of coupling with the microwave-field in a rotating
            frame in matrix representation.

        Returns
        -------
        None.

        Examples
        --------
        >>> h = Hamiltonian(2)
        >>> s = Spinoperator(1/2)
        >>> h.microwave_coupling(9.75, 1.4, s)
        >>> h.matrix
        array([[ 4.875+0.j, -0.7  +0.j],
               [-0.7  +0.j, -4.875+0.j]])

        """
        self.matrix = (
            spinoperator.get("x") * omega_nut - spinoperator.get("z") * omega_mw
        )

        self.build_vector()
        self.build_superoperator()

        return None

    def zeeman_coupling(
        self, g: "np.ndarray", B_z: float, spinoperator: object
    ) -> None:
        """
        Calculate the hamiltonian of the Zeeman coupling of a spin
        (represented by a spin vector operator) and a magnetic field along the
        z-Axis.

        Parameters
        ----------
        g : np.ndarray
            g-Tensor of electron interacting with the magnetic field in
            xyz-frame. The dimension is 3 x 3.
        B_z : float
            Magnetic field strenght in z-direction. All other directions are
            set to zero!
        spinoperator : object
            Spin vector operator of the interacting spin. This is a
            Spinoperator object.

        Attributes
        ----------
        matrix : np.ndarray
            Hamiltonian of coupling with a static magnetic field in
            z-direction.

        Returns
        -------
        None.

        Examples
        --------
        >>> h = Hamiltonian(2)
        >>> s = Spinoperator(1/2)
        >>> g = Tensor(np.arange(1, 4))
        >>> h.zeeman_coupling(g.matrix, 42.5, s)
        >>> h.matrix
        array([[ 63.75+0.j,   0.  +0.j],
               [  0.  +0.j, -63.75+0.j]])
        """
        self.matrix = odh.create_linear_hamiltonian(g, spinoperator.matrix)
        self.matrix *= B_z
        self.build_vector()
        self.build_superoperator()

        return None

    def dipol_coupling(
        self, spinoperator1: object, D_tensor: "np.ndarray", spinoperator2: object
    ) -> None:
        """
        Calculate the hamiltonian of a dipolar coupling of two spins.

        Parameters
        ----------
        spinoperator1: object
            Spin vector operator of the first interacting spin. This is a
            Spinoperator object.
        D_tensor : np.ndarray
            Dipolar coupling tensor in xyz-frame. The shape is 3 x 3.
        spinoperator2 : object
            Spin vector operator of the first interacting spin. This is a
            Spinoperator object.

        Attributes
        ----------
        matrix : np.ndarray
            Hamiltonian of dipolar coupling.

        Returns
        -------
        None.

        Examples
        --------
        >>> h = Hamiltonian(4)
        >>> s1 = Spinoperator(1/2, 1/2)
        >>> s2 = mt.Spinoperator(1/2, 1/2)
        >>> s2.matrix = s1.matrix_coupling_spins[0]
        >>> d = Tensor(np.arange(1, 4))
        >>> h.dipol_coupling(s1, d.matrix, s2)
        >>> h.matrix
        array([[ 4.5+0.j,  0. +0.j,  0. +0.j, -0.5+0.j],
               [ 0. +0.j,  1.5+0.j,  1.5+0.j,  0. +0.j],
               [ 0. +0.j,  1.5+0.j,  1.5+0.j,  0. +0.j],
               [-0.5+0.j,  0. +0.j,  0. +0.j,  4.5+0.j]])

        """
        if spinoperator1.dimension != spinoperator2.dimension:
            raise IndexError("Spinoperatordimensions must not differ.")

        self.matrix = odh.create_bilinear_hamiltonian(
            spinoperator1.matrix, D_tensor, spinoperator2.matrix
        )
        self.build_vector()
        self.build_superoperator()

        return None
