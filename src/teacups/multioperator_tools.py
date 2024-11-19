import numpy as np

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32

class Multimatrix:
    def __init__(self, dimension, grid_points: int, b_points: 'np.ndarray'):

        self.dimension = dimension
        self.angle_shape = (grid_points, self.dimension,
                            self.dimension)
        self.B_angle_shape = (b_points, grid_points,
                              self.dimension, self.dimension)

        self.matrix = np.zeros(
            (self.dimension, self.dimension), dtype=COMPLEX_TYPE)
        self.angle_matrix = np.zeros((self.angle_shape), dtype=COMPLEX_TYPE)
        self.B_angle_matrix = np.zeros(
            (self.B_angle_shape), dtype=COMPLEX_TYPE)

    def matrix_changed(self) -> None:
        """
        Change all matrix attributes in angle_matrix attribute and in
        B_angle_matrix attribute to the current value of matrix attribute. For
        example this can be used for isotropic operators, whose matrices look
        the same for each B-field point and each angle combination.

        Attributes
        ----------
        angle_matrix : np.ndarray
            Contains the same matrix array for each pair of phi and
            theta angles. The values will be that of the matrix attribute.
        B_angle_matrix: np.ndarray
            Contains the same matrix array for each pair of phi and theta
            angles and for each B-field point. The values will be that of
            the matrix attribute.

        Returns
        -------
        None.

        """
        self.angle_matrix = np.broadcast_to(
            self.matrix, self.angle_shape).copy()
        self.B_angle_matrix = np.broadcast_to(
            self.matrix, self.B_angle_shape).copy()

        return None

    def angle_matrix_changed(self) -> None:
        """
        Change all angle_matrix attributes in B_angle matrix attribute to the
        current value of angle_matrix attribute. For example this can be used
        for operators which are not dependent on different B-fields.

        Attributes
        ----------
        B_angle_matrix: np.ndarray
            Contains the same angle_matrix for each B-field point. The values
            will be that of the angle_matrix attribute.

        Returns
        -------
        None.

        """
        self.B_angle_matrix = np.broadcast_to(
            self.angle_matrix, self.B_angle_shape).copy()

        return None


    def scalar(self, factors: 'np.ndarray') -> None:
        """
        Build the scalar product of the whole B_angle_matrix
        (this means each number in the array) with the product of all factors.

        Parameters
        ----------
        factors : np.ndarray
            1D-Array containing all scalars which shall be multiplied with each
            other and with the B_angle_matrix.

        Attributes
        ----------
        B_angle_matrix: np.ndarray
            Each number in the matrix is multiplied by the product of factors.

        Returns
        -------
        None.

        """
        self.B_angle_matrix *= np.prod(factors)

        return None

    def product(self, scd_matrix: 'np.ndarray', left=False) -> None:
        """
        Build the matrix product of each matrix in B_angle_matrix and a second
        matrix being multiplied either from right (default) or from left side.

        Parameters
        ----------
        scd_matrix : np.ndarray
            Matrix of same dimension as matrix attribute, which shall be
            multiplied with all matrices in B_angle_matrix attribute.
        left : boolean, optional
            If left is set to True the second matrix is multiplied from
            the left side. If left is set to False it is multiplied from
            right side. The default is False.

        Attributes
        ----------
        B_angle_matrix: np.ndarray
            Each matrix attribute is multiplied by the second matrix.

        Returns
        -------
        None.

        """
        if scd_matrix.shape != self.matrix.shape:
            if scd_matrix.shape != self.angle_shape:
                if scd_matrix.shape != self.B_angle_shape:
                    raise IndexError("Multiplied matrix has to have either\
                                     the shape of the matrix, angle_matrix\
                                     or B_angle_matrix attribute.")

        if not left:
            self.B_angle_matrix = self.B_angle_matrix[:, :] @ scd_matrix
        else:
            self.B_angle_matrix = scd_matrix @ self.B_angle_matrix[:, :]

        return None

    def basis_transformation(self, trans: 'np.ndarray', inverse_left=True,
                            orthonormal=False) -> None:
        """
        Change the basis of all matrices in B_angle_matrix.
        trans is the transformation matrix: m = trans^-1 @ m @ trans.

        Parameters
        ----------
        trans : np.ndarray
            Transformation matrix containing arrays of the new basis. The shape
            has to be the same as that of the matrix attribute.
        orthonormal : boolean, optional
            If orthonormal is set to True the basis transforation is done by
            using the adjungate instead of the inverse transformation matrix
            in calculation. This is possible if the old and the new basis
            consist only of orthonormal vectors. The default is False.

        Attributes
        ----------
        B_angle_matrix: np.ndarray
            Each matrix attribute is basis transformed by T.

        Returns
        -------
        None.

        """
        if trans.shape != self.matrix.shape:
            if trans.shape != self.angle_shape:
                if trans.shape != self.B_angle_shape:
                    raise IndexError("Transformation matrix has to have either\
                                     the shape of the matrix, angle_matrix\
                                     or B_angle_matrix attribute.")

        if inverse_left is True:
            if not orthonormal:
                self.B_angle_matrix = (np.linalg.inv(trans) @
                                       self.B_angle_matrix[:, :] @ trans)
            else:
                self.B_angle_matrix = np.conj(trans.T) @ self.B_angle_matrix[:, :]\
                    @ trans

        else:
            if not orthonormal:
                self.B_angle_matrix = trans @ self.B_angle_matrix[:, :] @\
                    (np.linalg.inv(trans))
            else:
                self.B_angle_matrix = trans @ self.B_angle_matrix[:, :]\
                    @ np.conj(trans.T)


class Multioperator_(Multimatrix):
    def __init__(self, dimension, grid_points: int, b_points: 'np.ndarray'):
        Multimatrix.__init__(self, dimension, grid_points, b_points)
        self.B_angle_vector = self.build_vector()
        self.B_angle_superop = self.build_superoperator()

    def build_vector(self) -> None:
        """
        Changing the dimension of each matrix operator attribute in
        B_angle_matrix to a 1D.vector using .reshape()-function for numpy
        arrays. Lines are simply written in only one long line.

        Attributes
        ----------
        B_angle_vector np.ndarray
            Each matrix attribute is flattend to 1d array.

        Returns
        -------
        None.

        """
        self.B_angle_vector = self.B_angle_matrix.reshape(
            (self.B_angle_matrix.shape[0], self.B_angle_matrix.shape[1],
             self.dimension**2))

        return None

    def build_superoperator(self, swap=False) -> None:
        """
        Double the dimension of each matrix operator in B_angle_matrix
        (in case of use in a higher dimensional space). Tensor product of
        operator with unit matrix of the same dimension is built using the
        numpy.kron function.

        Parameters
        ----------
        swap : Boolean, optional
            If swap is False the tensor product is built with matrix on left
            and unit matrix on right site. If left is set to True unit matrix
            is on the left side and matrix attribute on the right.
            The default is False.

        Attributes
        ----------
        B_angle_superop : np.ndarray
        array
            Result of tensor product of matrix and unit matrix for each B-field
            point and each angle combination.

        Returns
        -------
        None.

        """
        eye = np.eye(self.dimension, dtype=self.B_angle_matrix.dtype)
        if not swap:
            self.B_angle_superop = np.kron(self.B_angle_matrix[:, :], eye)
        else:
            self.B_angle_superop = np.kron(eye, self.B_angle_matrix[:, :])

        return None



class Multioperator(Multioperator_):
    """
    An object of class Multioperator contains matrix attributes of
    different dimensions (see below). Furthermore it contains a tensor, a
    spinoperator and a magnetic fiels vector setting the dimensions.
    Spinoperator, rotated tensor and magnetic field vector are given to
    initialisation function.

    Parameters
    ----------
    spinop : object
        Initialised attribute of the class Spinoperator. Therefore, it
        contains the spin of the system.
    grid_points : int
        Number of different orientations that shall be calculated. This
        determines the angle-dimension.
    B : np.ndarray
        Magnetic field vector. The length of this vector determines the
        B-dimension of the multioperator. This is a 1D-array

    Attributes
    ----------
    spinop : object
        The spinop attribute contains the spinoperator given.
    B : np.ndarray
        The B attribute contains the magnetic field vector given.
    dimension : int
        The cartesian spinoperator matrices have the shape (dimension,
        dimension)
    angle_shape : tuple
        Dimension of angles and cartesian spinoperator matrices. The tuple
        order is: (grid_points, dimension, dimension)
    B_angle_shape : tuple
        Dimension of B-variation, angle-variations and cartesian
        spinoperator matrices. The tuple order is: (len(B), grid_points,
        dimension, dimension).
    matrix : np.ndarray
        The matrix  attribute contains only zeros but values can be
        changed. Represents the operator matrix for a single orientation
        and a single B-field-value. Its dimension is dimension x dimension.
    angle_matrix : np.ndarray
        Contains a dimension x dimension array for each pair of phi and
        theta angles. Its shape is equal to angle_shape.
    B_angle_matrix: np.ndarray
        Contains a angle_matrix for each B-fieldpoint. Its shape is equal to
        B_angle_shape.
    B_angle_vector: None
        After initialisation it is NoneType but attribute can be changed
        later by the function build_vector. Afterwards it represents the
        B_angle_matrix in vector-dimension.
    B_angle_superop: None
        After initialisation it is NoneType but attribute can be changed
        later by the function build_superoperator. Afterwards it represents
        the B_angle_matrix in superoperator-dimension.

    """

    def __init__(self, spinop: object, grid_points: int, B: 'np.ndarray'):
        Multioperator_.__init__(self, spinop.dimension, grid_points, len(B))
        self.spinop = spinop
        self.B = B

    def create_linear_operator(self, tensor) -> None:
        """
        Create a linear interaction hamiltonian (e.g. Zeeman interaction)
        between the spinoperator attribute spinop and the interaction matrix
        tensor in case that the magnetic field vector is defined in
        z-direction. This is done for each angle combination of phi and theta
        by using the rotated tensors from tensor.multirot. The interaction
        matrices are saved to angle_matrix attribute.

        Parameters
        ----------
        tensor : object
            Tensor of for the linear interaction. E.g. for the Zeeman
            interaction this would be the g-tensor. This parameter has to be an
            object from class tensor. The multiroation-function has to be
            carried out.

        Attributes
        ----------
        angle_matrix : np.ndarray
            Contains the linear interaction operators for each angle
            combination of phi and theta.

        Returns
        -------
        None.

        """
        sx = np.broadcast_to(self.spinop.get('x'), (self.angle_shape))
        sy = np.broadcast_to(self.spinop.get('y'), (self.angle_shape))
        sz = np.broadcast_to(self.spinop.get('z'), (self.angle_shape))
        tx = tensor.multirot[:, 0, 2, np.newaxis, np.newaxis]
        ty = tensor.multirot[:, 1, 2, np.newaxis, np.newaxis]
        tz = tensor.multirot[:, 2, 2, np.newaxis, np.newaxis]
        self.angle_matrix = sx*tx + sy*ty + sz*tz
        self.angle_matrix = self.angle_matrix.astype(COMPLEX_TYPE)

        self.angle_matrix_changed()
        self.build_vector()
        self.build_superoperator()

        return None

    def create_bilinear_operator(self, tensor, spinop2: object) -> None:
        """
        Create a bilinear interaction hamiltonian (e.g. dipol interaction)
        between the spinoperator attribute spinop, the interaction matrix
        tensor and a second spinoperator given to the function. The interaction
        hamiltonian is built for all angle combinations of phi and theta by
        using the rotated tensors from tensor.multirot. The interaction
        matrices are saved to angle_matrix attribute.

        Parameters
        ----------
        tensor : object
            Tensor of for the bilinear interaction. E.g. for the dipolar
            interaction this would be the D-tensor. This parameter has to be an
            object from class tensor. The multiroation-function has to be
            carried out.
        spinop2 : object
            Spin vector operator of the second interacting spin. This has to
            be an object of class Spinoperator.

        Attributes
        ----------
        angle_matrix : np.ndarray
            Contains the bilinear interaction operators for each angle
            combination of phi and theta.

        Returns
        -------
        None.

        """
        if self.spinop.dimension != spinop2.dimension:
            raise IndexError("Spinoperator dimensions must not differ.")

        sx1 = np.broadcast_to(self.spinop.get('x'), (self.angle_shape))
        sy1 = np.broadcast_to(self.spinop.get('y'), (self.angle_shape))
        sz1 = np.broadcast_to(self.spinop.get('z'), (self.angle_shape))
        S1 = np.array([sx1, sy1, sz1])

        sx2 = np.broadcast_to(spinop2.get('x'), (self.angle_shape))
        sy2 = np.broadcast_to(spinop2.get('y'), (self.angle_shape))
        sz2 = np.broadcast_to(spinop2.get('z'), (self.angle_shape))

        S2 = np.array([sx2, sy2, sz2])

        txx = tensor.multirot[:, 0, 0, np.newaxis, np.newaxis]
        txy = tensor.multirot[:, 0, 1, np.newaxis, np.newaxis]
        txz = tensor.multirot[:, 0, 2, np.newaxis, np.newaxis]

        tyx = tensor.multirot[:, 1, 0, np.newaxis, np.newaxis]
        tyy = tensor.multirot[:, 1, 1, np.newaxis, np.newaxis]
        tyz = tensor.multirot[:, 1, 2, np.newaxis, np.newaxis]

        tzx = tensor.multirot[:, 2, 0, np.newaxis, np.newaxis]
        tzy = tensor.multirot[:, 2, 1, np.newaxis, np.newaxis]
        tzz = tensor.multirot[:, 2, 2, np.newaxis, np.newaxis]

        tensor_ges = np.array([[txx, txy, txz],
                               [tyx, tyy, tyz],
                               [tzx, tzy, tzz]])

        ham = 0
        for i in range(0, 3):
            for j in range(0, 3):
                ham = np.add(ham, S1[i] @ (tensor_ges[i, j] * S2[j]))
        self.angle_matrix = ham
        self.angle_matrix = self.angle_matrix.astype(COMPLEX_TYPE)

        self.angle_matrix_changed()
        self.build_vector()
        self.build_superoperator()

        return None

    def exchange_coupling(self, J_ex: float, spinop2: object) -> None:
        """
        Create an isotropic exchange coupling hamiltonian of two spins. Matrix
        attribute is changed and afterwards filled in angle_matrix and
        B_angle_matrix by using the function matrix_changed().

        Parameters
        ----------
        J_ex : float
            Exchange coupling constant.
        spinop2 : object
            Spin vector operator of the second interacting spin. This has to
            be an object of class Spinoperator.

        Attributes
        ----------
        matrix : np.ndarray
            The exchange coupling hamiltonian of the spin and a second spin.
        angle_matrix : np.ndarray
            Contains the new built values of the matrix attribute
            for each angle combination.
        B_angle_matrix: np.ndarray
            Contains the new built values of the matrix attribute
            for each angle combination and each B-field point.

        Returns
        -------
        None.

        """
        if self.spinop.dimension != spinop2.dimension:
            raise IndexError("Spinoperator dimensions must not differ.")

        self.matrix = -J_ex*(0.5*np.eye(self.dimension) + 2
                             * (self.spinop.get('x')@spinop2.get('x')
                                + self.spinop.get('y')@spinop2.get('y')
                                + self.spinop.get('z')@spinop2.get('z')))
        self.matrix = self.matrix.astype(COMPLEX_TYPE)

        self.matrix_changed()
        self.build_vector()
        self.build_superoperator()


        return None

    def microwave_coupling(self, omega_nut: float, omega_mw: float) -> None:
        """
        Calculate an isotropic interaction hamiltonian of spinoperator with a
        magnetic microwave-field in the rotating frame representation.
        Therefore, the hamiltonian is set up as the sum of the interaction
        hamiltonian with the microwave-field and the offset-hamiltonian which
        describes the shift of all lamor frequencies in rotating frame:

        .. math::
            H = H_\mathrm{mw} - H_\mathrm{off}.

        The matrix attribute is changed and afterwards filld in
        angle_matrix and B_angle_matrix by using the function matrix_changed().

        Parameters
        ----------
        omega_mw : float
            Frequency of the microwave-field. Used as rotation frequency of
            the rotating frame.
        omega_nut : float
            Nutation frequency of precession of the spin around microwave-
            field. Calculatable by omega_nut = 1/2*g*mu_B*B_x (all scalars).

        Attributes
        ----------
        matrix : np.ndarray
            The microwave coupling hamiltonian of the spin and a microwave
            field in rotating frame representation.
        angle_matrix : np.ndarray
            Contains the new built values of the matrix attribute
            for each angle combination.
        B_angle_matrix: np.ndarray
            Contains the new built values of the matrix attribute
            for each angle combination and each B-field point.

        Returns
        -------
        None.

        """
        self.matrix = self.spinop.get('x') * \
            omega_nut - self.spinop.get('z')*omega_mw

        self.matrix_changed()
        self.build_vector()
        self.build_superoperator()

        return None

    def zeeman_coupling(self, g_tensor) -> None:
        """
        Claculate the hamiltonian of the zeeman coupling. Therefore, the
        interaction matrix is calculated by using the function
        create_linear_operator() for each angle combination. Afterwards each
        element from B attribute is multiplied to all hamiltonians
        (for each angle combination). The matrices
        (different for all angle pairs and B-field points) are saved
        to B_angle_matrix.

        Parameters
        ----------
        tensor : object
            g-Tensor of for the Zeeman interaction. This parameter has to be an
            object from class tensor. The multiroation-function has to be
            carried out.

        Attributes
        ----------
        B_angle_matrix: np.ndarray
            Contains the zeeman interaction hamiltonian for each magnetic field
            point and each angle combination of phi and theta.

        Returns
        -------
        None.

        """
        self.create_linear_operator(g_tensor)
        self.angle_matrix_changed()
        self.B_angle_matrix *= np.broadcast_to(
            self.B[:, np.newaxis, np.newaxis, np.newaxis],
            self.B_angle_shape)

        self.build_vector()
        self.build_superoperator()

        return None
