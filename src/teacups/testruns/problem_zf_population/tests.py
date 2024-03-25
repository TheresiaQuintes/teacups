#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:04:05 2024

@author: theresia
"""
import teacups.matrix_tools as mt
import teacups.orientation_dependent_ham as odh
import numpy as np
import sympy as sym


def euler_matrix(phi: float, theta: float,
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
    return eulermatrix


S = mt.Spinoperator(1)
S = S.matrix

h_d = S[2]@S[2] - 1/3*(S[0]@S[0]+S[1]@S[1]+S[2]@S[2])
h_z = S[2]


s_x = sym.Matrix(S[0])
s_y = sym.Matrix(S[1])
s_z = sym.Matrix(S[2])

D, E, B = sym.symbols('D E B')

D_ten = sym.Matrix([[D+E, 0, 0], [0, -2*D, 0], [0, 0, D-E]])
euler = euler_matrix(1, 1)
D_ten = euler@D_ten@euler.T
h_d_ten = odh.create_bilinear_hamiltonian(S, D_ten, S)
h_d_ten = sym.Matrix(h_d_ten)


h_d = D*(s_z**2-1/3*(s_x**2+s_y**2+s_z**2))+E*(s_x**2-s_y**2)

euler = euler_matrix(1, 1)
h_d_rot = euler.T@h_d@euler


eigvec, h_d_xyz = h_d.diagonalize()


h_z = B*s_z

h_z_xyz = eigvec**-1*h_z*eigvec
h_z_xyz_rot = euler@h_z_xyz@euler.T  # so rum oder anders?

h = h_d+h_z

h_xyz = eigvec**-1*h*eigvec
