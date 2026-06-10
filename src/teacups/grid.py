import numpy as np
from teacups.epr_grid import Grid


def sphere_fibonacci_grid_points(ng: int, hemisphere=True) -> "np.ndarray":
    """
    Calculate Fibonacci spiral gridpoints on a hemisphere or full sphere.

    Parameters
    ----------
    ng : int
        Number of points that shall be calculated.
    hemisphere : boolean, optional
        Define wether a full or a hemisphere is calculated. Default is True.

    Returns
    -------
    xg : np.ndarray
        Coordinates of the desired number of grid points. The three cartesian
        coordinates are given. The shape of the array is 3xng.

    Licensing
    ---------

      This code is distributed under the GNU LGPL license.
      https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid.py

    Modified
    --------

      15 May 2015

    Author
    ------

      John Burkardt

    Reference
    ---------

      Richard Swinbank, James Purser,
      Fibonacci grids: A novel approach to global modelling,
      Quarterly Journal of the Royal Meteorological Society,
      Volume 132, Number 619, July 2006 Part B, pages 1769-1793.

    """
    if hemisphere is True:
        ng *= 2

    phi = (1.0 + np.sqrt(5.0)) / 2.0

    theta = np.zeros(ng)
    sphi = np.zeros(ng)
    cphi = np.zeros(ng)

    for i in range(0, ng):
        i2 = 2 * i - (ng - 1)
        theta[i] = 2.0 * np.pi * float(i2) / phi
        sphi[i] = float(i2) / float(ng)
        cphi[i] = np.sqrt(float(ng + i2) * float(ng - i2)) / float(ng)

    xg = np.zeros((ng, 3))

    for i in range(0, ng):
        xg[i, 0] = cphi[i] * np.sin(theta[i])
        xg[i, 1] = cphi[i] * np.cos(theta[i])
        xg[i, 2] = sphi[i]

    if hemisphere is True:
        grid = xg[0 : int(ng / 2)]
    else:
        grid = xg
        pass
    return grid


def cartesian2sphereical(xyz: "np.ndarray") -> "np.ndarray":
    """
    Convert a set of three cartesian coordinates (x, y and z) to a set of
    three spherical coordinates (r, theta and phi).

    Parameters
    ----------
    xyz : np.ndarray
        This array contains n sets of cartesian coordinates x, y and z and its
        shape is (nx3)

    Returns
    -------
    rtp : np.ndarray
        This array contains the transformed sets of xyz. Its shape is (nx3).

    """
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2), xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    rtp = np.array([r, theta, phi]).T
    return rtp


def spherical2cartesian(rtp: "np.ndarray") -> "np.ndarray":
    """
    Convert a set of three spherical coordinates (r, theta and phi) to a set of
    three cartesian coordinates (x, y and z).

    Parameters
    ----------
    rtp : np.ndarray
        This array contains n sets of spherical coordinates r, theta and phi
        and its shape is (nx3).

    Returns
    -------
    xyz : np.ndarray
        This array contains n sets of cartesian coordinates x, y and z. Its
        shape is (nx3).

    """
    r = rtp[:, 0]
    t = rtp[:, 1]
    p = rtp[:, 2]
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)

    xyz = np.array([x, y, z]).T
    return xyz


def fibonacci_grid(grid_points: int) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Get a number (grid_points) of angle pairs theta-phi describing points
    equally distributet on a fibonacci sphere. The radius is 1.

    Parameters
    ----------
    grid_points : int
        Number of points on the sphere.

    Returns
    -------
    theta : np.ndarray
        1D-Array with grid_points theta angles. Each angle is the theta-part
        of the spherical coordinate of each point.
    phi : np.ndarray
        1D-Array with grid_points phi angles. Each angle is the phi-part of the
        spherical coordinate of each point.

    """
    xyz = sphere_fibonacci_grid_points(grid_points)
    rtp = cartesian2sphereical(xyz)
    theta = rtp[:, 1]
    phi = rtp[:, 2]
    return theta, phi


def sophe_grid(
    grid_size: int, sym: str
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Calculate the angles phi and theta of a set of unique orientations on a
    sphere. The grid used is called SOPHE grid (see: D. Wang,
    G. R. Hanson J.Magn.Reson. A, 117, 1-8 (1995)
    https://doi.org/10.1006/jmra.1995.9978) or Y. Kurihara, Monthly Weather
    Review 93(7), 399-415 (July 1965)
    https://doi.org/10.1175/1520-0493(1965)093<0399:NIOTPE>2.3.CO;2).
    The grid is set up by the epr_grid module, which is written by Florian
    Quintes.

    Parameters
    ----------
    grid_size : int
        Number of points between theta=0 and theta=pi/2.
    sym : str
        Point group symmetry. "C1" returns the full sphere, other point groups
        result in smaller parts of the sphere.

    Returns
    -------
    phi : np.ndarray
        Set of spherical angles phi for all orientations.
    theta : np.ndarray
        Set of spherical angles theta for all orientations.
    weights : np.ndarray
        Associated weights for each orientation.

    """
    grid = Grid("SOPHE", point_group=sym, knots=grid_size)
    spherical = grid.get_grid(sym, cartesian=False)
    weights = grid.get_areas()
    theta, phi = spherical[:, 1], spherical[:, 2]

    return theta, phi, weights
