import numpy as np
import scipy.ndimage as snd


def voigt_convolution(width: float, spectrum: 'np.ndarray') -> 'np.ndarray':
    """
    Calculate the Voigt profile of a Lorentzian function by convolution of the
    existing Lorentzian function and a Gaussian distribution with a given FWHH.
    A 2D-spectrum is convolved with the Gaussian function along its second
    axis, which should be the B-field-axis.

    Parameters
    ----------
    width : float
        Gaussian line width, standard deviation. Has to have the same
        unit as the second axis of the spectrum.
    spectrum : np.ndarray
        2D-Array containing values of the ordinate of a spectrum.
        In case of TREPR spectra this would be the intensity for each time and
        magnetic field point. The convolution is carried out along the
        second axis.

    Returns
    -------
    spectrum_conv : np.ndarray
        Array containing convoluted values of the ordinate. In case of TREPR
        spectra this would be the covoluted intensities of the spectrum for
        each magnetic field and time point. Spectrum_conv is computed using the
        convolution alogrithm of SciPy.

    """

    spectrum_conv = snd.gaussian_filter(spectrum, [0, width])

    return spectrum_conv


def generalized_pascal(n: int, spin: float) -> 'np.ndarray':
    """
    Calculate relative intensities of a hyperfine pattern which results from n
    identical nuclei with a given spin quantum number.

    Parameters
    ----------
    n : int
        Numer of coupling nuclei.
    spin : float
        Spin quantum number of coupling nuclei. Can be either spin half or
        integer spin.

    Returns
    -------
    intensities : np.ndarray
        1D-Array containing intensities of hyperfine pattern. Its length is
        (2*spin*n+1)

    Examples
    --------
    >>> generalized_pascal(2, 1/2)
    array([1., 2., 1.])
    >>> generalized_pascal(0, 1/2)
    array([1.])

    """
    if type(n) is not int:
        raise ValueError("Use only integer for number of nuclei!")
    elif ((2*spin) % 1) != 0.:
        raise ValueError("Use only integer or half numbers for spin!")

    # no nucleus
    elif n == 0:
        intensities = np.array([1.])
        return intensities

    # nucleus without spin
    elif spin == 0:
        intensities = np.array([1.])
        return intensities

    else:
        s0 = int(2*spin*n+1)
        intensities = np.zeros((n, s0))
        intensities[0, 0:int(2*spin+1)] = 1
        spin2 = 2*spin
        for i in range(1, n):
            for j in range(0, s0):
                if j+spin2 >= s0:
                    ub = int(s0-1)
                else:
                    ub = int(j+spin2)
                if j-spin2 < 0:
                    lb = 0
                else:
                    lb = int(j-spin2)
                intensities[i, j] = np.sum(
                    intensities[i-1, lb:ub-int(2*spin)+1])
        return intensities[n-1, :]


def hyperfine_convolution(nucleus_number: int, spin: float, aiso: float,
                          B: 'np.ndarray', signal: 'np.ndarray',
                          gausswidth: float) -> 'np.ndarray':
    """
    Convolve an EPR-signal with the hyperfine pattern of a given number of
    nuclei with given spin and isotropic coupling constant.

    Parameters
    ----------
    nucleus_number : int
        Number of coupling spins.
    spin : float
        Spin quantum number of coupling nuclei. Can be either spin half or
        integer spin.
    aiso : float
        Isotropic coupling constant.
    B : np.ndarray
        Magnetic field array (1D).
    signal : np.ndarray
        Signal array. Has to have the same dimension as B.
    gausswidth : float
        Anisotropic gaussian lineshape. Voigt convolution of the hyperfine
        pattern will be carried out before the signal is convoluted.

    Returns
    -------
    signal_convolve : np.ndarray
        Signal convoluted with the hyperfine template. Has the same dimension
        as B.

    """

    # generate hyperfine pattern from pascals triangle
    hyperfine_template = generalized_pascal(nucleus_number, spin)

    # find center of magnetic field array
    if len(B) % 2 == 0:
        B_center = int(len(B)/2)
    else:
        B_center = int(len(B)/2 + 0.5)

    # generate a delta spectrum with the hyperfine pattern and distances aiso
    intensities = np.zeros(len(B))
    index = np.zeros(len(hyperfine_template))

    hyperfine_template_center = (len(hyperfine_template) - 1)/2

    x = 0
    while x < len(hyperfine_template):
        index[x] = B[B_center] + aiso*(x - hyperfine_template_center)
        x += 1

    index = np.digitize(index, B)

    z = 0
    for y in index:
        intensities[y] = hyperfine_template[z]
        z += 1

    # convolve hyperfinepattern with anisotropic line with
    intensities = snd.gaussian_filter(intensities, gausswidth)

    # convolve signal with intensities
    signal_convolve = snd.convolve1d(signal, intensities)

    return signal_convolve
