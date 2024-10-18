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
