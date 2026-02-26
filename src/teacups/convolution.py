import numpy as np
import scipy.ndimage as snd


def voigt_convolution(sigma_time: float, width: float, spectrum: 'np.ndarray') -> 'np.ndarray':
    """
    Calculate the Voigt profile of a Lorentzian function by convolution of the
    existing Lorentzian function and a Gaussian distribution with a given FWHH.
    A 2D-spectrum is convolved with the Gaussian function along its second
    axis, which should be the B-field-axis. Additional the spectrum is convolved
    with a second Gaussian distribution along the first axis (time-axis) for
    simulating the time resoltuion of the signal.

    Parameters
    ----------
    width_time : float
        Standardderivation of the Gauß-Filter, Time resolution of the signal.
        In pixels.
    width : float
        Gaussian line width, standard deviation. In pixels.
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

    spectrum_conv = snd.gaussian_filter(spectrum, [sigma_time, width])

    return spectrum_conv
