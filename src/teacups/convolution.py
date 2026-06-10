import numpy as np
import scipy.ndimage as snd


def voigt_convolution(
    sigma_time: float, width: float, spectrum: "np.ndarray", extend_t=False
) -> "np.ndarray":
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
    extend_t : bool
        If true the time axis is extended by a zero-baseline with
        5*time_sigma pixels. Default is False.

    Returns
    -------
    spectrum_conv : np.ndarray
        Array containing convoluted values of the ordinate. In case of TREPR
        spectra this would be the covoluted intensities of the spectrum for
        each magnetic field and time point. Spectrum_conv is computed using the
        convolution alogrithm of SciPy.

    """
    pad_t = int(5 * sigma_time)
    spec_pad = np.pad(
        spectrum, ((pad_t, 0), (0, 0)), mode="constant", constant_values=0
    )
    spec_conv_pad = snd.gaussian_filter(spec_pad, [sigma_time, width])

    if extend_t:
        spectrum_conv = spec_conv_pad
    else:
        spectrum_conv = spec_conv_pad[pad_t:, :]

    return spectrum_conv


def extend_time_axis(t: np.ndarray, spec: np.ndarray) -> "np.ndarray":
    """
    If the time axis is shorter than the time-dimension of the spectrum it gets
    extended by negative values.

    Parameters
    ----------
    t : np.ndarray
        Time axis.
    spec : np.ndarray
        2D-Array containing values of the ordinate of a spectrum.
        In case of TREPR spectra this would be the intensity for each time and
        magnetic field point. The convolution is carried out along the
        second axis.

    Returns
    -------
    t_extended : np.ndarray
        Time axis. If it has been shorter than the first dimension of spec
        before it is extended by negative values.

    """
    if len(t) < spec.shape[0]:
        dt = t[1] - t[0]
        diff = spec.shape[0] - len(t)

        t_neg = np.arange(-diff, 0) * dt
        t_extended = np.concatenate((t_neg, t))

        return t_extended

    else:
        return t
