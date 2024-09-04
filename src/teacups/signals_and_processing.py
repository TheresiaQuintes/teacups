import numpy as np
import teacups.grid as gri
import teacups.convolution as con
from scipy.signal import find_peaks
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import cpu_count, Pool

COMPLEX_TYPE = np.complex64


def propagation(sys: object, opt: object, cal: object) -> None:
    """
    Calculate a time propagation operator matrix for a given hamiltonian.
    It can be chosen in which space (hilbert or liouville) the operator will
    be set up. A relaxation operator is taken into account.

    Parameters
    ----------
    sys : object
        Contains parameters of the spin system. This function uses the
        relaxation time attributes T_relax_1 and T_relax_2 (if space
        is set to 'liouville') and decay (if space ist set to 'hilbert').
    opt : object
        Contains simulation option parameters. This function uses opt.space
        to choose in which space the propagation matrix is calculated.
    cal : object
        Contains calculated results during the simulation. This function uses
        cal.ham (the hamiltonian) and cal.t (time space).

    Attributes
    ----------
    cal.propagation : np.ndarray
        Time propagation operator. If opt.space is hilbert the dimension of
        the matrix will be B x grid_points x 4 x 4.
        If opt.space is liouville the dimension of the matrix will be
        B x grid_points x 16 x 16. It contains a single propagation
        operator which has to be used on the density matrix t_points times in
        a row.

    Returns
    -------
    None.

    """

    if opt.space == 'hilbert':
        step = cal.t[1]-cal.t[0]
        propagation = np.zeros(cal.ham.shape, dtype=COMPLEX_TYPE)

        eigval, vec = np.linalg.eigh(cal.ham)
        exp_arg = 1j*eigval

        n = propagation.shape[-1]
        propagation[:, :, range(n), range(n)] = np.exp(exp_arg*step)
        print('exponential ready...')

        propagation = vec @ propagation @ np.conj(
            np.transpose(vec, (0, 1, 3, 2)))
        print('propagator ready...')

    elif opt.space == 'liouville':
        step = cal.t[1]-cal.t[0]

        cal.ham_superop *= -1j*step
        print('superoperator ready...')

        eigval, vec = np.linalg.eig(cal.ham_superop)
        print('eigenvalues ready...')

        propagation = vec @ (np.exp(eigval)[:, :, np.newaxis] *
                             np.eye(eigval.shape[-1], dtype=COMPLEX_TYPE))\
            @ np.linalg.inv(vec)
        print('propagator ready...')
    else:
        print('opt.space has to be either hilbert or liouville')

    cal.propagation = np.array(propagation)

    return


def make_signal(exp: object, opt: object, cal: object) -> None:
    """
    Calculate the signal of a transient EPR experiment. Signals will be given
    back as an array containing the intensity in abitrary units for each
    combination of time, magnetic field and orientation.

    It can be chosen if the signal will be calculated in liouville or in
    hilbert space.
    In lioville space optionally (if opt.pop_evolution is set to True) the
    time evolution of the population of the eigenstates is calculated.
    The signal is returned in cal.signal, the population evolution in
    cal.pop_evolution.

    Parameters
    ----------
    exp : object
        Contains experimental parameters. This function uses exp.B_z.
    opt : object
        Contains simulation option parameters. This function uses opt.space
        for choosing the space in which the signal shall be calculated
        (hilbert or liouville) and opt.grid_points. In liouville space the
        attribute opt.pop_evolution is necessary (True or Flase) to choose if
        the population evolution shall be calculated. Further the number of
        desired  cpu cores has to be given to opt.cpu_cores.
    cal : object
        Contains the results calculated during the simulation. This function
        uses the space cal.t (e.g. built by the function
        set_up_spaces), cal.propagation (propagation matrix see above), cal.rho
        (see set_up_density_matrix) and cal.observable (see set_up_observable).

    Attributes
    ----------
    cal.signal : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z and all orientation points.
        So the shape is len(t)xlen(B)xgrid_points.
    cal.pop_evolution : np.array
        Conrtains the populations of all eigenstates as a function of time.
        As eigenstates the states of the first magnetic field point and the
        first angle point are chosen.

    Returns
    -------
    None.

    """
    cal.signal = np.zeros((len(cal.t), len(exp.B_z), opt.grid_points),
                          dtype=COMPLEX_TYPE)
    time_evolution_multicore(opt, cal)

    # calculate population evolution
    if opt.pop_evolution is True and opt.space == 'liouville':
        print("starting population evolution...")
        rho_prop = cal.rho[1, 0, :, np.newaxis]
        cal.pop_evolution = []
        for i in range(0, len(cal.t)):
            pop_matrix = np.conj(np.transpose(cal.eigvec[0, 0])) @\
                rho_prop.reshape((cal.s.dimension, cal.s.dimension)) @\
                cal.eigvec[0, 0]
            pops = np.diag(pop_matrix)
            cal.pop_evolution.append(pops)
            rho_prop = cal.propagation[0, 0]@rho_prop
        cal.pop_evolution = np.array(cal.pop_evolution)
        print("population evolution ready...")

    return None


def time_evolution_hilbert(cal: object) -> "np.ndarray":
    """
    Calculate the time evoluted signal in hilbert space.

    Parameters
    ----------
    cal : object
        Contains premier calculated results. This function uses the propagator
        cal.propagation, the time axis cal.t, the density matrix cal.rho and
        the observable operator cal.observable.

    Returns
    -------
    cal.signal : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z and all orientation points.
        So the shape is len(t)xlen(B)xgrid_points.

    """
    rho_prop = cal.rho.copy()
    propagation_invers = np.linalg.inv(cal.propagation)
    for i in tqdm(range(len(cal.t))):
        cal.signal[i] = np.trace(
            (rho_prop @ cal.observable), axis1=-1, axis2=-2)
        rho_prop = propagation_invers @ rho_prop @ cal.propagation
    return cal.signal


def time_evolution_liouville(cal: object) -> "np.ndarray":
    """
    Calculate the time evoluted signal in liouville space.

    Parameters
    ----------
    cal : object
        Contains premier calculated results. This function uses the propagator
        cal.propagation (in superoperator dimension), the time axis cal.t,
        the density matrix cal.rho (as a vector) and the observable operator
        cal.observable (as a vector).

    Returns
    -------
    cal.signal : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z and all orientation points.
        So the shape is len(t)xlen(B)xgrid_points.

    """
    rho_prop = cal.rho[:, :, :, np.newaxis]
    for i in tqdm(range(0, len(cal.t))):
        cal.signal[i] = np.dot(rho_prop[:, :, :, 0], cal.observable)
        rho_prop = cal.propagation@rho_prop
    return cal.signal


def time_evolution_multicore(opt: object, cal: object) -> None:
    """
    Calculate the time evolution of a signal on multiple cpu-cores. It can be
    chosen in the calculation is done in hilbert or in liouville space.

    Parameters
    ----------
    opt : object
        Simulation options object. This function uses opt.space to choose the
        calculation space and opt.cpu_cores to determine the number of cores
        used for calculation. Further opt.grid_points has to be given.
    cal : object
        Previously calculated results. This function uses the propagator
        cal.propagation, the time axis cal.t, the density matrix cal.rho and
        the observable operator cal.observable.
    Attributes
    ----------
    cal.signal : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z and all orientation points.
        So the shape is len(t)xlen(B)xgrid_points.

    Returns
    -------
    None

    """
    if opt.space == 'hilbert':
        multicore_sim = multicore(time_evolution_hilbert)
    elif opt.space == 'liouville':
        multicore_sim = multicore(time_evolution_liouville)
    cal.signal = multicore_sim(opt, cal)
    return


def multicore(time_evolution_function: callable) -> callable:
    """
    Using multiprocessing.Pool() with starmap() for parallel computing of
    a time evolution. The spectrum is split along the orientations-axis.

    Parameters
    ----------
    time_evolution_function : callable
        Time evolution function dependent on the calculations object cal.

    Returns
    -------
    multicore_wrapper : callable
        The origin time evolution callable as multicore version.

    """

    def multicore_wrapper(opt: object, cal: object) -> "np.ndarray":
        """
        Change a time_evolution_function for using multiple cpu cores.

        Parameters
        ----------
        opt : object
            Simulation options. This function uses opt.grid_points and
            opt.cpu_cores. If opt.cpu_cores = 0 all available cores are used.
        cal : object
            Contains previously calculated arrays that are needed for
            time propagation namely cal.propagation, cals.rho, cal.signal and
            cal.observable.
        Returns
        -------
        signal : np.ndarray
            This matrix contains the intensities in abitrary units of a
            transient epr spectrum for all time points t in cal.t and all
            magnetic field points in exp.B_z and all orientation points.
            So the shape is len(t)xlen(B)xgrid_points.

        """
        if opt.cpu_cores == 0:
            opt.cpu_cores = cpu_count()

        points_per_core = opt.grid_points // opt.cpu_cores

        Cal_list = np.empty(opt.cpu_cores, dtype=object)
        for core in range(opt.cpu_cores):
            Calculations = deepcopy(cal)
            start = core * points_per_core
            if core+1 < opt.cpu_cores:
                end = (core+1) * points_per_core
                Calculations.propagation = cal.propagation[:, start:end]
                Calculations.rho = cal.rho[:, start:end]
                Calculations.signal = cal.signal[:, :, start:end]
            else:
                Calculations.propagation = cal.propagation[:, start:]
                Calculations.rho = cal.rho[:, start:]
                Calculations.signal = cal.signal[:, :, start:]
            Cal_list[core] = Calculations

        # [Multi-Core Calculation]
        pool = Pool(processes=opt.cpu_cores)

        single_signals = pool.starmap(time_evolution_function, zip(Cal_list))
        pool.close()
        pool.join()

        signal_tuple = tuple(single_signals)
        signal = np.concatenate(signal_tuple, axis=-1)
        return signal

    return multicore_wrapper


def powder_average(exp: object, opt: object, cal: object) -> None:
    """
    Build the powder average of a signal. Depending on the chosen grid, points
    are just summed up for all orientations (opt.grid='fibonacci',
    opt.grid='single') or the signal is interpolated (opt.grid='sophe').

    Parameters
    ----------
    exp : object
        Contains experimental parameters. This function uses exp.B_z for the
        interpolation.
    opt : object
        Contains simulation option parameters. This function uses opt.grid in
        order to determine if interpolation shall be done. In case of a sophe
        grid the attributes opt.number_of_peaks (number of peaks that are
        expected for each orientational spectrum) and opt.width_intp(the scaled
        gaussian line width for convolution before peak finding is done) are
        used.

    cal : object
        Contains the results calculated during the simulation. This function
        uses cal.signal. This array contains the intensities of a signal for
        all time points, magnetic field points and orientational points and
        can be built by the function make_signal.

    Attributes
    ----------
    cal.spec_sim : np.ndarray
        This matrix contains the intensities in abitrary units of a transient
        epr spectrum for all time points t in cal.t and all magnetic field
        points in exp.B_z. So the shape is len(t)xlen(B).

    Returns
    -------
    None

    """
    if opt.grid == 'sophe':
        weightened_signal = cal.signal*cal.weights
        signal_powder_average = np.sum(weightened_signal, (-1))
        cal.spec_sim += signal_powder_average

    elif opt.grid == 'fibonacci':
        signal_powder_average = np.sum(cal.signal, (-1))
        cal.spec_sim += signal_powder_average

    elif opt.grid == 'single':
        signal_multiple_orientations = np.sum(cal.signal, (-1))
        cal.spec_sim += signal_multiple_orientations

    return None


def maximal_peak_finder(signal: 'np.ndarray', number_of_peaks: int
                        ) -> 'np.ndarray':
    """
    Find a variable number of maximal absolute peaks in a spectrum.

    Parameters
    ----------
    signal : np.ndarray
        Signal in which to find the peaks. This is a 1-dimensional array, that
        contains the intensities of the spectrum.
    number_of_peaks : int
        Number of peaks to detect.

    Returns
    -------
    peak_idx : np.ndarray
        Indices in the signal array of the maximal absolute peaks.

    """
    all_peaks = find_peaks(abs(signal), height=0)
    height = all_peaks[1]['peak_heights']
    idx_in_all_peaks = np.argpartition(
        height, -number_of_peaks)[-number_of_peaks:]
    peak_idx = all_peaks[0][idx_in_all_peaks]

    return peak_idx


def get_peaks(signal: 'np.ndarray', number_of_peaks: int,
              convolution_width: float) -> ('np.ndarray', 'np.ndarray'):
    """
    Get the peak indices and peak intensities in a spectrum for each time and
    orientation point. Before the peaks are determined the spectrum is
    convolved with a gaussian function.

    Parameters
    ----------
    signal : np.ndarray
        Signal in which to find the peaks. This is a 3-dimensional array, that
        contains the intensities of the spectrum for each time/magnetic field/
        orientation point.
    number_of_peaks : int
        Number of peaks to detect for each subspectrum.
    convolution_width : float
        Gaussian line width for the convolution.

    Returns
    -------
    peak_indices : 'np.ndarray'
        Peak index for each time point and each orientation point. The shape
        of the array will be time_points x number_of_peaks x grid_points.
    peak_intensities : 'np.ndarray'
        Intensity of the convolved spectrum at the position of the indices.
        Has the same dimension as peak_indices.

    """

    # convolve signal with gaussian function
    for orientation in range(signal.shape[-1]):
        signal[:, :, orientation] = con.voigt_convolution(
            convolution_width, signal[:, :, orientation])

    # find indices and intensities of the maximal peaks
    # for each time and orientation
    peak_indices = np.zeros(
        (signal.shape[0], number_of_peaks, signal.shape[-1]), dtype=np.int)
    peak_intensities = np.zeros(
        (signal.shape[0], number_of_peaks, signal.shape[-1]),
        dtype=COMPLEX_TYPE)

    for time in range(1, signal.shape[0]):
        for orientation in range(0, signal.shape[-1]):
            spc = np.real(signal[time, :, orientation])
            peak_idx = maximal_peak_finder(spc, number_of_peaks)
            peak_idx = np.sort(np.array(peak_idx))
            peak_indices[time, :, orientation] = peak_idx
            peak_intensities[time, :, orientation] = signal[
                time,  peak_idx, orientation]

    return peak_indices, peak_intensities


def interpolate_signal(x_axis: 'np.ndarray', signal: 'np.ndarray',
                       phi: 'np.ndarray', theta: 'np.ndarray',
                       number_of_peaks: int, convolution_width: float
                       ) -> 'np.ndarray':
    """
    Do an interpolation on a spectrum. The spectrum is time dependent and the
    following steps are done for every time point: A given number of peaks is
    searched for every orientation. Triangles on the sophe grid are determined
    and the peaks are sorted into groups of three (for each triangle). Then
    for each peak-grup the spectrum is interpolated along the x_axis by
    calculating a triangle along the x_axis. All triangle-spectra are summed
    up to the interpolated spectrum.

    Parameters
    ----------
    x_axis : 'np.ndarray'
        Axis along which the interpolation is done. This is the magnetic field.
    signal : 'np.ndarray'
        Array with intensities of the spectrum. Its shape should be
        time-points x magnetic-field-points x orientation-points.
    phi : 'np.ndarray'
        Array with the phi angles of the sophe grid.
    theta : 'np.ndarray'
        Array with the phi angles of the sophe grid.
    number_of_peaks : int
        Number of peaks to detect for each subspectrum.
    convolution_width : float
        Gaussian line width for the convolution for peak-finding.

    Returns
    -------
    interpolated_signal : 'np.ndarray'
        Interpolated powder pattern of the signal. Its shape is
        time-points x magnetic-field-points.

    """
    # build triangles from the sophe-grid
    idx, area = gri.triangles(phi, theta)
    number_of_triangles = area.shape[0]

    # find peaks in all spectra (for each orientation and time)
    peak_indices, peak_intensities = get_peaks(
        signal, number_of_peaks, convolution_width)

    # get the peaks for each triangle
    peak_intensities_tri = peak_intensities[:, :, idx]
    peak_indices_tri = peak_indices[:, :, idx]

    # calculate maximum height for each triangle-spectrum
    h_max = peak_intensities_tri.sum(-1)/3
    area_tri = (h_max*area)

    # get ascending values on the x-axis for each peak of each triangle
    peak_indices_tri = np.sort(peak_indices_tri, axis=-1)
    x_1 = x_axis[peak_indices_tri[:, :, :, 0]]
    x_2 = x_axis[peak_indices_tri[:, :, :, 1]]
    x_3 = x_axis[peak_indices_tri[:, :, :, 2]]

    # interpolate a triangle-spectrum in each triangle on the sphere
    interpolated_signal = np.zeros(
        (signal.shape[0], len(x_axis)), dtype=COMPLEX_TYPE)
    for time in range(0, signal.shape[0]):
        for peak in range(0, number_of_peaks):
            for triangle in range(0, number_of_triangles):
                x_idx_1 = peak_indices_tri[time, peak, triangle, 0]
                x_idx_2 = peak_indices_tri[time, peak, triangle, 1]
                x_idx_3 = peak_indices_tri[time, peak, triangle, 2]

                # first part of interpolated triangle-spectrum
                interpolated_signal[time, x_idx_1: x_idx_2] +=\
                    area_tri[time, peak, triangle] * (
                        x_axis[x_idx_1:x_idx_2]-x_1[time, peak, triangle]) / (
                            (x_2[time, peak, triangle]-x_1[time, peak, triangle]) * (
                                x_3[time, peak, triangle]-x_1[time, peak, triangle]))

                # second part of interpolated triangle-spectrum
                interpolated_signal[time, x_idx_2:x_idx_3] +=\
                    area_tri[time, peak, triangle] * (
                        x_axis[x_idx_2:x_idx_3]-x_2[time, peak, triangle]) / (
                            (x_3[time, peak, triangle]-x_2[time, peak, triangle]) * (
                                x_3[time, peak, triangle]-x_1[time, peak, triangle]))

    return interpolated_signal


def signal_hilbert_decay(sys: object, cal: object) -> None:
    """
    Multiply each time-spectrum with an exponential term for simulating
    a decay of the system in hilbert space.

    Parameters
    ----------
    sys : object
        Contains parameters of the spin system. This function uses sys.decay
        (decay time in s).
    cal : object
        Contains results calculated during the simulation. This function uses
        cal.spec_sim (from the function make_signal) and cal.t (time-array).

    Attributes
    ----------
    cal.spec_sim : np.ndarray
        The time arrays are multiplied with an exponential decay. The
        shape is not changed.

    Returns
    -------
    None

    """
    for b_trace in range(0, cal.spec_sim.shape[1]):
        cal.spec_sim[:, b_trace] = np.exp(
            -cal.t/sys.decay)*cal.spec_sim[:, b_trace]
