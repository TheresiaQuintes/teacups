import numpy as np
import juliacall as jc
from juliacall import Main as jl

COMPLEX_TYPE = np.complex64


def propagation(sys: object, opt: object, cal: object) -> None:
    """
    This is a developer version of the function "propagation" from the
    module signals and processing using the programming language JULIA
    instead of python to calculate the matrix exponential
    in liouvlle space. Thre different options are provided, two are always
    commented out.

    Calculate a time propagation operator matrix for a given hamiltonian.
    It can be chosen in which space (hilbert or liouville) the operator will
    be set up. A relaxation operator is taken into account. This function
    could be used inside the module signals_and_processing instead of the
    propagation-function, that is defined there.

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
        propagation = vec @ propagation @ np.conj(
            np.transpose(vec, (0, 1, 3, 2)))

    elif opt.space == 'liouville':
        step = cal.t[1]-cal.t[0]

        cal.ham_superop *= -1j*step

        x = jc.convert(jl.Array[jl.ComplexF64, 4], cal.ham_superop)
        jl.include("propagation_julia.jl")
        # OPT 1
        propagation = jl.propagation_julia(x)
        # OPT 2
        # propagation = jl.propagation_julia_fast_expm(x)
        # OPT 3
        # r = jc.convert(jl.Array[jl.ComplexF64, 3], cal.rho)
        # cal.r = []
        # for time_point in cal.t:
        #     rho_prop = jl.expmv_from_expokit(time_point, x, r)
        #     cal.r.append(rho_prop)

    else:
        print('opt.space has to be either hilbert or liouville')

    cal.propagation = np.array(propagation)

    return
