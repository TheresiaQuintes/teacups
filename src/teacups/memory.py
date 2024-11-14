import numpy as np
import psutil

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


def chunk_size(bottleneck: str, bp: int, gp: int) -> int:
    """
    Calculate the number of pieces in which to split the arrays of the
    simulation to avoid memory overflow. This is dependent on the chosen
    bottleneck, the size of the arrays and the available memory.

    Parameters
    ----------
    bottleneck : str
        String defining which part of the simulation causes the memory
        bottleneck. If bottleneck is set to None the chunksize is set to 1
        as no bottleneck is defined. Otherwise the need of memory is calculated
        in different ways. Dependent on the available memory the chunksize
        is returned
    bp : int
        Number of magnetic field points. Important for the determination of
        the need of memory.
    gp : int
        Number of grid points. Important for the determination of the need
        of memory.

    Returns
    -------
    chunksize : int
        Number of pieces in which to split the arrays to avoid a memory
        overflow during the simulation

    """
    if bottleneck is None:
        chunksize = 1
    else:
        if bottleneck == "set_up_density_matrix":
            need_of_memory = nom_tdp_zf_set_up_density_matrix(bp, gp)

        available_memory = psutil.virtual_memory().available * 0.8
        chunksize = np.ceil(need_of_memory/available_memory)

    if chunksize > 1:
        chunksize += 1
    return chunksize


def define_bottleneck(sys: object) -> str:
    """
    Define the simulations bottleneck dependent on the chosen spin system.
    The proper string for the function chunk_size() is returned.

    Parameters
    ----------
    sys : object
        Container with the definition of the spin system. This is used to
        find out, which the simulations bottleneck is. The attributes
        spin_system and precursor are used.

    Returns
    -------
    bottleneck : str
        String defining the bottleneck of the function.

    """
    if sys.spin_system == 'tdp' and sys.precursor == 'triplet-zf':
        bottleneck = "set_up_density_matrix"
    else:
        bottleneck = None

    return bottleneck


def nom_tdp_zf_set_up_density_matrix(bp: int, gp: int) -> float:
    """
    Calculate the need of memory for the function set_up_density_matrix from
    the module densit_matrices for the case of a "zf-triplet" precursor and
    "tdp" spin system.

    Parameters
    ----------
    bp : int
        Number of magnetic field points.
    gp : int
        Number of angle points.

    Returns
    -------
    float
        Need of memory in bytes.

    """
    complex_memory = 8
    multioperator = bp*gp*6*6*complex_memory
    hams = 6*multioperator
    eigvecs = 3*multioperator
    basistransformation = 5*multioperator
    nom = hams+eigvecs+basistransformation
    return nom


def chunk_size_for_gpu(bp: int, gp: int) -> int:
    """
    Calculate the cunk size for the calculations using the GPU in the function
    densit_matrices.set_up_density_matrix() for the case of a "zf-triplet"
    precursor and "tdp" spin system.

    Parameters
    ----------
    bp : int
        Number of magnetic field points.
    gp : int
        Number of angle points.

    Returns
    -------
    chunksize : int
        Number of pieces in which to split the arrays to avoid a memory
        overflow on the GPU during setup of the density matrix.

    """
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
    )
    need_of_memory = nom_tdp_zf_set_up_density_matrix(bp, gp)

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    free = info.free
    chunksize = np.ceil(need_of_memory/free)
    return chunksize
