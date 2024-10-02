import numpy as np
import psutil

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32

# Funktion, die Array splittet, abhängig von
# 1.) Der aktuell verfügbaren Arbeitsspeichermenge
# 2.) Der am aktuellen Bottleneck benötigten Arbeitsspeichermenge
# Flexibel um beliebige Funktion drappierbar?


def chunk_size(bottleneck, bp, gp):
    if bottleneck == "set_up_density_matrix":
        need_of_memory = nom_tdp_zf_set_up_density_matrix(bp, gp)
    available_memory = psutil.virtual_memory().available * 0.8
    chunksize = np.ceil(need_of_memory/available_memory)

    if bottleneck == None:
        chunksize = 1
    return chunksize


def define_bottleneck(sys):
    if sys.spin_system == 'tdp' and sys.precursor == 'triplet-zf':
        bottleneck = "set_up_density_matrix"
    else:
        bottleneck = None

    return bottleneck


def split_grid(cal, chunk_size):
    cal.theta_split = np.array_split(cal.theta, chunk_size)
    cal.phi_split = np.array_split(cal.phi, chunk_size)
    if hasattr(cal, "weights"):
        cal.weights_split = np.array_split(cal.weights, chunk_size)
    return


def nom_tdp_zf_set_up_density_matrix(bp, gp):
    complex_memory = 8
    multioperator = bp*gp*6*6*complex_memory
    hams = 6*multioperator
    eigvecs = 3*multioperator
    basistransformation = 5*multioperator
    nom = hams+eigvecs+basistransformation
    return nom


def chunk_size_for_gpu(bp, gp):
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
    available_memory = need_of_memory
    chunksize = np.ceil(need_of_memory/free)
    return chunksize
