import numpy as np
from copy import deepcopy
import teacups.multioperator_tools as mut
import teacups.matrix_tools as mt
import teacups.creators as cr
import teacups.hamiltonians as ham
import teacups.memory as mem

import scipy.constants as const
MU_B = const.physical_constants["Bohr magneton in Hz/T"][0]
K_B = const.physical_constants["Boltzmann constant in Hz/K"][0]
COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32


def set_up_density_matrix(sys: object, exp: object, opt: object, cal: object
                          ) -> None:
    """
    Calculate the density matrix in the basis and shape of the systems
    Hamiltonian.

    Parameters
    ----------
    sys : object
        Sys contains all spin system attributes. This function uses the
        following attributes:
        sys.precursor: The density matrix, for which the populations are given,
        is set up in the precursors basis. Depending on the spin system the
        precursor can be set to the following strings: 'zf' (population of
        zero-field levels), 'eigen' (population of eigenvalues of the system),
        'singlet' (a radical pair in pure singlet state), 'triplet-zf' (a
        triplet precursor populated in the triplets zero-field basis of a
        radical pair or a triplet doublet pair), 'triplet-eigen' (a
        triplet precursor populated in the triplets eigen basis of a
        radical pair or a triplet doublet pair), 'coupled' (a triplet doublet
        pair in the coupled basis), 'basis' (basis of the population input
        is not changed)
        sys.spin_system: The spin system attribute has to be set to a string
        defining the spin system. This may be 'rp' (radical pair), 'trip'
        (triplet), 'doub' (doublet), or 'tdp' (triplet doublet pair).
        sys.population: Diagonal elements of the density matrix/ population
        of the states in the basis given by the precursor attribute. The
        length of the population vector equals the dimension of the spin
        system.
    exp : object
        Contains the experimental parameters. This function needs the attribute
        exp.B_z (static magnetic field vector).
    opt : object
        Contains the simulation option parameters. This function needs the
        attributes opt.space (defining the space for the calculations as
        hilbert or liouville) and opt.grid_points (number of angle
        combinations).
    cal : object
        Contains calculated results during the simulation. This function
        uses the attribute cal.s (the spin operator) and dependent on the
        choice of precursor and spin system optionally different hamiltonians.
        For a zf-triplet-precursor cal.phi and cal.theta are used as well.

    Raises
    ------
    AttributeError
        In case a wrong combination of spin system and basis is given.

    Returns
    -------
    None

    Attributes
    ----------
    cal.rho : np.ndarray
        The density matrix is added as an attribute to the cal-object. Its
        shape is B_angle_matrix like in case of hilbert space calcultations
        ans B_angle_vector like for liouvlle space calculations (more details
        in the multioperator module). The density matrix is returned in the
        basis of the system Hamiltonian.

    """
    if sys.precursor == "zf":
        if sys.spin_system == "trip":

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)

            # diagonalize hf-Hamiltonians
            eig_hf, vec_hf = np.linalg.eigh(cal.ham_tri_hf)

            # basistransformation to the high field functions
            rho.B_angle_matrix = (
                np.conj(np.transpose((vec_hf), (0, 1, 3, 2)))
                @ np.diag(np.array(sys.population, dtype=FLOAT_TYPE))
                @ vec_hf
            )

            # kill all off-diagonal elements
            rho.B_angle_matrix *= np.eye(3, dtype=FLOAT_TYPE)

        else:
            raise AttributeError(
                'The spin_system attribute has to be "trip" \
                                 for a "zf"-precursor.'
            )

    elif sys.precursor == "eigen":
        if (
            sys.spin_system == "rp"
            or sys.spin_system == "trip"
            or sys.spin_system == "doub"
            or sys.spin_system == "tdp"
        ):
            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.matrix = np.diag(np.array(sys.population, dtype=FLOAT_TYPE))
            rho.matrix_changed()

            eig, vec = np.linalg.eigh(cal.ham_sys)
            rho.B_angle_matrix = (
                vec @ rho.B_angle_matrix @ np.conj(
                    np.transpose(vec, (0, 1, 3, 2)))
            )

        else:
            raise AttributeError(
                'The spin_system attribute only accepts \
                                  the following values: "rp", "trip", "doub" \
                                  or "tdp".'
            )

    elif sys.precursor == "singlet":
        if sys.spin_system == "rp":
            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.matrix[1, 1] = 1
            rho.matrix_changed()
        else:
            raise AttributeError(
                'The spin_system attribute only accpets \
                                 "rp" as a value for a singlet precursor.'
            )

    elif sys.precursor == "triplet-zf":
        if sys.spin_system == "rp":
            cal_tri = deepcopy(cal)
            sys_tri = deepcopy(sys)
            opt_tri = deepcopy(opt)
            opt_tri.space = "hilbert"
            sys_tri.s = 1
            cr.set_up_spinoperator(sys_tri, cal_tri)
            sys_tri.spin_system = "trip"
            sys_tri.precursor = "zf"
            set_up_density_matrix(sys_tri, exp, opt_tri, cal_tri)

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.B_angle_matrix[:, :, 0, 0] = cal_tri.rho[:, :, 0, 0]
            rho.B_angle_matrix[:, :, 2, 2] = cal_tri.rho[:, :, 1, 1]
            rho.B_angle_matrix[:, :, 3, 3] = cal_tri.rho[:, :, 2, 2]

        elif sys.spin_system == "tdp":
            # set up the full high field hamiltonian of the coupled system
            # in the product basis: +1a, 0a, -1a, +1b, 0b, -1b
            # ham_zfs is only the triplet-ZFS part of the coupled hamiltonian
            ham_hf, ham_zfs = ham.set_up_tdp_full_high_field_hamiltonian(
                sys, exp, opt, cal)
            # !!! Kontrolle hat ergeben, dass sich die Eigenwerte von ham_hf
            # und ham_sys nur um einen Offset von J/4 unterscheiden. Das liegt
            # an der unterschiedlichen Definition von H_ex (siehe Doku) von
            # RP (verwendet in multioperator_tools) und TDP. Fürs Spektrum
            # gibt es keinen Unterschied, allerdings könnte man das beim
            # Aufräumen vom Code angleichen (z.B. indem H_ex für alle Systeme
            # identisch aufgestellt wird)

            # define rho in xyz-basis using populations for triplet-zf levels
            rho_trip = np.diag(np.array(sys.population[2:], dtype=FLOAT_TYPE))
            rho_trip = np.kron(rho_trip, np.eye(2, dtype=FLOAT_TYPE))

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.matrix = rho_trip

            CUPY = opt.CUPY
            if CUPY:
                import cupy as cp

                chunk_size = mem.chunk_size_for_gpu(
                    len(exp.B_z), opt.grid_points)*4
                print(chunk_size)

                ham_hf_split = np.array_split(ham_hf, chunk_size)
                ham_zfs_split = np.array_split(ham_zfs, chunk_size)
                ham_sys_split = np.array_split(cal.ham_sys, chunk_size)

                rho_split = []

                from pynvml import (
                    nvmlInit,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                )

                for chunk in range(len(ham_hf_split)):
                    ham_zfs_gpu = cp.asarray(ham_zfs_split[chunk])
                    eig_zfs, vec_zfs = cp.linalg.eigh(ham_zfs_gpu)
                    del eig_zfs, ham_zfs_gpu

                    ham_hf = cp.asarray(ham_hf_split[chunk])
                    ham_xyz = cp.conj(cp.transpose(
                        vec_zfs, (0, 1, 3, 2))) @ ham_hf @ vec_zfs
                    del vec_zfs, ham_hf

                    eig_hf, vec_hf = cp.linalg.eigh(ham_xyz)
                    del eig_hf, ham_xyz

                    cal_ham_sys = cp.asarray(ham_sys_split[chunk])
                    eig_sys, vec_sys = cp.linalg.eigh(cal_ham_sys)
                    del eig_sys, cal_ham_sys

                    rho_matrix = cp.asarray(rho.matrix)
                    rho_B_angle_matrix = (
                        cp.conj(cp.transpose((vec_hf), (0, 1, 3, 2)))
                        @ rho_matrix
                        @ vec_hf
                    )
                    del rho_matrix, vec_hf
                    rho_B_angle_matrix *= cp.eye(6, dtype=FLOAT_TYPE)

                    rho_B_angle_matrix = (
                        vec_sys
                        @ rho_B_angle_matrix
                        @ cp.conj(cp.transpose((vec_sys), (0, 1, 3, 2)))
                    )
                    del vec_sys
                    rho_B_angle_matrix *= cp.eye(6, dtype=FLOAT_TYPE)

                    rho_split.append(rho_B_angle_matrix.get())
                    del rho_B_angle_matrix
                rho.B_angle_matrix = np.concatenate(rho_split)

            else:
                # the eigenbasis of ham_zfs is xx yy zz (= xyz-basis)
                eig_zfs, vec_zfs = np.linalg.eigh(ham_zfs)

                # transform high field hamiltonian to xyz-basis
                ham_xyz = np.conj(np.transpose(
                    vec_zfs, (0, 1, 3, 2))) @ ham_hf @ vec_zfs

                # diagonalise high field hamiltonian to get the eigenvectors for
                # transformation xyz-basis <-> TDP-eigenbasis
                eig_hf, vec_hf = np.linalg.eigh(ham_xyz)

                # diagonalise the spin system hamiltonian to get the eigenvectors
                # for the transformation product basis <-> TDP-eigenbasis
                eig_sys, vec_sys = np.linalg.eigh(cal.ham_sys)

                # basistransformation rho: xyz-basis -> TDP-eigenbasis
                rho.B_angle_matrix = (
                    np.conj(np.transpose((vec_hf), (0, 1, 3, 2)))
                    @ rho.matrix
                    @ vec_hf
                )
                rho.B_angle_matrix *= np.eye(6, dtype=FLOAT_TYPE)

                # basistransformation rho: TDP-eigenbasis -> product basis
                rho.B_angle_matrix = (
                    vec_sys
                    @ rho.B_angle_matrix
                    @ np.conj(np.transpose((vec_sys), (0, 1, 3, 2)))
                )

                rho.B_angle_matrix *= np.eye(6, dtype=FLOAT_TYPE)

            # add density matrix for the doublet in product basis
            rho_doub = np.diag(np.array(sys.population[:2], dtype=FLOAT_TYPE))
            rho_doub = np.kron(rho_doub, np.eye(3, dtype=FLOAT_TYPE))
            rho.B_angle_matrix += rho_doub
        else:
            raise AttributeError(
                'The spin_system attribute only accpets \
                                 "rp" or "tdp" as a value for a triplet\
                                  precursor.'
            )

    else:
        raise AttributeError(
            'The precursor attribute of the Spinsystem-class \
                             only accepts the following values: "zf", "eigen",\
                            "singlet" or "triplet-zf",\
                            For more details see the documentation.'
        )

    if opt.space == "liouville":
        rho.build_vector()
        cal.rho = rho.B_angle_vector
    elif opt.space == "hilbert":
        cal.rho = rho.B_angle_matrix
    else:
        raise AttributeError("opt.space has to be either hilbert or liouville")

    return
