import numpy as np
from copy import deepcopy
import teacups.multioperator_tools as mut
import teacups.creators as cr
import teacups.hamiltonians as ham

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
            # resort populations
            rho_0_tri = np.array(sys.population, dtype=FLOAT_TYPE)
            rho_0_tri = cr.create_tensor(rho_0_tri, cal.phi, cal.theta)

            # fill in ZF-populations for all orientations
            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.angle_matrix = rho_0_tri.multirot
            rho.angle_matrix_changed()

            # diagonalize hf-Hamiltonians
            eig_hf, vec_hf = np.linalg.eigh(cal.ham_tri_hf)

            # basistransformation to the high field functions
            rho.B_angle_matrix = (
                np.conj(np.transpose(vec_hf, (0, 1, 3, 2)))
                @ rho.B_angle_matrix
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
            cal_tri = deepcopy(cal)
            sys_tri = deepcopy(sys)
            opt_tri = deepcopy(opt)
            opt_tri.space = "hilbert"
            sys_tri.s = 1
            cr.set_up_spinoperator(sys_tri, cal_tri)
            sys_tri.spin_system = "trip"
            sys_tri.precursor = "zf"
            sys_tri.population = np.array(sys.population[2:], dtype=FLOAT_TYPE)
            set_up_density_matrix(sys_tri, exp, opt_tri, cal_tri)

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.B_angle_matrix = np.kron(
                np.diag(np.array(sys.population[:2], dtype=FLOAT_TYPE)
                        ), cal_tri.rho)

        else:
            raise AttributeError(
                'The spin_system attribute only accpets \
                                 "rp" or "tdp" as a value for a triplet\
                                  precursor.'
            )

    elif sys.precursor == "triplet-eigen":
        if sys.spin_system == "rp":
            cal_tri = deepcopy(cal)
            sys_tri = deepcopy(sys)
            opt_tri = deepcopy(opt)
            opt_tri.space = "hilbert"
            sys_tri.s = 1
            cr.set_up_spinoperator(sys_tri, cal_tri)
            sys_tri.spin_system = "trip"
            sys_tri.precursor = "eigen"
            cal_tri.ham_sys = ham.set_up_triplet_hamiltonian(exp, opt, cal_tri)
            set_up_density_matrix(sys_tri, exp, opt_tri, cal_tri)

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.B_angle_matrix[:, :, 0, 0] = cal_tri.rho[:, :, 0, 0]
            rho.B_angle_matrix[:, :, 2, 2] = cal_tri.rho[:, :, 1, 1]
            rho.B_angle_matrix[:, :, 3, 3] = cal_tri.rho[:, :, 2, 2]

        elif sys.spin_system == "tdp":
            cal_tri = deepcopy(cal)
            sys_tri = deepcopy(sys)
            opt_tri = deepcopy(opt)
            opt_tri.space = "hilbert"
            sys_tri.s = 1
            cr.set_up_spinoperator(sys_tri, cal_tri)
            sys_tri.spin_system = "trip"
            sys_tri.precursor = "eigen"
            cal_tri.ham_sys = ham.set_up_triplet_hamiltonian(exp, opt, cal_tri)
            sys_tri.population = np.array(sys.population[2:], dtype=FLOAT_TYPE)
            set_up_density_matrix(sys_tri, exp, opt_tri, cal_tri)

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.B_angle_matrix = np.kron(
                np.diag(np.array(sys.population[:2], dtype=FLOAT_TYPE)
                        ), cal_tri.rho)
        else:
            raise AttributeError(
                'The spin_system attribute only accpets \
                                 "rp" or "tdp" as a value for a triplet\
                                  precursor.'
            )

    elif sys.precursor == "coupled":
        if sys.spin_system == "tdp":
            s13 = np.sqrt(1 / 3)
            s23 = np.sqrt(2 / 3)
            trans = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, s23, 0, s13, 0, 0],
                    [0, 0, s13, 0, s23, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, -s13, 0, s23, 0, 0],
                    [0, 0, s23, 0, -s13, 0],
                ],
                dtype=FLOAT_TYPE,
            )

            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.matrix = np.diag(np.array(sys.population, dtype=FLOAT_TYPE))
            rho.matrix = np.transpose(trans) @ rho.matrix @ trans
            rho.matrix *= np.eye(6)
            rho.matrix_changed()

        else:
            raise AttributeError(
                'The spin_system attribute only accpets \
                                 "tdp" as a value for a coupled precursor.'
            )

    elif sys.precursor == "basis":
        if (
            sys.spin_system == "rp"
            or sys.spin_system == "trip"
            or sys.spin_system == "doub"
            or sys.spin_system == "tdp"
        ):
            rho = mut.Multioperator(cal.s, opt.grid_points, exp.B_z)
            rho.matrix = np.diag(np.array(sys.population, dtype=FLOAT_TYPE))
            rho.matrix_changed()

        else:
            raise AttributeError(
                'The spin_system attribute only accepts \
                                  the following values: "rp", "trip", "doub" \
                                  or "tdp".'
            )

    else:
        raise AttributeError(
            'The precursor attribute of the Spinsystem-class \
                             only accepts the following values: "zf", "eigen",\
                            "singlet", "triplet-zf", "triplet-eigen", \
                            "coupled" or basis".\
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

