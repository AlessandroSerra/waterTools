#!/usr/bin/env python3

# NOTE: classical velocities are set to zero in this code

import logging
import re
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.linalg import eigh_tridiagonal

# Configure logger
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
#                   --- Constants and Parameters ---
# --------------------------------------------------------------
# "C_U": 0.01036427,  # cm^-1 / fs (speed of light in cm/fs) -- not used directly

# NOTE: using 1amu = 1.6605390666e-27 kg
CONSTANTS = {
    "HBAR_amu_A2_fs": 0.00635078,  # amu * A^2 / fs (hbar in amu*A^2/fs)
    "AMU_A2_fs2_to_eV": 103.642,
    "K_B_eV": 8.617333262e-5,  # eV/K (Boltzmann constant)
    "EV_TO_CM1": 8065.54429,  # eV to cm^-1 conversion factor
    "MASSES": {"O": 15.999, "H": 1.008},
}
CONSTANTS["MU_OH"] = (CONSTANTS["MASSES"]["O"] * CONSTANTS["MASSES"]["H"]) / (
    CONSTANTS["MASSES"]["O"] + CONSTANTS["MASSES"]["H"]
)

# --- Lippincott-Schroeder Potential Parameters from Koji Ando (DOI: 10.1063/1.2210477) ---
LS_PARAMS = {
    "r0": 0.97,  # Å
    "n": 9.8,  # Å^-1
    "g": 1.45,
    "r0_star_factor": 0.95,
    "R_OO": 2.85,  # Å
    "D_OH": 4.82,  # eV
}
LS_PARAMS["n_star"] = LS_PARAMS["g"] * LS_PARAMS["n"]
LS_PARAMS["r0_star"] = LS_PARAMS["r0"] * LS_PARAMS["r0_star_factor"]
LS_PARAMS["D_star"] = LS_PARAMS["D_OH"] / LS_PARAMS["g"]  # eV

# --- Grid Parameters for Numerical Calculations ---
GRID_PARAMS = {
    "r_min": 0.6,  # Å - minimum r for grid
    "r_max": 2.3,  # Å - maximum r for grid
    "Nr": 301,  # Number of points for r grid (use odd for Simpson's rule)
    "Np": 301,  # Number of points for p grid
    "p_range_factor": 3,  # Factor to multiply p_max for p grid range
}

N_EIGENSTATES = 3  # Number of eigenstates to compute


# --- Atom Object ---
@dataclass
class Atom:
    index: int
    atom_type: int
    atom_string: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class Simulation:
    n_atoms: int
    lattice_string: str
    cell_vectors: NDArray[np.float64]
    properties_string: str


# custom types fot Atom and Molecule
AtomType = TypeVar("AtomType", bound="Atom")
SimulationType = TypeVar("SimulationType", bound="Simulation")
MoleculeType = List[AtomType]


# --------------------------------------------------------------
#       --- Function to read LAMMPS dump file ---
# --------------------------------------------------------------
def readGPUMDdump(
    filename: str, atom_per_molecule: int, keep_vels: bool
) -> Tuple[List[List[Atom]], Simulation]:
    # read GPUMD xyz dump file
    with open(f"{filename}", "r") as f:
        lines = f.readlines()

        n_atoms = int(lines[0])
        comment_line = lines[1]

        lattice_string_match = re.search(r'Lattice="([^"]*)"', comment_line)
        properties_string_match = re.search(r'Properties=([^"]*)', comment_line)

        if lattice_string_match is None or properties_string_match is None:
            raise ValueError(
                "Invalid file format: missing lattice or properties string."
            )

        lattice_string = lattice_string_match.group(1)
        properties_string = properties_string_match.group(1)
        lattice_values = np.array([float(x) for x in lattice_string.split()])
        cell_vectors = lattice_values.reshape((3, 3))
        current_molecule = []

        simulation = Simulation(
            n_atoms=n_atoms,
            lattice_string=lattice_string,
            cell_vectors=cell_vectors,
            properties_string=properties_string,
        )

        molecules = []

        for i, line in enumerate(lines[2:]):
            line_split = line.split()
            atom_string = line_split[0]
            atom_type = 1 if atom_string == "O" else 2
            atom_position = np.array([float(x) for x in line_split[1:4]])
            atom_mass = float(line_split[4])
            atom_velocity = (
                np.array([float(x) for x in line_split[5:8]])
                if keep_vels
                else np.zeros(3)
            )

            atom = Atom(
                index=(i + 1),
                atom_type=atom_type,
                atom_string=atom_string,
                mass=atom_mass,
                position=atom_position,
                velocity=atom_velocity,
            )
            current_molecule.append(atom)
            if len(current_molecule) == atom_per_molecule:
                molecules.append(current_molecule)
                current_molecule = []

        # --- Consistency checks ---
        # 1. Assicurati che non ci siano atomi "orfani" non assegnati a una molecola
        if len(current_molecule) != 0:
            raise ValueError(
                f"File {filename}: number of atoms ({n_atoms}) "
                f"is not an exact multiple of atom_per_molecule ({atom_per_molecule})."
            )

        # 2. Verifica che il numero di atomi totali coincida con la somma per molecole
        total_atoms_parsed = len(molecules) * atom_per_molecule
        if total_atoms_parsed != n_atoms:
            raise ValueError(
                f"Mismatch in atom count: header reports {n_atoms}, "
                f"but parser assembled {total_atoms_parsed} atoms "
                f"({len(molecules)} molecules × {atom_per_molecule} atoms/molecule)."
            )

        return molecules, simulation


# --------------------------------------------------------------
#          --- Lippincott-Schroeder Potential ---
# --------------------------------------------------------------


# --- OH term (Using eV for energy) ---
def V_LS_bond1(r: float) -> float:
    """Potential for bond I (OH stretch) in eV"""
    r0 = LS_PARAMS["r0"]
    n = LS_PARAMS["n"]
    d_OH = LS_PARAMS["D_OH"]
    if r <= 1e-6:
        return np.inf  # Avoid division by zero or negative r
    delta_r = r - r0
    exponent = -n * delta_r**2 / (2 * r)
    return d_OH * (1 - np.exp(exponent))


# --- O--H term (Using eV for energy) ---
def V_LS_bond2(r: float) -> float:
    """Potential for bond II (H---O interaction) in eV"""
    R = LS_PARAMS["R_OO"]
    r0_star = LS_PARAMS["r0_star"]
    n_star = LS_PARAMS["n_star"]
    D_star = LS_PARAMS["D_star"]
    r_ho = R - r
    if r_ho <= 1e-6:
        return np.inf  # H cannot be beyond the second O
    delta_r_star = r_ho - r0_star  # How much H---O is stretched/compressed
    exponent = -n_star * delta_r_star**2 / (2 * r_ho)
    # Using the approximation V2 = -D* exp(...)
    return -D_star * np.exp(exponent)


# --- Total Potential for H motion at fixed R ---
def V_LS_hydrogen_motion(r: float) -> float:
    """Total potential for H motion at fixed R in eV"""
    v1 = V_LS_bond1(r)
    v2 = V_LS_bond2(r)
    return v1 + v2


# ----------------------------------------------------------------
# --- Numerical Schrödinger Solver (Finite Difference Method) ---
# ----------------------------------------------------------------


def solve_schrodinger_1d(
    potential_func: Callable[[float], float], r_grid: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Numerically solves the 1D TISE using finite difference method.

    Args:
        potential_func: Function V(r) defining the potential in eV.
        r_grid: 1D array of r values (Angstrom) for discretization.
        num_states: Number of lowest energy states to return.

    Returns:
        eigenvalues: Array of energy levels (E_n) in eV.
        eigenvectors: Array where columns are the numerical wavefunctions psi_n(r).
        Normalized such that integral |psi_n|^2 dr = 1.
    """

    logger.info("\tSolving Schrödinger equation numerically...")

    dr = r_grid[1] - r_grid[0]
    N = len(r_grid)

    # Potential energy on the grid
    V_grid = np.array([potential_func(r) for r in r_grid])  # eV
    V_min = np.min(V_grid)
    V_grid -= V_min  # Shift minimum to zero (doesn't affect wavefunctions)

    # Kinetic energy term T = -hbar^2 / (2*mu) * d^2/dr^2
    # Factor hbar^2 / (2*mu) in eV * A^2
    hbar_sq_over_2mu_native = CONSTANTS["HBAR_amu_A2_fs"] ** 2 / (
        2 * CONSTANTS["MU_OH"]
    )  # amu*A^2/fs^2
    hbar_sq_over_2mu_eV_A2 = hbar_sq_over_2mu_native * CONSTANTS["AMU_A2_fs2_to_eV"]

    # Finite difference matrix for -d^2/dr^2 (using centered difference)
    # Off-diagonal elements (-1) and Diagonal elements (2)
    diag = np.ones(N) * 2.0
    offdiag = np.ones(N - 1) * -1.0

    # Construct Hamiltonian matrix H = T + V
    # T matrix elements = (hbar^2 / (2*mu*dr^2)) * [2, -1, ...]
    # V matrix elements = V_grid on the diagonal
    H_diag = (hbar_sq_over_2mu_eV_A2 / dr**2) * diag + V_grid
    H_offdiag = (hbar_sq_over_2mu_eV_A2 / dr**2) * offdiag

    # Solve the eigenvalue problem H * psi = E * psi
    # Use eigh_tridiagonal for efficiency and numerical stability
    eigenvalues_raw, eigenvectors_raw = eigh_tridiagonal(
        H_diag, H_offdiag, select="i", select_range=(0, N_EIGENSTATES - 1)
    )

    # Add back the potential minimum shift and ensure correct normalization
    eigenvalues = eigenvalues_raw + V_min  # Energies in eV
    eigenvectors = np.zeros_like(eigenvectors_raw)

    # Normalize eigenvectors: integral |psi|^2 dr = 1
    for i in range(N_EIGENSTATES):
        psi = eigenvectors_raw[:, i]
        norm_sq = simpson(y=np.abs(psi) ** 2, x=r_grid)
        if norm_sq > 1e-9:
            eigenvectors[:, i] = psi / np.sqrt(norm_sq)
        else:
            eigenvectors[:, i] = psi  # Avoid division by zero if norm is tiny

    logger.info("\tSchrödinger equation solved.")
    return eigenvalues, eigenvectors


# ------------------------------------------------------------
# --- Wigner Function Calculation (Numerical Integration) ---
# -----------------------------------------------------------


def calculate_wigner_function(
    psi_n: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
    num_y_points: int = 201,
) -> NDArray[np.float64]:
    """
    Calculates the Wigner function W_n(r, p) on a grid.

    Args:
        psi_n: 1D array of the numerical wavefunction psi_n(r).
        r_grid: 1D array of r values (Angstrom) where psi_n is defined.
        p_grid: 1D array of p values (amu * A / fs) for the output grid.
        num_y_points: Number of points for numerical integration over y.

    Returns:
        wigner_grid: 2D array W_n(r, p) evaluated at r_grid[i], p_grid[j].
                     Units: 1 / (eV * fs) if p is in amu*A/fs and r in A.
    """
    Nr = len(r_grid)
    Np = len(p_grid)
    wigner_grid = np.zeros((Nr, Np))

    # Interpolate wavefunction for evaluation at r+y and r-y
    # Use linear interpolation, ensure bounds_error=False, fill_value=0
    psi_interp = interp1d(
        r_grid, psi_n, kind="linear", bounds_error=False, fill_value=0.0
    )

    # Determine integration range for y
    # Needs to cover the extent where psi_n is non-zero
    r_min_psi, r_max_psi = r_grid[0], r_grid[-1]
    # Heuristic: integrate y over the range of r_grid should be sufficient
    y_max = (r_max_psi - r_min_psi) / 2.0
    y_grid = np.linspace(-y_max, y_max, num_y_points)

    # Loop over r and p, calculate integral numerically
    for i, r_val in enumerate(r_grid):
        # Precompute psi values needed for this r_val
        psi_r_plus_y = psi_interp(r_val + y_grid)
        psi_r_minus_y = psi_interp(r_val - y_grid)  # psi is real, so psi* = psi

        for j, p_val in enumerate(p_grid):
            # Integrand: conj(psi(r+y)) * psi(r-y) * exp(2*i*p*y/hbar)
            # Since psi is real: psi(r+y) * psi(r-y) * exp(2*i*p*y/hbar)
            exponent_term = (2.0j * p_val / CONSTANTS["HBAR_amu_A2_fs"]) * y_grid
            integrand = psi_r_plus_y * psi_r_minus_y * np.exp(exponent_term)

            # Integrate using Simpson's rule (real part, as Wigner is real)
            # Factor 1 / (pi * hbar)
            integral_val = simpson(y=integrand, x=y_grid)
            # Wigner function is real, take real part (imaginary part should be ~0 due to symmetry)
            wigner_grid[i, j] = integral_val.real / (
                np.pi * CONSTANTS["HBAR_amu_A2_fs"]
            )

    return wigner_grid


# ----------------------------------------------------
#   --- Wigner Sampling (Acceptance-Rejection) ---
# ----------------------------------------------------


def sample_from_wigner(
    wigner_grid: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
) -> list[tuple[float, float]]:
    """
    Samples (r, p) pairs from the Wigner distribution using acceptance-rejection.
    Handles positive and negative values in Wigner function.

    Args:
        wigner_grid: 2D array W_n(r, p).
        r_grid: 1D r coordinates (Angstrom) for the grid.
        p_grid: 1D p coordinates (amu * A / fs) for the grid.

    Returns:
        samples: List of (r, p) tuples.
    """
    logger.debug("\tSampling points from Wigner distribution...")
    samples = []
    Nr = len(r_grid)
    Np = len(p_grid)

    # Use absolute value for proposal distribution envelope
    wigner_abs = np.abs(wigner_grid)
    w_max = np.max(wigner_abs)

    if w_max < 1e-15:
        logger.warning("\tWigner function seems to be zero everywhere. Cannot sample.")
        # Return samples at the center of the grid or raise error
        r_center = r_grid[Nr // 2]
        p_center = p_grid[Np // 2]
        return [(r_center, p_center)]

    # Create interpolator for efficient lookup of W(r,p)
    # Use RectBivariateSpline for 2D interpolation
    wigner_interp = RectBivariateSpline(r_grid, p_grid, wigner_grid)

    max_attempts = 1000  # Prevent infinite loop if sampling is hard
    attempts = 0

    while len(samples) == 0 and attempts < max_attempts:
        attempts += 1
        # Sample uniformly within the grid range
        r_try = np.random.uniform(r_grid[0], r_grid[-1])
        p_try = np.random.uniform(p_grid[0], p_grid[-1])

        # Get Wigner value at the sampled point using interpolation
        w_val = wigner_interp(r_try, p_try, grid=False)  # Get value at single point

        # Acceptance-Rejection step
        # Compare |W(r,p)| with a random number scaled by max(|W|)
        acceptance_prob = np.abs(w_val) / w_max
        if np.random.rand() < acceptance_prob:
            # Accept the sample (r_try, p_try)
            samples.append((r_try, p_try))
            break  # Found one sample, exit loop

    if len(samples) == 0:
        logger.warning(
            f"\tReached max attempts ({max_attempts}) without finding a valid sample."
        )

    logger.debug(f"\tSampling complete. Generated {len(samples)} samples.")

    return samples


# Sampling 1 points from Wigner distribution...
# Sampling complete. Generated 1 samples.

# -------------------------------------------------------
# --- Coordinate Remapping (Unchanged from previous) ---
# -------------------------------------------------------


def remap_coords_vels(
    pos_O: NDArray[np.float64],
    pos_H: NDArray[np.float64],
    vel_O: NDArray[np.float64],
    vel_H: NDArray[np.float64],
    r_new: float,
    p_new: float,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Remaps O, H positions and velocities for a target r_new, p_new."""

    # Current state
    vec_OH = pos_H - pos_O  # A
    r_old = np.linalg.norm(vec_OH)  # A
    if r_old < 1e-9:  # Avoid division by zero if atoms overlap
        raise ZeroDivisionError("Warning: O and H atoms are too close for sampling!\n")
    else:
        unit_vec_OH = vec_OH / r_old

    # Calculate changes needed
    delta_r = r_new - r_old  # A

    # Update positions (preserve center of mass)
    pos_O_new = (
        pos_O - (CONSTANTS["MU_OH"] / CONSTANTS["MASSES"]["O"]) * delta_r * unit_vec_OH
    )  # A
    pos_H_new = (
        pos_H + (CONSTANTS["MU_OH"] / CONSTANTS["MASSES"]["H"]) * delta_r * unit_vec_OH
    )  # A

    # Update velocities (preserve center of mass velocity)
    # p_new is relative momentum along the bond in amu * A / fs
    # Velocity change dv = p_rel / m
    # scaled_p = p_new / 12
    # p_new = scaled_p
    vel_O_new = vel_O - (p_new / CONSTANTS["MASSES"]["O"]) * unit_vec_OH  # A / fs
    vel_H_new = vel_H + (p_new / CONSTANTS["MASSES"]["H"]) * unit_vec_OH  # A / fs

    return pos_O_new, pos_H_new, vel_O_new, vel_H_new


# -------------------------------------------------------
#       --- Molecules Excitation Function ---
# -------------------------------------------------------
def excite_molecules(
    molecules: List[List[Atom]], context: dict
) -> Tuple[List[List[Atom]], NDArray[np.int64]]:
    # unpack context
    excite_perc = context["excite_perc"]
    excite_lvl = context["excite_lvl"]
    do_plot = context["plot"]

    num_molecules = len(molecules)
    num_to_excite = int(num_molecules * excite_perc)
    exc_mol_idxs = np.random.choice(num_molecules, num_to_excite, replace=False)
    molecules_to_excite = [molecules[idx] for idx in exc_mol_idxs]

    # Grid parameters from global configuration
    r_min = GRID_PARAMS["r_min"]
    r_max = GRID_PARAMS["r_max"]
    Nr = GRID_PARAMS["Nr"]
    r_grid = np.linspace(r_min, r_max, Nr)

    energies, wavefunctions = solve_schrodinger_1d(V_LS_hydrogen_motion, r_grid)

    # NOTE: Compute p-grid using the fundamental energy gap
    E_fin = energies[1]  # eV
    E_init = energies[0]  # eV
    delta_E_eV = np.abs(E_fin - E_init)  # eV
    delta_E_native = (
        delta_E_eV / CONSTANTS["AMU_A2_fs2_to_eV"]
    )  # Convert back to amu*A^2/fs^2
    p_max = np.sqrt(2 * CONSTANTS["MU_OH"] * delta_E_native)  # amu * A / fs
    Np = GRID_PARAMS["Np"]
    p_range_factor = GRID_PARAMS["p_range_factor"]
    p_grid = np.linspace(-p_max * p_range_factor, p_max * p_range_factor, Np)

    # Extract wavefunctions for all states (always calculate 3 states)
    psi_0 = wavefunctions[:, 0]
    psi_1 = wavefunctions[:, 1]
    psi_2 = wavefunctions[:, 2]

    # Log all energy levels (always 3 states available)
    logger.info(
        f"\tEnergy levels: E0={energies[0]:.3f} eV "
        f"({(energies[0] * CONSTANTS['EV_TO_CM1']):.2f} cm⁻¹), "
        f"E1={energies[1]:.3f} eV "
        f"({(energies[1] * CONSTANTS['EV_TO_CM1']):.2f} cm⁻¹), "
        f"E2={energies[2]:.3f} eV "
        f"({(energies[2] * CONSTANTS['EV_TO_CM1']):.2f} cm⁻¹)"
    )
    logger.info(
        f"\tEnergy gap E{excite_lvl}-E{excite_lvl - 1} = {(delta_E_eV):.3f} eV "
        f"({(delta_E_eV * CONSTANTS['EV_TO_CM1']):.2f} cm⁻¹)\n"
        f"\tCorresponding reduced momentum: {p_max:.4f} amu·Å/fs"
    )

    logger.info(f"\tCalculating Wigner function for n={excite_lvl} level...")

    # Calculate Wigner function for the target excitation level
    if excite_lvl == 0:
        wigner_target = calculate_wigner_function(psi_0, r_grid, p_grid)
    elif excite_lvl == 1:
        wigner_target = calculate_wigner_function(psi_1, r_grid, p_grid)
    elif excite_lvl == 2:
        wigner_target = calculate_wigner_function(psi_2, r_grid, p_grid)
    else:
        raise NotImplementedError(f"Excitation level {excite_lvl} is not implemented!")

    logger.info("\tWigner function calculated.")

    modified_indices = set()  # Keep track of modified atoms
    all_samples = []

    for exc_mol_idx, exc_molecule in enumerate(molecules_to_excite):
        Oatom = exc_molecule[0]
        Hatom = np.random.choice(exc_molecule[1:3])  # Randomly select one H atom
        Oatom_idx = Oatom.index
        Hatom_idx = Hatom.index

        # Check if atoms were already modified (e.g., if exciting both bonds)
        if Oatom_idx in modified_indices or Hatom_idx in modified_indices:
            logger.info(
                f"\tSkipping molecule {exc_mol_idxs[exc_mol_idx]}, bond already modified."
            )
            continue
        logger.debug(
            f"\tExciting bond O({Oatom_idx})-H({Hatom_idx}) in molecule {exc_mol_idxs[exc_mol_idx]}..."
        )

        # Get current positions and velocities
        pos_O = Oatom.position
        pos_H = Hatom.position
        vel_O = Oatom.velocity
        vel_H = Hatom.velocity

        # Perform Wigner sampling for this bond
        samples_rp = sample_from_wigner(wigner_target, r_grid, p_grid)
        if not samples_rp:
            logger.warning(
                f"\tFailed to sample for molecule {exc_mol_idxs[exc_mol_idx]}. Skipping."
            )
            continue

        all_samples.extend(samples_rp)
        r_sampled, p_sampled = samples_rp[0]
        modified_indices.add(Oatom_idx)
        modified_indices.add(Hatom_idx)

        pos_O_new, pos_H_new, vel_O_new, vel_H_new = remap_coords_vels(
            pos_O, pos_H, vel_O, vel_H, r_sampled, p_sampled
        )

        # Update positions and velocities in the molecule
        Oatom.position = pos_O_new
        Hatom.position = pos_H_new
        Oatom.velocity = vel_O_new
        Hatom.velocity = vel_H_new

        delta_R = np.linalg.norm((pos_H_new - pos_O_new) - (pos_H - pos_O))
        delta_V = np.linalg.norm((vel_H_new - vel_O_new) - (vel_H - vel_O))
        new_V_mod = np.linalg.norm(vel_H_new - vel_O_new)

        logger.debug(
            f"\tSampled displacement: {r_sampled:.3f} Å\n"
            f"\tDelta R: {delta_R:.4f} Å\n"
            f"\tSampled momentum: {p_sampled:.3f} amu·Å/fs\n"
            f"\tNew velocity modulus: {new_V_mod:.4f} Å/fs\n"
            f"\tNew velocity for H atom: {np.linalg.norm(vel_H_new):.3f} Å/fs\n"
            f"\tNew velocity for O atom: {np.linalg.norm(vel_O_new):.3f} Å/fs\n"
            f"\tDelta V: {delta_V:.4f} Å/fs\n"
        )

    exc_indexes = np.array(list(modified_indices), dtype=int)

    if do_plot:
        plot_wigner(
            r_grid,
            energies,
            wavefunctions,
            wigner_target,
            p_grid,
            all_samples,
            excite_lvl,
        )

    return molecules, exc_indexes


# -------------------------------------------------------
#       --- Function to write XYZ data file ---
# -------------------------------------------------------
def writeXYZ(
    filename: str, molecules: List[List[Atom]], simulation: Simulation
) -> None:
    lattice_string = f'Lattice="{simulation.lattice_string}"'
    properties_string = f"Properties={simulation.properties_string}"

    with open(filename, "w") as f:
        f.write(f"{simulation.n_atoms}\n")
        f.write('pbc="T T T"' + " " + lattice_string + " " + properties_string)

        for molecule in molecules:
            for atom in molecule:
                # charge = 0.5564 if atom.atom_type == 2 else -1.1128
                f.write(
                    f"{atom.atom_string} {atom.position[0]} {atom.position[1]} {atom.position[2]} {atom.mass} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]}\n"
                )

    logger.info(f"\tWrote XYZ data file: {filename}")


def writeLAMMPSdata(
    filename: str,
    molecules: List[List[Atom]],
    box_bounds: NDArray[np.float64],
    atom_style: str,
    units: str,
) -> None:
    Natoms = len(molecules) * 3
    Nbonds = len(molecules) * 2
    Nangles = len(molecules)

    with open(filename, "w") as f:
        f.write("LAMMPS data file for Wigner excitation\n\n")
        f.write(f"{Natoms} atoms\n")
        f.write("2 atom types\n")

        if atom_style == "full":
            f.write(f"{Nbonds} bonds\n")
            f.write("1 bond types\n")
            f.write(f"{Nangles} angles\n")
            f.write("1 angle types\n")

        f.write(f"\n-0.0 {box_bounds[0][0]} xlo xhi\n")
        f.write(f"-0.0 {box_bounds[1][1]} ylo yhi\n")
        f.write(f"-0.0 {box_bounds[2][2]} zlo zhi\n\n")

        f.write(f"Atoms # {atom_style}\n\n")

        if atom_style == "full":
            for mol_idx, molecule in enumerate(molecules):
                for atom in molecule:
                    if units == "metal":
                        atom.velocity *= 1e3  # Convert back from A/fs to A/ps
                    charge = 0.5564 if atom.atom_type == 2 else -1.1128
                    f.write(
                        f"{atom.index} {mol_idx + 1} {atom.atom_type} {charge} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
                    )

        else:
            for mol_idx, molecule in enumerate(molecules):
                for atom in molecule:
                    if units == "metal":
                        atom.velocity *= 1e3  # Convert back from A/fs to A/ps
                    f.write(
                        f"{atom.index} {atom.atom_type} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
                    )

        f.write("\nVelocities\n\n")
        for mol_idx, molecule in enumerate(molecules):
            for atom in molecule:
                f.write(
                    f"{atom.index} {atom.velocity[0]} {atom.velocity[1]} {atom.velocity[2]}\n"
                )

        if atom_style == "full":
            f.write("\nBonds\n\n")
            bond_idx = 1
            for molecule in molecules:
                Oatom = molecule[0]
                Hatom1 = molecule[1]
                Hatom2 = molecule[2]
                f.write(f"{bond_idx} 1 {Oatom.index} {Hatom1.index}\n")
                bond_idx += 1
                f.write(f"{bond_idx} 1 {Oatom.index} {Hatom2.index}\n")
                bond_idx += 1

            f.write("\nAngles\n\n")
            for angle_idx, molecule in enumerate(molecules):
                Oatom = molecule[0]
                Hatom1 = molecule[1]
                Hatom2 = molecule[2]
                f.write(
                    f"{angle_idx + 1} 1 {Hatom1.index} {Oatom.index} {Hatom2.index}\n"
                )

    logger.info(f"\tWrote LAMMPS data file: {filename}")


def plot_wigner(
    r_grid: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    wigner_grid: Optional[NDArray[np.float64]],
    p_grid: Optional[NDArray[np.float64]],
    samples: Optional[List[Tuple[float, float]]],
    state_level: int,
) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook"])

    # Determine figure layout based on whether we have Wigner data
    if wigner_grid is not None and p_grid is not None:
        # Two-panel layout: potential/wavefunctions on left, Wigner function on right
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        # Single panel: only potential/wavefunctions
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # ========== Left panel: Potential and wavefunctions ==========
    V_grid = np.array([V_LS_hydrogen_motion(r) for r in r_grid])
    energies = eigenvalues
    scale = 1  # Scale factor for wavefunction visualization

    for i in range(eigenvectors.shape[1]):
        ax1.plot(
            r_grid,
            energies[i] * CONSTANTS["EV_TO_CM1"]
            + scale * eigenvectors[:, i] * CONSTANTS["EV_TO_CM1"],
            label=f"$\\psi_{i}$ (E={energies[i] * CONSTANTS['EV_TO_CM1']:.0f} cm$^{{-1}}$)",
            color=f"C{i}",
        )
        ax1.axhline(
            energies[i] * CONSTANTS["EV_TO_CM1"],
            color=f"C{i}",
            linestyle="--",
            lw=1,
            alpha=0.5,
        )
        # Add transition energy gap labels from ground state (only for i > 0)
        if i > 0:
            delta_E = (energies[i] - energies[0]) * CONSTANTS["EV_TO_CM1"]
            # Position labels closer to the target energy level to avoid overlap
            y_pos = energies[i] * CONSTANTS["EV_TO_CM1"] - 800
            ax1.text(
                r_grid[-1] * 0.98,
                y_pos,
                f"$\\Delta E_{{0\\to{i}}}$ = {delta_E:.0f} cm$^{{-1}}$",
                color=f"C{i}",
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor=f"C{i}",
                ),
            )

    ax1.plot(r_grid, V_grid * CONSTANTS["EV_TO_CM1"], color="black", label="V(r)", lw=2)
    ax1.set_xlabel("r [Å]")
    ax1.set_ylabel("Energy [cm$^{-1}$]")
    ax1.set_xlim(r_grid[0], r_grid[-1])
    ax1.set_title("Lippincott-Schroeder Potential and Wavefunctions")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # ========== Right panel: Wigner function for selected state ==========
    if wigner_grid is not None and p_grid is not None:
        # Plot the Wigner function for the sampled state
        im = ax2.contourf(r_grid, p_grid, wigner_grid.T, levels=50, cmap="seismic")  # type: ignore
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)  # type: ignore
        cbar.set_label(f"Wigner Function (n={state_level})")

        ax2.set_xlabel("r [Å]")  # type: ignore
        ax2.set_ylabel("p [amu·Å/fs]")  # type: ignore
        ax2.set_title(f"Wigner Distribution for n={state_level} (sampled state)")  # type: ignore
        ax2.grid(alpha=0.3)  # type: ignore

        # Plot sampled points if provided
        if samples is not None and len(samples) > 0:
            r_samples, p_samples = zip(*samples)
            ax2.scatter(  # type: ignore
                r_samples,
                p_samples,
                color="black",
                marker="o",
                s=20,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
                label=f"Samples (N={len(samples)})",
                zorder=10,
            )
            ax2.legend(loc="upper right")  # type: ignore

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# --- Main function to run the code ---
# -------------------------------------------------------
def main():
    # --- Argument parsing ---
    parser = ArgumentParser(
        prog="wigner.py",
        description="Script to excite water molecules in a XYZ dump file using Wigner Sampling.",
    )
    parser.add_argument(
        "dumpfile",
        type=str,
        help="Input XYZ dump file (e.g., 'dump.xyz').",
    )
    parser.add_argument(
        "-a",
        "--atom_per_molecule",
        type=int,
        default=3,
        help="Number of atoms per molecule [default: 3].",
    )
    parser.add_argument(
        "-e",
        "--excite_perc",
        type=float,
        default=0.1,
        help="Fraction (0-1) of molecules to excite [default: 0.1 = 10 percent].",
    )
    parser.add_argument(
        "-l",
        "--excite_lvl",
        type=int,
        choices=[0, 1, 2],
        help="Excitation level: 0 (ground state), 1 (1st excited state) or 2 (2nd excited state) [default=1].",
    )
    parser.add_argument(
        "-d",
        "--datafile",
        type=str,
        default="excited.xyz",
        help="Output XYZ data file name [default: 'excited.xyz'].",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the Wigner function and wavefunctions.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Enable verbose output. Use -vv for more verbosity.",
    )
    parser.add_argument(
        "--keep_vels",
        action="store_true",
        default=True,
        help="Keep original Maxwell-Boltzmann velocities of the molecules.",
    )
    # parser.add_argument(
    #     "-u",
    #     "--units",
    #     type=str,
    #     default="metal",
    #     choices=["metal", "real"],
    #     help="Units for LAMMPS data file [default: metal].",
    # )

    args: Namespace = parser.parse_args()

    # Configure logger for this module only (isolated from matplotlib, etc.)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    # Set logging level based on verbosity
    if args.verbose is None:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    # matplotlib and other libraries remain at their default WARNING level
    # (no need to explicitly silence them since we're not using basicConfig)

    context = {
        "plot": args.plot,
        "excite_lvl": args.excite_lvl,
        "excite_perc": args.excite_perc,
    }

    molecules, simulation = readGPUMDdump(
        args.dumpfile, args.atom_per_molecule, args.keep_vels
    )

    exc_molecules, exc_indexes = excite_molecules(molecules, context)

    logger.info(
        f"\tSuccessfully excited {int(len(molecules) * args.excite_perc)} out of {len(molecules)} molecules."
    )

    writeXYZ(args.datafile, exc_molecules, simulation)

    exc_file = args.datafile.replace(".xyz", "-indexes.dat")
    np.savetxt(exc_file, np.sort(exc_indexes), fmt="%d")
    logger.info(f"\tModified atom indexes saved in {exc_file}.")

    # writeLAMMPSdata(
    #     args.datafile.replace(".xyz", ".data"),
    #     exc_molecules,
    #     simulation.cell_vectors,
    #     atom_style="atomic",
    #     units="metal",
    # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("NOOOO DON'T KILL MEEEEeeee...\n")
        sys.exit(0)
