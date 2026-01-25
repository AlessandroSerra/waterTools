import math
from typing import Tuple

import numpy as np
from numba import njit, prange
from scipy.integrate import simpson


@njit(parallel=True, fastmath=True, cache=True)
def _rdf_hist_all_frames(positions, idx_i, idx_j, Lx, Ly, Lz, r_max, dr, same_species):
    """
    Calcola l'istogramma delle distanze per TUTTI i frame in parallelo.

    positions : (n_frames, n_atoms, 3)
    idx_i, idx_j : indici degli atomi per la coppia considerata
    restituisce: hist_t con shape (n_frames, n_bins)
    """
    n_frames = positions.shape[0]
    n_bins = int(r_max / dr)
    hist_t = np.zeros((n_frames, n_bins), dtype=np.float64)

    for f in prange(n_frames):
        pos_f = positions[f]
        hist = hist_t[f]

        for a in range(idx_i.size):
            i = idx_i[a]
            xi = pos_f[i, 0]
            yi = pos_f[i, 1]
            zi = pos_f[i, 2]

            for b in range(idx_j.size):
                j = idx_j[b]
                if same_species and i == j:
                    continue

                xj = pos_f[j, 0]
                yj = pos_f[j, 1]
                zj = pos_f[j, 2]

                dx = xi - xj
                dy = yi - yj
                dz = zi - zj

                # MIC ortorombica
                dx -= Lx * np.rint(dx / Lx)
                dy -= Ly * np.rint(dy / Ly)
                dz -= Lz * np.rint(dz / Lz)

                r = math.sqrt(dx * dx + dy * dy + dz * dz)

                if 0.0 < r <= r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 1.0

    return hist_t


def compute_rdf(
    positions: np.ndarray,
    cell_vectors: np.ndarray,
    r_max: float | None = None,
    n_bins: int = 200,
    return_time_series: bool = True,
    excitation_mode: str = "none",
    excited_index_file: str | None = None,
):
    """
    Compute O–O, O–H and H–H radial distribution functions for a water system
    with atoms ordered as O–H–H per molecule. The calculation applies PBC
    (orthorhombic box) and parallelizes over frames using numba.

    Parameters
    ----------
    positions : ndarray, shape (n_frames, n_atoms, 3)
        Atomic coordinates for each frame (O–H–H repeating order).
    cell_vectors : ndarray, shape (3, 3)
        Box matrix (assumed orthorhombic; lengths taken from diagonal).
    r_max : float, optional
        Maximum distance for the RDF. Defaults to half of the smallest box length.
    n_bins : int
        Number of histogram bins.
    return_time_series : bool
        If True, return g(r) for each frame individually; otherwise return the
        time-averaged RDF.
    excitation_mode : {"none", "excited", "ground", "all"}
        - "none": ignore excitation, use all molecules (standard RDF).
        - "excited": compute RDF using only excited molecules.
        - "ground": compute RDF using only non-excited molecules.
        - "all": compute RDF for full system, excited-only and ground-only.
        Requires excited_index_file for any mode other than "none".
    excited_index_file : str or None
        Path to a .dat file containing 1-based *atomic* indices of excited atoms.
        Molecules are marked as excited if at least one of their atoms appears
        in this list.

    Returns
    -------
    dict
        If excitation_mode == "none" or excited_index_file is None:
            {'OO': (r, g_OO[_t]), 'OH': (r, g_OH[_t]), 'HH': (r, g_HH[_t])}
        If excitation_mode == "excited" or "ground":
            same structure, but computed on the selected subset only.
        If excitation_mode == "all":
            {
                "full":   {'OO': (r, g_OO[_t]), ...},
                "excited":{'OO': (r, g_OO_exc[_t]), ...},
                "ground": {'OO': (r, g_OO_gs[_t]), ...},
            }
        where g_XX has shape (n_bins,) if return_time_series=False,
        or (n_frames, n_bins) if return_time_series=True.
    """
    positions = np.asarray(positions)
    cell_vectors = np.asarray(cell_vectors, dtype=float)

    assert positions.ndim == 3 and positions.shape[2] == 3
    assert cell_vectors.shape == (3, 3)

    # Lunghezze di cella (cella ortorombica)
    Lx = cell_vectors[0, 0]
    Ly = cell_vectors[1, 1]
    Lz = cell_vectors[2, 2]

    n_frames, n_atoms, _ = positions.shape
    assert n_atoms % 3 == 0, "n_atoms deve essere multiplo di 3 (OHH)."

    n_mol = n_atoms // 3

    if r_max is None:
        r_max = 0.5 * min(Lx, Ly, Lz)

    dr = r_max / n_bins
    r = (np.arange(n_bins) + 0.5) * dr
    volume = Lx * Ly * Lz

    shell_volumes = (4.0 * np.pi / 3.0) * ((r + 0.5 * dr) ** 3 - (r - 0.5 * dr) ** 3)

    # Helper: calcola RDF per un dato set di indici O/H
    def _rdf_for_subset(idx_O, idx_H_all):
        NO_loc = idx_O.size
        NH_loc = idx_H_all.size

        if NO_loc == 0 or NH_loc == 0:
            raise ValueError("Subset vuoto: nessun O o H selezionato.")

        # Istogrammi frame-per-frame (paralleli sui frame)
        hist_OO_t = _rdf_hist_all_frames(
            positions, idx_O, idx_O, Lx, Ly, Lz, r_max, dr, True
        )
        hist_OH_t = _rdf_hist_all_frames(
            positions, idx_O, idx_H_all, Lx, Ly, Lz, r_max, dr, False
        )
        hist_HH_t = _rdf_hist_all_frames(
            positions, idx_H_all, idx_H_all, Lx, Ly, Lz, r_max, dr, True
        )

        # Normalizzazioni
        norm_pairs_OO = NO_loc * (NO_loc - 1)
        norm_pairs_OH = NO_loc * NH_loc
        norm_pairs_HH = NH_loc * (NH_loc - 1)

        base_norm_OO = volume / norm_pairs_OO
        base_norm_OH = volume / norm_pairs_OH
        base_norm_HH = volume / norm_pairs_HH

        if return_time_series:
            g_OO_t = base_norm_OO * (hist_OO_t / shell_volumes)
            g_OH_t = base_norm_OH * (hist_OH_t / shell_volumes)
            g_HH_t = base_norm_HH * (hist_HH_t / shell_volumes)
            return {
                "OO": (r, g_OO_t),
                "OH": (r, g_OH_t),
                "HH": (r, g_HH_t),
            }
        else:
            hist_OO = hist_OO_t.sum(axis=0)
            hist_OH = hist_OH_t.sum(axis=0)
            hist_HH = hist_HH_t.sum(axis=0)

            g_OO = (base_norm_OO / n_frames) * (hist_OO / shell_volumes)
            g_OH = (base_norm_OH / n_frames) * (hist_OH / shell_volumes)
            g_HH = (base_norm_HH / n_frames) * (hist_HH / shell_volumes)
            return {
                "OO": (r, g_OO),
                "OH": (r, g_OH),
                "HH": (r, g_HH),
            }

    # Caso senza eccitazione (comportamento originale)
    if excited_index_file is None or excitation_mode == "none":
        # Indici globali per tutti gli atomi
        idx_O = np.arange(0, n_atoms, 3)
        idx_H1 = np.arange(1, n_atoms, 3)
        idx_H2 = np.arange(2, n_atoms, 3)
        idx_H_all = np.concatenate((idx_H1, idx_H2))
        return _rdf_for_subset(idx_O, idx_H_all)

    if excitation_mode not in {"excited", "ground", "all"}:
        raise ValueError(
            "excitation_mode deve essere uno tra 'none', 'excited', 'ground', 'all'."
        )

    # -----------------------
    # Lettura indici ECCITATI (ATOMICI, 1-based)
    # -----------------------
    exc_atoms = np.loadtxt(excited_index_file, dtype=int)
    exc_atoms = np.atleast_1d(exc_atoms) - 1  # 0-based, array 1D

    if exc_atoms.min() < 0 or exc_atoms.max() >= n_atoms:
        raise ValueError("Indici atomici eccitati fuori range.")

    # Mappa atomi eccitati → molecole eccitate
    # ogni molecola ha 3 atomi: [3*m, 3*m+1, 3*m+2]
    exc_mol = np.unique(exc_atoms // 3)  # indici di molecola 0-based

    mol_mask_exc = np.zeros(n_mol, dtype=bool)
    mol_mask_exc[exc_mol] = True
    mol_mask_gs = ~mol_mask_exc

    # Indici atomici per molecola: shape (n_mol, 3) → [O, H1, H2]
    idx_atoms = np.arange(n_atoms).reshape(n_mol, 3)

    # Subset eccitato
    idx_O_exc = idx_atoms[mol_mask_exc, 0]
    idx_H_exc = idx_atoms[mol_mask_exc, 1:].ravel()

    # Subset non eccitato (ground state)
    idx_O_gs = idx_atoms[mol_mask_gs, 0]
    idx_H_gs = idx_atoms[mol_mask_gs, 1:].ravel()

    # Subset completo (tutti)
    idx_O_all = idx_atoms[:, 0]
    idx_H_all = idx_atoms[:, 1:].ravel()

    if excitation_mode == "excited":
        return _rdf_for_subset(idx_O_exc, idx_H_exc)

    if excitation_mode == "ground":
        return _rdf_for_subset(idx_O_gs, idx_H_gs)

    # excitation_mode == "all": ritorna tutti e tre
    rdf_full = _rdf_for_subset(idx_O_all, idx_H_all)
    rdf_exc = _rdf_for_subset(idx_O_exc, idx_H_exc)
    rdf_gs = _rdf_for_subset(idx_O_gs, idx_H_gs)

    return {
        "full": rdf_full,
        "excited": rdf_exc,
        "ground": rdf_gs,
    }


def compute_R1(bins: np.ndarray, g_r: np.ndarray, r1: float, r2: float) -> float:
    """
    Calcola la distanza media R1 di un shell dal RDF g(r), pesata per r^2:

        R1 = ∫_{r1}^{r2} r * (r^2 * g(r)) dr / ∫_{r1}^{r2} (r^2 * g(r)) dr
           = ∫_{r1}^{r2} r^3 * g(r) dr / ∫_{r1}^{r2} r^2 * g(r) dr

    Parametri
    ---------
    bins : array 1D
        Valori di r (Å).
    g_r : array 1D
        Valori di g(r) corrispondenti.
    r1, r2 : float
        Limiti di integrazione (Å).

    Ritorna
    -------
    float
        Distanza media del shell (Å).
    """
    mask = (bins >= r1) & (bins <= r2)
    r_sel = bins[mask]
    g_sel = g_r[mask]

    if r_sel.size < 2:
        raise ValueError("Intervallo [r1, r2] troppo piccolo o non coperto dai bins.")

    # Numeratore: ∫ r^3 * g(r) dr
    num = simpson((r_sel**3) * g_sel, x=r_sel)

    # Denominatore: ∫ r^2 * g(r) dr
    den = simpson((r_sel**2) * g_sel, x=r_sel)

    if den == 0:
        raise ZeroDivisionError("Denominatore nullo: g(r) ≈ 0 su [r1, r2].")

    return num / den


def propagate_R1_error(
    bins: np.ndarray,
    g_r: np.ndarray,
    g_err: np.ndarray,
    r1: float,
    r2: float,
    n_samples: int = 10_000,
) -> Tuple[float, float]:
    """
    Propaga l'errore su g(r) a R1 via Monte Carlo.
    Versione corretta con peso r^2 e robusta per numero di punti pari/dispari.

    Ritorna
    -------
    R1_mean : float
        Valore medio campionato di R1.
    R1_std : float
        Deviazione standard di R1 (errore su R1).
    """
    mask = (bins >= r1) & (bins <= r2)
    r_sel = bins[mask]
    g_sel = g_r[mask]
    err_sel = g_err[mask]

    if r_sel.size < 2:
        raise ValueError("Intervallo [r1, r2] troppo piccolo per Simpson.")

    # campioni g(r)
    # g_pert shape: (n_samples, n_points)
    g_pert = g_sel[np.newaxis, :] + np.random.normal(
        scale=err_sel, size=(n_samples, r_sel.size)
    )

    # Numeratore: ∫ r^3 * g(r) dr
    # Moltiplichiamo g_pert per r^3
    integrand_num = g_pert * (r_sel**3)[np.newaxis, :]
    # Integriamo lungo l'asse dei punti (axis=-1)
    num = simpson(integrand_num, x=r_sel, axis=-1)

    # Denominatore: ∫ r^2 * g(r) dr
    integrand_den = g_pert * (r_sel**2)[np.newaxis, :]
    den = simpson(integrand_den, x=r_sel, axis=-1)

    R1_samples = num / den
    # filtra eventuali numeri non finiti
    R1_samples = R1_samples[np.isfinite(R1_samples)]

    return R1_samples.mean(), R1_samples.std(ddof=1)


def compute_I2(bins: np.ndarray, g_r: np.ndarray, r1: float, r2: float) -> float:
    """
    Calcola l'integrale di g(r) sul range [r1, r2]:

        I2 = ∫_{r1}^{r2} g(r) dr

    Parametri
    ---------
    bins : array 1D
        Valori di r (Å).
    g_r : array 1D
        Valori di g(r).
    r1, r2 : float
        Limiti di integrazione (Å).

    Ritorna
    -------
    float
        I2 = ∫ g(r) dr su [r1, r2].
    """
    mask = (bins >= r1) & (bins <= r2)
    r_sel = bins[mask]
    g_sel = g_r[mask]

    if r_sel.size < 2:
        raise ValueError("Intervallo [r1, r2] troppo piccolo o non coperto dai bins.")

    return simpson(g_sel, x=r_sel)


def propagate_I2_error(
    bins: np.ndarray,
    g_r: np.ndarray,
    g_err: np.ndarray,
    r1: float,
    r2: float,
    n_samples: int = 1000,
) -> Tuple[float, float]:
    """
    Propaga l'errore su g(r) a I2 via Monte Carlo:

        I2 = ∫_{r1}^{r2} g(r) dr

    Ritorna
    -------
    I2_mean : float
        Media campionata di I2.
    I2_std : float
        Deviazione standard di I2 (errore su I2).
    """
    mask = (bins >= r1) & (bins <= r2)
    r_sel = bins[mask]
    g_sel = g_r[mask]
    err_sel = g_err[mask]

    if r_sel.size < 2:
        raise ValueError("Intervallo [r1, r2] troppo piccolo o non coperto dai bins.")

    # Vectorized implementation for speed and robustness
    g_pert = g_sel[np.newaxis, :] + np.random.normal(
        scale=err_sel, size=(n_samples, r_sel.size)
    )
    I2_samples = simpson(g_pert, x=r_sel, axis=-1)

    I2_samples = I2_samples[np.isfinite(I2_samples)]
    return I2_samples.mean(), I2_samples.std(ddof=1)
