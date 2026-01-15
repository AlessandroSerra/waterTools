import numpy as np


def unwrap_coords(
    positions_wrapped: np.ndarray,
    cell_vectors: np.ndarray,
    oxygen_unwrapped: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ricostruisce coordinate unwrapped per tutte le molecole, usando gli O come riferimento.
    Per velocità, assume che ogni molecola abbia esattamente 3 atomi e siano disposti come OHH.

    Parametri
    ----------
    positions_wrapped : np.ndarray
        Array di shape (n_frames, n_atoms, 3) con le coordinate wrapped
        (come escono dal codice MD, dentro il box).
    cell_vectors : np.ndarray
        Matrice 3x3 dei vettori di cella.
        Si assume cella ortorombica (vettori ~ diagonali).

    Ritorna
    -------
    unwrapped : np.ndarray
        Array di shape (n_frames, n_atoms, 3) con coordinate unwrapped
        coerenti per molecola (gli H sono sempre vicini al proprio O).
    """

    pos = np.asarray(positions_wrapped, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("positions_wrapped deve avere shape (n_frames, n_atoms, 3).")

    n_frames, n_atoms, _ = pos.shape

    if n_atoms % 3 != 0:
        raise ValueError("Questa versione assume 3 atomi per molecola (OHH).")

    n_mol = n_atoms // 3

    # reshape in (frames, molecole, atomi_per_mol=3, 3)
    pos_mol = pos.reshape(n_frames, n_mol, 3, 3)

    # lunghezze di cella
    cell_vectors = np.asarray(cell_vectors, dtype=float)
    if cell_vectors.shape != (3, 3):
        raise ValueError("cell_vectors deve avere shape (3, 3).")

    L = np.linalg.norm(cell_vectors, axis=1)  # (3,)
    L_b = L.reshape(1, 1, 1, 3)  # per broadcasting

    # O wrapped (indice 0) per tutte le molecole e frame
    O_wrapped = pos_mol[:, :, 0:1, :]  # (F, M, 1, 3)

    # O anchor (unwrapped di riferimento)
    if oxygen_unwrapped is None:
        O_anchor = O_wrapped[..., 0, :]  # (F, M, 3)
    else:
        oxy_unw = np.asarray(oxygen_unwrapped, dtype=float)
        if oxy_unw.shape != pos.shape:
            raise ValueError(
                "oxygen_unwrapped deve avere la stessa shape di positions_wrapped."
            )
        O_anchor = oxy_unw.reshape(n_frames, n_mol, 3, 3)[:, :, 0, :]  # (F, M, 3)

    # output
    unwrapped_mol = np.empty_like(pos_mol)

    # assegna O
    unwrapped_mol[:, :, 0, :] = O_anchor

    # H wrapped (indici 1 e 2)
    H_wrapped = pos_mol[:, :, 1:3, :]  # (F, M, 2, 3)

    # vettore H-O in coordinate wrapped
    delta_wrapped = H_wrapped - O_wrapped  # broadcasting -> (F, M, 2, 3)

    # minimum image su tutte le H in un colpo solo
    delta_local = delta_wrapped - np.round(delta_wrapped / L_b) * L_b

    # O_anchor con dimensione extra per le H
    O_anchor_expanded = O_anchor[:, :, None, :]  # (F, M, 1, 3)

    # H unwrapped = O_anchor + minimum-image(H-O)
    unwrapped_mol[:, :, 1:3, :] = O_anchor_expanded + delta_local

    # torna a (n_frames, n_atoms, 3)
    return unwrapped_mol.reshape(n_frames, n_atoms, 3)
