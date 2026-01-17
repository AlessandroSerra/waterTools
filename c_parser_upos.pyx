from libc.stdio  cimport FILE, fopen, fclose, fgets
from libc.string cimport strtok, strstr
from libc.stdlib cimport strtoll, strtod
from libc.errno  cimport errno
from libc.stdint cimport int64_t, int8_t
# from libc.ctype  cimport isspace

import numpy as np
cimport numpy as cnp

cnp.import_array()

DEF MAX_LENGTH = 4096  # lunghezza massima di UNA riga (header/atomo)


cpdef tuple parse_xyz_all(bytes filename_bytes, Py_ssize_t every=1):
    """
    Parsing completo di un file extended XYZ con blocchi:

      n_atoms
      header (con Lattice="...")
      n_atoms righe: symbol x y z mass vx vy vz upx upy upz

    Restituisce:
      positions:           (n_frames_kept, n_atoms, 3) float64
      velocities:          (n_frames_kept, n_atoms, 3) float64
      unwrapped_positions: (n_frames_kept, n_atoms, 3) float64
      cell_vecs:           (3, 3) float64
      atom_types:          (n_atoms,) int8   (1=O, 2=H)
    """

    cdef const char* filename = filename_bytes
    cdef FILE* fptr
    cdef char buffer[MAX_LENGTH]
    cdef char* endptr
    cdef Py_ssize_t frame = 0
    cdef Py_ssize_t kept_frames = 0
    cdef Py_ssize_t n_atoms = -1
    cdef Py_ssize_t n_atoms_this
    cdef int64_t n_val

    cdef cnp.npy_intp dims3[3]
    cdef cnp.npy_intp dims2[2]
    cdef cnp.npy_intp dims1[1]

    cdef cnp.ndarray[cnp.float64_t, ndim=3] pos_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=3] vel_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=3] unwrapped_pos_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=2] cell_arr
    cdef cnp.ndarray[cnp.int8_t,    ndim=1] types_arr

    cdef double* pos_data
    cdef double* vel_data
    cdef double* unwrapped_pos_data
    cdef double* cell_data
    cdef int8_t* types_data

    cdef Py_ssize_t i, j, k
    cdef int c
    cdef char* token
    cdef char* p
    cdef char* lat_ptr
    cdef double val

    cdef Py_ssize_t base_pos
    cdef Py_ssize_t base_vel
    cdef Py_ssize_t base_unwrapped_pos

    if every <= 0:
        raise ValueError("every deve essere >= 1")

    # -----------------------------
    # Primo passaggio: conta frame e n_atoms
    # -----------------------------
    fptr = fopen(filename, b"r")
    if fptr == NULL:
        raise OSError("Failed opening file")

    try:
        while True:
            if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                break  # EOF

            # parse n_atoms from first line
            errno = 0
            n_val = strtoll(buffer, &endptr, 10)
            if errno != 0 or endptr == buffer:
                raise ValueError("Riga n_atoms non valida")

            n_atoms_this = <Py_ssize_t> n_val
            if n_atoms_this <= 0:
                raise ValueError("n_atoms deve essere > 0")

            if n_atoms < 0:
                n_atoms = n_atoms_this
            else:
                if n_atoms_this != n_atoms:
                    raise ValueError("n_atoms variabile fra i frame (non supportato)")

            # header line
            if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                raise ValueError("File troncato: manca header dopo n_atoms")

            # skip tutte le righe degli atomi
            for i in range(n_atoms):
                if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                    raise ValueError("File troncato nelle righe atomo")

            if (frame % every) == 0:
                kept_frames += 1

            frame += 1
    finally:
        fclose(fptr)

    if n_atoms <= 0 or kept_frames <= 0:
        raise ValueError("Nessun frame valido trovato nel file")

    # -----------------------------
    # Allocazione array NumPy
    # -----------------------------
    # positions / velocities / unwrapped_positions: (kept_frames, n_atoms, 3)
    dims3[0] = kept_frames
    dims3[1] = n_atoms
    dims3[2] = 3

    pos_arr = <cnp.ndarray> cnp.PyArray_SimpleNew(3, dims3, cnp.NPY_FLOAT64)
    vel_arr = <cnp.ndarray> cnp.PyArray_SimpleNew(3, dims3, cnp.NPY_FLOAT64)
    unwrapped_pos_arr = <cnp.ndarray> cnp.PyArray_SimpleNew(3, dims3, cnp.NPY_FLOAT64)
    if pos_arr is None or vel_arr is None or unwrapped_pos_arr is None:
        raise MemoryError("Allocazione array pos/vel/unwrapped_pos fallita")

    pos_data = <double*> pos_arr.data
    vel_data = <double*> vel_arr.data
    unwrapped_pos_data = <double*> unwrapped_pos_arr.data

    # cell_vecs: (3,3)
    dims2[0] = 3
    dims2[1] = 3
    cell_arr = <cnp.ndarray> cnp.PyArray_SimpleNew(2, dims2, cnp.NPY_FLOAT64)
    if cell_arr is None:
        raise MemoryError("Allocazione cell_vecs fallita")
    cell_data = <double*> cell_arr.data

    # atom_types: (n_atoms,) int8
    dims1[0] = n_atoms
    types_arr = <cnp.ndarray> cnp.PyArray_SimpleNew(1, dims1, cnp.NPY_INT8)
    if types_arr is None:
        raise MemoryError("Allocazione atom_types fallita")
    types_data = <int8_t*> types_arr.data

    for i in range(n_atoms):
        types_data[i] = 0

    # -----------------------------
    # Secondo passaggio: parsing reale
    # -----------------------------
    fptr = fopen(filename, b"r")
    if fptr == NULL:
        raise OSError("Failed reopening file")

    frame = 0
    cdef Py_ssize_t kept_idx = 0

    try:
        while True:
            if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                break  # EOF

            # n_atoms linea
            errno = 0
            n_val = strtoll(buffer, &endptr, 10)
            if errno != 0 or endptr == buffer:
                raise ValueError("Riga n_atoms non valida (secondo passaggio)")
            n_atoms_this = <Py_ssize_t> n_val
            if n_atoms_this != n_atoms:
                raise ValueError("n_atoms incoerente nel secondo passaggio")

            # header
            if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                raise ValueError("File troncato: manca header nel secondo passaggio")

            if (frame % every) == 0 and kept_idx == 0:
                lat_ptr = strstr(buffer, b"Lattice=\"")
                if lat_ptr == NULL:
                    raise ValueError("Header privo di campo Lattice=\"...\"")
                lat_ptr += 9  # len("Lattice=\"")
                p = lat_ptr

                for k in range(9):
                    errno = 0
                    val = strtod(p, &endptr)
                    if errno != 0 or endptr == p:
                        raise ValueError("Parsing Lattice= fallito")
                    cell_data[k] = val
                    p = endptr


            if (frame % every) != 0:
                for i in range(n_atoms):
                    if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                        raise ValueError("File troncato nelle righe atomo (skip)")
                frame += 1
                continue

            # --- Frame da tenere: parse delle n_atoms righe ---
            for i in range(n_atoms):
                if fgets(buffer, MAX_LENGTH, fptr) == NULL:
                    raise ValueError("File troncato nelle righe atomo (parse)")

                token = strtok(buffer, b" \t\n")
                if token == NULL:
                    raise ValueError("Riga atomo vuota o malformata")

                # atom_types: solo al primo frame tenuto
                if kept_idx == 0:
                    c = <unsigned char> token[0]
                    if c >= ord('a') and c <= ord('z'):
                        c = c - ord('a') + ord('A')

                    if c == ord('O'):
                        types_data[i] = <int8_t> 1
                    else:
                        types_data[i] = <int8_t> 2

                base_pos = (kept_idx * n_atoms + i) * 3
                base_vel = (kept_idx * n_atoms + i) * 3
                base_unwrapped_pos = (kept_idx * n_atoms + i) * 3

                for k in range(10):
                    token = strtok(NULL, b" \t\n")
                    if token == NULL:
                        raise ValueError("Numero di colonne numeriche insufficiente in riga atomo")

                    errno = 0
                    val = strtod(token, &endptr)
                    if errno != 0 or endptr == token:
                        raise ValueError("Token numerico invalido in riga atomo")

                    if k < 3:
                        # x, y, z
                        pos_data[base_pos + k] = val
                    elif k == 3:
                        # mass: ignorata
                        pass
                    elif k < 7:
                        # vx, vy, vz
                        vel_data[base_vel + (k - 4)] = val
                    else:
                        # upx, upy, upz
                        unwrapped_pos_data[base_unwrapped_pos + (k - 7)] = val

            kept_idx += 1
            frame += 1

    finally:
        fclose(fptr)

    return pos_arr, vel_arr, unwrapped_pos_arr, cell_arr, types_arr
