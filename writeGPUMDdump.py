import numpy as np

def writeXYZ(
    filename: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    cellsv: np.ndarray,
    types: np.ndarray,
    unwr_positions: np.ndarray | None = None,
) -> None:
    """
    Scrive le coordinate unwrapped in un file XYZ.
    """
    N_atoms = positions.shape[0]
    latticeStr = f'"{cellsv[0, 0]} 0.0 0.0 0.0 {cellsv[1, 1]} 0.0 0.0 0.0 {cellsv[2, 2]}"'
    masses = {1: 15.999, 2: 1.008}  # Example masses for O and H
    species = {1: "O", 2: "H"}

    if unwr_positions is not None:
        propertiesStr = 'species:S:1:pos:R:3:mass:R:1:vel:R:3:unwrapped:R:3'

        with open(filename, "w") as f:
            f.write(f"{N_atoms}\n")
            f.write(
                f'pbc="T T T" Lattice={latticeStr} Properties={propertiesStr}\n'
            )
            for i in range(N_atoms):
                f.write(
                    f"{species[types[i]]} {positions[i][0]} {positions[i][1]} {positions[i][2]} {masses[types[i]]} {velocities[i][0]} {velocities[i][1]} {velocities[i][2]} {unwr_positions[i][0]} {unwr_positions[i][1]} {unwr_positions[i][2]}\n"
                )

            
    else:
        propertiesStr = 'species:S:1:pos:R:3:mass:R:1:vel:R:3'

        with open(filename, "w") as f:
            f.write(f"{N_atoms}\n")
            f.write(
                f'pbc="T T T" Lattice={latticeStr} Properties={propertiesStr}\n'
            )
            for i in range(N_atoms):
                f.write(
                    f"{species[types[i]]} {positions[i][0]} {positions[i][1]} {positions[i][2]} {masses[types[i]]} {velocities[i][0]} {velocities[i][1]} {velocities[i][2]}\n"
                )

    print(f"File {filename} written successfully.")
