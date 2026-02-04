from . import (
    c_parser,
    lmp2gpumd,
    spectral,
    temps,
    temps_numba_parallel,
    unwrap_coords,
    wignerEXC,
    wignerEXC2B,
    rdf,
    exyz2gpumd,
    writeGPUMDdump,
)

# Expose main functions for better IntelliSense
from .temps import analyzeTEMPS
from .rdf import compute_rdf, compute_R1, compute_I2, propagate_R1_error, propagate_I2_error
from .spectral import calculateVACFgroup, calculateVDOS
from .unwrap_coords import unwrap_coords as unwrap_coordinates
from .exyz2gpumd import read_exyz, write_gpumd
from .writeGPUMDdump import writeXYZ

__all__ = [
    # Modules
    "spectral",
    "unwrap_coords",
    "wignerEXC",
    "wignerEXC2B",
    "lmp2gpumd",
    "c_parser",
    "temps",
    "temps_numba_parallel",
    "rdf",
    "exyz2gpumd",
    "writeGPUMDdump",
    # Functions
    "analyzeTEMPS",
    "compute_rdf",
    "compute_R1",
    "compute_I2",
    "propagate_R1_error",
    "propagate_I2_error",
    "calculateVACFgroup",
    "calculateVDOS",
    "unwrap_coordinates",
    "read_exyz",
    "write_gpumd",
    "writeXYZ",
]
