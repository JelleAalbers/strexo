"""Data types that are/may be used by multiple plugins
"""
import numpy as np

import strax

__all__ = ["nest_hits", "mc_decays"]


nest_hits = strax.time_fields + [
    ("x", np.float64, "X-coordinate of hit [cm]"),
    ("y", np.float64, "Y-coordinate of hit [cm]"),
    ("z", np.float64, "Z-coordinate of hit [cm]"),
    # TODO: should this be eV instead for consistency with units.py?
    ("energy", np.float64, "Energy of hit [keV]"),
    ("n_photons", np.int64, "Number of scintillation photons"),
    ("n_electrons", np.int64, "Number of thermalized electrons"),
    ("interaction_type", np.uint8, "NEST interaction type number"),
    (
        "parent_time",
        np.int64,
        "Time of interaction that generated the hit, since epoch [ns]",
    ),
    (
        "source_index",
        np.int16,
        "Index of the parent decay's source in config['mc_sources']",
    ),
]

mc_decays = strax.time_fields + [
    ("source_index", np.int16, "Index of the decay's source in config['mc_sources']",),
    ("energy", np.float64, "Sum of energies of all NEST hits [keV]",),
    # TODO: xyz positions of original decay.
]
