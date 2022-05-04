"""Put commonly used constants here"""

from immutabledict import immutabledict as idict

import numpy as np

import strax

export, __all__ = strax.exporter()

##
# Detector geometry constants
# You could convert these to config options, if you want to vary them
##

ANODE_Z = 40  # cm
CATHODE_Z = -80  # cm
TPC_LENGTH = ANODE_Z - CATHODE_Z  # cm; sensitivity paper says 118.3 cm
TPC_RADIUS = 56.7  # cm, from sensitivity paper
ANODE_PITCH = 0.6  # cm
CHARGE_READOUT_DT = 10  # ns
TILE_WIDTH_IN_PADS = 16


__all__ += [
    "ANODE_Z",
    "CATHODE_Z",
    "TPC_LENGTH",
    "TPC_RADIUS",
    "ANODE_PITCH",
    "CHARGE_READOUT_DT",
    "TILE_WIDTH_IN_PADS",
]

##
#  Data types used by more than one plugin
##
nest_hits_dtype = strax.time_fields + [
    ("x", np.float64, "X-coordinate of hit [cm]"),
    ("y", np.float64, "Y-coordinate of hit [cm]"),
    ("z", np.float64, "Z-coordinate of hit [cm]"),
    ("energy", np.float64, "Energy of hit [keV]"),
    ("n_photons", np.int64, "Number of scintillation photons"),
    ("n_electrons", np.int64, "Number of thermalized electrons"),
    ("interaction_type", np.uint8, "NEST interaction type number"),
    (
        "t_parent",
        np.int64,
        "Time of interaction that generated the hit, ns since epoch [ns]",
    ),
]
__all__ += ["nest_hits_dtype"]


##
# Other constants
##

NEST_INTERACTION_TYPE = idict(ion=6, gammaray=7, beta=8)
__all__ += ["NEST_INTERACTION_TYPE"]
