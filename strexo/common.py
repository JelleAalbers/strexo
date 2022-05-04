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
