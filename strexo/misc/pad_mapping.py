"""Functions relating pad indices, positions, and channel numbers

Pad indices can be negative; with (0,0) the pad with positive coordinates
and the TPC center at its lower left corner

For simplicity, we assume entire square lid on TPC is instrumented,
and clip x & y outside TPC radius.
"""

from re import X
import numpy as np

import strax
import strexo

export, __all__ = strax.exporter()

# Compute basic geometry
anode_width_in_tiles = np.ceil(
    2 * strexo.TPC_RADIUS / strexo.ANODE_PITCH / strexo.TILE_WIDTH_IN_PADS
).astype(int)
anode_width_in_pads = strexo.TILE_WIDTH_IN_PADS * anode_width_in_tiles
channels_per_tile = strexo.TILE_WIDTH_IN_PADS * 2
channels_per_tile_row = channels_per_tile * anode_width_in_tiles

# Compute channel map for one tile, starting at 0
n = strexo.TILE_WIDTH_IN_PADS
alt_mask = np.array([(-1) ** (i + j) == 1 for i in range(n) for j in range(n)])
x_channels = np.repeat(np.arange(n), n)
y_channels = np.tile(np.arange(n, 2 * n), n)
tile_channels = (x_channels * alt_mask + y_channels * ~alt_mask).reshape(n, n)
assert tile_channels.min() == 0
assert tile_channels.max() == channels_per_tile - 1

# Compute full channel map
pad_channel = np.hstack(
    [
        np.vstack(
            [
                tile_channels + xi * channels_per_tile + yi * channels_per_tile_row
                for xi in range(anode_width_in_tiles)
            ]
        )
        for yi in range(anode_width_in_tiles)
    ]
)
# All channels are used
assert np.all(np.diff(np.unique(pad_channel)) == 1)
# Each channel connected to TILE_WIDTH_IN_PADS//2 pads
assert np.all(np.bincount(pad_channel.ravel()) == strexo.TILE_WIDTH_IN_PADS // 2)

center_i = int(anode_width_in_pads / 2)


@export
def pos_to_pad(x, y):
    """Return (pad_xi, pad_yi, dx, dy) at position x,y

    Arguments:
      - x: array of floats, x-coordinate relative to TPC center
      - y: array of floats, y-coordinate relative to TPC center

    Returns: 4-tuple of arrays
      - pad_xi: x-index of pads reading out (x, y)
      - pad_yi: y-index of pads reading out (x, y)
      - dx: x-distance between lower left of the pad and (x, y) position
      - dy: y-distance between lower left of the pad and (x, y) position
    """
    ap = strexo.ANODE_PITCH
    pad_xi = (x // ap).astype(np.int32)
    pad_yi = (y // ap).astype(np.int32)
    dx, dy = x % ap, y % ap
    return pad_xi, pad_yi, dx, dy


@export
def pos_to_channel(x, y):
    """Return channel read out by the pad at position (x,y)

    Arguments:
      - x: array of floats, x-coordinate relative to TPC center
      - y: array of floats, y-coordinate relative to TPC center

    Returns: array of ints, channel numbers corresponding to positions
    """
    # Convert to pad index accepted by pad_to_channel
    x_index = (np.asarray(x) // strexo.ANODE_PITCH).astype(int)
    y_index = (np.asarray(y) // strexo.ANODE_PITCH).astype(int)
    return pad_to_channel(x_index, y_index)


@export
def pad_to_channel(x_index, y_index):
    """Return channels read out by pads with indices (xi, yi)"""
    # Convert to 0-based indices in pad_channel array
    x_index = center_i + np.asarray(x_index)
    y_index = center_i + np.asarray(y_index)
    max_index = len(pad_channel) - 1
    return pad_channel[x_index.clip(0, max_index), y_index.clip(0, max_index)]


# Compute (x, y) coordinates of each pad
assert anode_width_in_pads % 2 == 0
center_i = int(anode_width_in_pads / 2)
offset = np.arange(anode_width_in_pads) - center_i
assert offset[center_i] == 0
pad_xmin = (offset[:, None] + 0 * offset[None, :]) * strexo.ANODE_PITCH
pad_ymin = (0 * offset[:, None] + offset[None, :]) * strexo.ANODE_PITCH
assert pad_xmin[center_i, center_i] == pad_ymin[center_i, center_i] == 0
assert np.all(np.isclose(np.diff(pad_ymin[0]), strexo.ANODE_PITCH))
assert np.all(np.diff(pad_ymin[0]), 0)
pad_xmax = pad_xmin + strexo.ANODE_PITCH
pad_ymax = pad_ymin + strexo.ANODE_PITCH

# Compute (xmin, xmax, ymin, ymax) bounding box per channel
# Useful for plotting
# Very inneficient code.. fortunately just runs once
channel_box = dict()
n_channels = pad_channel.max() + 1
for ch in range(n_channels):
    mask = pad_channel == ch
    if not mask.sum():
        continue
    channel_box[ch] = q = dict(
        xmin=pad_xmin[mask].min(),
        xmax=pad_xmax[mask].max(),
        ymin=pad_ymin[mask].min(),
        ymax=pad_ymax[mask].max(),
    )


@export
def channel_rectangle(channel):
    """Return dict with coordinates of rectangle bounding pads read in channel
    """
    return channel_box[channel]


# Check channel_box and pos_to_channel agree with each other
for ch, pos in channel_box.items():
    assert ch == pos_to_channel(pos["xmin"] + 0.1, pos["ymin"] + 0.1)
