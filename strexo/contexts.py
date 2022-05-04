import strax

import strexo
from strexo import units


def demo(output_folder="./strax_data", **kwargs):
    """Demonstration / test context"""
    # strexo.create_test_data()
    config = dict(
        mc_sources=dict(
            test_source=('./from_jason/test_tl208_strax.root', 1 * units.Hz)),
        run_start=0)
    config = {**config, **kwargs}

    return strax.Context(
        storage=[strax.DataDirectory('./strax_data')],
        register=[strexo.ReadNESTROOT, strexo.PadWaveforms],
        config=config
    )
