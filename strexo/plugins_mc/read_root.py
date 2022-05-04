import warnings

import numpy as np
import uproot
import awkward as ak

import strax

import strexo
from strexo import units

export, __all__ = strax.exporter()

_prefix = "NESTHit"
# ROOT branches / aliases to load
# Constructed so we get strexo.nest_hits_dtype (from common.py)
# after converting from awkward to numpy
branches = dict(
    time=_prefix + "T",
    # strax requires an endtime; just setting it to start + 1 ns.
    endtime=_prefix + "T + 1",
    x=_prefix + "X",
    y=_prefix + "Y",
    z=_prefix + "Z",
    energy=_prefix + "E",
    n_photons=_prefix + "NOP",
    n_electrons=_prefix + "NTE",
    interaction_type=_prefix + "Type",
    # Wwill be set properly later, just making sure the column exists for now
    t_parent=_prefix + "T",
)
branch_types = dict(
    interaction_type=np.uint8,
    n_electrons=np.int64,
    n_photons=np.int64,
    time=np.int64,
    endtime=np.int64,
    t_parent=np.int64,
    # default is float64
)


@export
@strax.takes_config(
    strax.Option("mc_sources", help="dict of source_name -> (file_path, rate_Ghz)"),
    strax.Option(
        "run_duration",
        default=int(60 * units.s),
        type=int,
        help="Duration of the run [ns]",
    ),
    strax.Option(
        "nest_chunk_duration",
        default=int(10 * units.s),
        type=int,
        track=False,
        help="Target duration of initially produced NEST hit chunks [ns]",
    ),
    strax.Option(
        "run_start",
        default=1650932812_000000000,
        type=int,
        help="Start time of the simulated run, since epoch [ns]",
    ),
)
class ReadNESTROOT(strexo.SimulationPlugin):
    """Combine NEST hits from multiple GEANT4-generated ROOT files

    The ROOT files are assumed to contain a simple sequence of events,
    each containing NEST hits with time=0 the time of the primary decay.

    The plugin draws decay times using the provided rates,
    loads NEST hits for each decay, adjusts times appropriately,
    and combines these into a single output stream.

    TODO: apply some kind of micro-clustering to speed up the simulation?
        Or perhaps in a separate plugin that makes nest_hits_clustered
        from nest_hits?
    """

    provides = "nest_hits"
    depends_on = tuple()
    dtype = strexo.nest_hits_dtype

    simulation_complete = False

    # dict source_name -> iterator over ROOT events
    source_iterators = None

    # dict source_name -> rate
    rates = None

    def setup(self):
        c = self.config
        self.run_end = c["run_duration"] + c["run_start"]
        # Open iterators for all the ROOT files
        self.source_iterators = dict()
        self.rates = dict()
        for source_name, (_, rate) in c["mc_sources"].items():
            self._restart_file_read(source_name)
            self.rates[source_name] = rate

    def source_finished(self):
        return self.simulation_complete

    def is_ready(self, chunk_i):
        c = self.config
        chunk_start = c["run_start"] + c["nest_chunk_duration"] * chunk_i
        return chunk_start < self.run_end

    def compute(self, chunk_i):
        c = self.config

        chunk_start = c["run_start"] + c["nest_chunk_duration"] * chunk_i
        chunk_end = chunk_start + c["nest_chunk_duration"]
        if chunk_end >= self.run_end:
            chunk_end = self.run_end
            self.simulation_complete = True
        chunk_dt = chunk_end - chunk_start
        assert chunk_start <= chunk_end

        hit_arrays = []
        for source_name, rate in self.rates.items():
            if chunk_dt == 0:
                continue
            source_it = self.source_iterators[source_name]
            # Draw primary decay count and times
            n = np.random.poisson(rate * chunk_dt)
            ts = chunk_start + chunk_dt * np.random.rand(n)
            ts = np.round(ts).astype(np.int64)
            # Collect hits for each of the decays
            for t in ts:
                try:
                    hits = next(source_it)
                except StopIteration:
                    # File exhausted, restart from the beginning
                    source_it = self._restart_file_read(source_name)
                    hits = next(source_it)
                hits = flatten_awkward_records(hits, branch_types)
                # Hits should already have the right dtype, except for the
                # titles fields
                hits = hits.astype(self.dtype_for("nest_hits"))

                hits["t_parent"] = t
                hits["time"] += t
                hits["endtime"] += t
                hit_arrays.append(hits)

        return self.simulation_results(*hit_arrays, start=chunk_start, end=chunk_end)

    def _restart_file_read(self, source_name):
        """(Re)acquires and returns uproot iterator over events for source_name"""
        self.source_iterators[source_name] = it = uproot.iterate(
            self.config["mc_sources"][source_name][0], branches.keys(), aliases=branches
        )
        return it


@export
def flatten_awkward_records(data, dtypes=None):
    """Return numpy record array from awkward array,
    assuming each field has the same number of elements.

    Args:
      - data: awkward record array, shape (n_events, *var*)
      - dtypes: dict mapping field name to data type. Default is float.
    """
    if dtypes is None:
        dtypes = dict()

    flat_data = ak.flatten(data, None).to_numpy().reshape(len(data.fields), -1)
    return strax.dict_to_rec(
        {
            field_name: flat_data[i].astype(dtypes.get(field_name, float))
            for i, field_name in enumerate(data.fields)
        }
    )
