import numpy as np
import uproot
import awkward as ak

import strax

import strexo
from strexo import units

export, __all__ = strax.exporter()
__all__ += ["NEST_INTERACTION_CODE"]

NEST_INTERACTION_CODE = dict(ion=6, gammaray=7, beta=8)

##
# ROOT branches / aliases to load
# Constructed so we get strexo.nest_hits_dtype (from common.py)
# after converting from awkward to numpy
##
_prefix = "NESTHit"
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
    # These will be set properly later; we're just making sure these columns
    # exist for now. Alternatively, we could just read the columns above
    # and copy their data into a new array.
    parent_source=_prefix + "Type",
    parent_time=_prefix + "T",
)
branch_types = dict(
    # type of branches not listed here is float64
    interaction_type=np.uint8,
    n_electrons=np.int64,
    n_photons=np.int64,
    time=np.int64,
    endtime=np.int64,
    parent_time=np.int64,
    parent_source=np.uint32,
)


@export
@strax.takes_config(
    strax.Option(
        "mc_sources", help="sequence of dicts with at least file_path and rate"
    ),
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
    TODO: the decay generation could be a separate (earlier) plugin,
        rather than making this a multi-output plugin.
    """

    provides = ("nest_hits", "mc_decays")
    depends_on = tuple()

    data_kind = dict(nest_hits="nest_hits", mc_decays="mc_decays")
    dtype = dict(nest_hits=strexo.dtypes.nest_hits, mc_decays=strexo.dtypes.mc_decays)

    simulation_complete = False

    # list of iterators over ROOT events from a file
    source_iterators: dict

    # list of source rates
    rates: dict

    def setup(self):
        c = self.config
        self.run_end = c["run_duration"] + c["run_start"]
        self.rates = [source["rate"] for source in c["mc_sources"]]
        # Open iterators for all the ROOT files
        n_sources = len(c["mc_sources"])
        self.source_iterators = [None] * n_sources
        for source_index in range(n_sources):
            self._restart_file_read(source_index)

    def source_finished(self):
        return self.simulation_complete

    def is_ready(self, chunk_i):
        c = self.config
        chunk_start = c["run_start"] + c["nest_chunk_duration"] * chunk_i
        return chunk_start < self.run_end

    def compute(self, chunk_i):
        c = self.config

        # Compute time interval to simulate decays in,
        # and check if this is the last chunk.
        chunk_start = c["run_start"] + c["nest_chunk_duration"] * chunk_i
        chunk_end = chunk_start + c["nest_chunk_duration"]
        if chunk_end >= self.run_end:
            chunk_end = self.run_end
            self.simulation_complete = True
        chunk_dt = chunk_end - chunk_start
        assert chunk_start <= chunk_end

        # For each source, generate decays and grab NEST hits
        hit_arrays = []  # list with arrays of hits, one per decay
        decay_info_dicts = []  # list of dicts with summary info for each decay
        for source_index, rate in enumerate(self.rates):
            if chunk_dt == 0:
                continue

            # Draw decay count and times randomly
            # TODO: allow passing a random seed as an option for reproducibility?
            n = np.random.poisson(rate * chunk_dt)
            ts = chunk_start + chunk_dt * np.random.rand(n)
            ts = np.round(ts).astype(np.int64)

            # Collect NEST hits from the ROOT file
            source_it = self.source_iterators[source_index]
            for t in ts:
                try:
                    hits = next(source_it)
                except StopIteration:
                    # File exhausted, restart from the beginning
                    source_it = self._restart_file_read(source_index)
                    hits = next(source_it)
                hits = flatten_awkward_records(hits, branch_types)
                # Hits should already have the right dtype, except for the
                # titles/comments of the fields. Hence we still need:
                hits = hits.astype(self.dtype_for("nest_hits"))

                hits["time"] += t
                hits["endtime"] += t
                hits["parent_time"] = t
                hits["source_index"] = source_index

                hit_arrays.append(hits)

                # Store summary info about the decay
                decay_info_dicts.append(
                    dict(
                        time=t,
                        endtime=t + 1,
                        source_index=source_index,
                        energy=hits["energy"].sum(),
                    )
                )

        # Convert decay_info_dics to a single record array
        # Ineffecient code here is fine; this data is extremely lightweight
        # TODO: this could be a useful utility function, like strax.dict_to_rec
        mc_decays = np.zeros(len(decay_info_dicts), dtype=self.dtype_for("mc_decays"))
        for field_name in mc_decays.dtype.names:
            mc_decays[field_name] = [q[field_name] for q in decay_info_dicts]

        return self.multiple_simulation_results(
            start=chunk_start, end=chunk_end, nest_hits=hit_arrays, mc_decays=mc_decays
        )

    def _restart_file_read(self, source_index):
        """(Re)acquires and returns uproot iterator over events for source_index"""
        self.source_iterators[source_index] = it = uproot.iterate(
            self.config["mc_sources"][source_index]["file_path"],
            branches.keys(),
            aliases=branches,
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
