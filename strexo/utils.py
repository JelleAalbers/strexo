from re import L
import numba
import numpy as np

import strax

export, __all__ = strax.exporter()


# This base class should probably move to strax itself, once it is tested
@export
class SimulationPlugin(strax.Plugin):
    """Plugin that creates data that extends into the future,
    possibly outside the initial chunk boundaries.

    To use, end compute with
      return self.simulation_results(results, start, end)
    instead of
      return results

    For multi-output plugins, end compute with
        return self.multi_simulation_results(start, end, dtype_1=results_1, ...)

    Unlike OverlapWindowPlugin, data extending beyond the final endtime of the
    inputs / source simulation is discarded. It would be unrealistic,
    containing only tails of earlier events, and no new events.
    """

    # dtype_name -> endtime of last chunk sent out
    last_endtime: dict

    # dtype_name -> Array (not strax chunk!) of results saved for future
    future_results: dict

    def __init__(self, *args, **kwargs):
        # Attribute names are different than OverlapWindowPlugin, since
        # a plugin might try to inherit from both...
        # ... though I doubt that would actually work / make sense!
        self.last_endtime = dict()
        self.future_results = dict()
        super().__init__(*args, **kwargs)

    def multiple_simulation_results(self, start, end, **results_per_dtype):
        """Return dict of strax.Chunks of plugin results with simulation_results,
        for each of the data types and results passed as keyword arguments.
        """
        result = dict()
        for data_type, results_per_type in results_per_dtype.items():
            if isinstance(results_per_type, np.ndarray):
                # One array of results
                y = self.simulation_results(
                    results_per_type, start=start, end=end, data_type=data_type
                )
            else:
                # Sequence of arays to be concatenated
                y = self.simulation_results(
                    *results_per_type, start=start, end=end, data_type=data_type
                )
            result[data_type] = y
        return result

    def simulation_results(self, *results, start, end, data_type=None):
        """Return strax.Chunk of plugin results, given data generated from times
        between start and end.

        Arguments:
         - results: array with simulation results, or sequence of arrays to be
            concatenated and sorted. Nones are automatically removed.
         - start: start of time range from which results were generated
         - end: end of time range from which results were generated
         - data_type: name of results's data type, for multi-output plugins.
            Can be omitted for plugins with a single output.

        Side effects: updates the following Plugin attributes:
          future_result[dtype_name] to array of results for future chunks
          last_endtime[dtype_name] to endtime of the returned chunk

        Notes:
         - Data reaching beyond end is saved in plugin attributes;
            data saved in earlier calls to simulation_results is added.
         - If some data straddles end, the returned chunk's end is decreased
            to a time without straddling data.
            The next chunk will reflect this, i.e. it starts earlier and
            may contain more data.
        """
        if data_type is None:
            assert (
                not self.multi_output
            ), "give dtype_name to simulation_results of a multi-output plugin"
            data_type = self.provides[0]

        last_endtime = self.last_endtime.get(data_type)
        if last_endtime is not None:
            # The endtime of the chunk we last sent is this chunk's start
            # (Note the user-passed start is discarded)
            start = last_endtime

        # Grab current and future results into a list of arrays
        # Note results is a tuple of arrays, thanks to * in function definition
        # (so list() here doesn't convert any data to python types,
        #  just tuple->list)
        results = [self.future_results.get(data_type)] + list(results)
        results = [x for x in results if x is not None and len(x)]

        if not results:
            # No results and nothing saved -> nothing to return
            self.last_endtime = end
            return self.chunk(start=start, end=end, data=self.empty_result())

        # Split around t=end, possibly earlier
        results, self.future_results[data_type], new_endtime = combine_and_break(
            *results, endtime=end, allow_early_split=True
        )

        # Send out the next valid chunk and update last_endtime
        self.last_endtime[data_type] = new_endtime
        return self.chunk(
            start=start, end=new_endtime, data=results, data_type=data_type
        )


@export
def combine_and_break(*args, endtime, allow_early_split=False):
    """Concatenates args, then returns (in_chunk, for_future, endtime)
    * in_chunk data ends at <= endtime
    * for_future starts >= endtime

    if allow_early_split is True, endtime may lay before endtime_target.
    Use this for data with non-unit duration
    """
    x = np.concatenate(args)
    x = sort_by_time_only(x)
    return strax.split_array(x, endtime, allow_early_split=allow_early_split)
    # If data has 1ns duration, this might be an alternative:
    # in_future = strax.endtime(x) > endtime
    # return strax.sort_by_time(x[~in_future]), x[in_future]


@export
@numba.njit(nogil=True, cache=True)
def sort_by_time_only(x):
    """Sort a record array x by time

    Unlike strax.sort_by_time, doesn't sort by channel, so
    it works even on data that doens't have a channel column
    or that has more than 10k channels.
    """
    # (5-10x) faster than np.sort(order=...), as np.sort looks at all fields
    return x[np.argsort(x["time"])]
