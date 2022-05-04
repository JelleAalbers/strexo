from re import L
import numba
import numpy as np

import strax

export, __all__ = strax.exporter()


@export
class SimulationPlugin(strax.Plugin):
    """Plugin that creates data that extends into the future,
    possibly outside the initial chunk boundaries.

    To use, end compute with
      return self.simulation_results(results, start, end)
    instead of
      return results

    Unlike OverlapWindowPlugin, data extending beyond the final endtime of the
    inputs / source simulation is discarded. It would be unrealistic,
    containing only tails of earlier events, and no new events.

    TODO: support multi-output plugins
    """

    # Attribute names are different than OverlapWindowPlugin, since
    # a plugin might try to inherit from both...
    # ... though I doubt that would actually work / make sense!

    # Last endtime of result chunks send out
    last_endtime = None

    # Array (not strax chunk!) of results saved for future
    future_results = None

    def simulation_results(self, *results, start, end):
        """Save new simulation results, then return chunk of data to send.

        Arguments:
         - results: array with simulation results
         - start: start of time range from which results were generated
         - end: end of time range from which results were generated

        Returns: strax Chunk with most of the generated data.
          Data saved from past chunks may be added.
          Data that straddles or lies beyond end is saved for future chunks.
          If some data straddles end, the chunk's end is brought forward
          to a time without straddling data.

        Side effect: sets attributes
          future_results to an array of results to send in future chunks
          last_endtime to the last endtime of
        """
        if self.last_endtime is not None:
            # This isn't the first chunk: we should use the last endtime
            # as the start of new data.
            start = self.last_endtime

        # Grab current and future results into a list of arrays
        results = [self.future_results] + list(results)
        results = [x for x in results if x is not None and len(x)]

        if not results:
            # No results and nothing saved -> nothing to return
            self.last_endtime = end
            return self.chunk(start=start, end=end, data=self.empty_result())

        # Split around t=end, possibly earlier
        results, self.future_results, new_endtime = combine_and_break(
            *results, endtime=end, allow_early_split=True
        )

        # Send out a valid chunk, update the last_endtime sent out
        results = self.chunk(start=start, end=new_endtime, data=results)
        self.last_endtime = new_endtime
        return results


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
    # If data had no duration, this might be an alternative:
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
