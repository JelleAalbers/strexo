import numpy as np

import strax
import strexo
from strexo import units

__all__ = ["PadWaveforms"]


@strax.takes_config(
    strax.Option(
        "electron_vdrift",
        default=0.171 * units.cm / units.us,
        help="Electron drift speed [cm/ns]",
    ),
    strax.Option(
        "electron_transverse_diffusion",
        default=100 * units.cm ** 2 / units.s,
        help="Electron transverse diffusion coefficient [cm^2/ns]",
    ),
    strax.Option(
        "electron_lifetime",
        default=1 * units.ms,
        help="Electron lifetime in the liquid xenon [ns]",
    ),
    strax.Option(
        "electron_longitudinal_diffusion",
        default=10 * units.cm ** 2 / units.s,
        help="Electron longitudinal diffusion coefficient [cm^2/ns]",
    ),
    strax.Option(
        "active_pad_grid_width",
        default=5,
        help="Width of the grid of pads for which to simulate waveforms."
        " Should be odd, to ensure there is a center pad.",
    ),
    strax.Option(
        "induction_waveform_databank",
        default=None,
        help="Path to the induction waveform template file to use",
    ),
    strax.Option(
        "induction_waveform_n_samples",
        default=1000,
        help="Length in samples of the induction waveform templates.",
        # Can't be part of induction_waveform_databank; that's loaded in setup
        # but this sets the dtype
    ),
)
class PadWaveforms(strexo.SimulationPlugin):
    """Generates noise-free charge collection waveforms
    for NEST hits in each pad.

    We only simulate the 5x5 grid of pads around the center of the collected
    electron cloud.

    Charge collection is simulated fully (at a single-electron level),
    assuming independent Gaussian diffusion in three dimensions.

    Induction waveforms from each pad are drawn from a template bank,
    indexed by (r, theta, sigma_r, sigma_z).
        Here (r, theta) is the position of the electron cloud center in polar
        coordinates with origin on the pad center,
        and (sigma_r, sigma_z) are the cloud's transverse and longitudinal
        diffusion standard deviations.

    Note the induction waveforms won't be completely realistic:
        (a) the templates assume each cloud comes in from deep in the TPC, and
        (b) the induction is not varied with the stochastic charge distribution
            in the cloud.
    (a) could be mitigated by an additional template index/dimension;
    (b) cannot be mitigated easily.
    (But even the full simulation in arXiv:1907.07512 cuts some corners, e.g.
     binning in 0.3x0.3cm voxels, drifting the cloud before diffusion, etc.)
    """

    depends_on = "nest_hits"
    provides = "pad_waveforms"
    data_kind = "pad_waveforms"

    # Time (in ns) between the induction waveform template start
    # and the time at which the electron cloud center is collected.
    template_rise_time: float

    def infer_dtype(self):
        c = self.config
        return strax.time_dt_fields + [
            (("Pad index", "pad"), np.int32),
            (
                ("Current waveform [electrons/sample]", "data"),
                np.float64,
                c["induction_waveform_n_samples"],
            ),
        ]

    def setup(self):
        c = self.config
        if c["induction_waveform_databank"] is None:
            # Use placeholder induction waveforms (sine waves)
            self.template_rise_time = (
                strexo.CHARGE_READOUT_DT * c["induction_waveform_n_samples"] / 2
            )
        else:
            # Load templates from file; should also define where
            # the template_center is.
            # Also check induction_waveform_n_samples is set to correct value.
            raise NotImplementedError(
                "Induction waveform template files not implemented"
            )

    def compute(self, nest_hits, start, end):
        c = self.config
        hits = nest_hits
        apgw = c["active_pad_grid_width"]

        # Compute size, collection time and position of electron cloud
        drift_length = strexo.ANODE_Z - hits["z"]
        drift_time = drift_length / c["electron_vdrift"]
        p_survival = np.exp(-drift_time / c["electron_lifetime"])
        n_electrons = np.random.binomial(n=hits["n_electrons"], p=p_survival)
        # If the drift field has imperfections, apply them here
        x_center = hits["x"]
        y_center = hits["y"]

        # Diffusion standard deviation in r, t, and z
        sigma_r = np.sqrt(2 * drift_time * c["electron_transverse_diffusion"])
        sigma_z = np.sqrt(2 * drift_time * c["electron_longitudinal_diffusion"])
        sigma_t = sigma_z / c["electron_vdrift"]

        # Index of the central pad (pad that would collect the cloud center),
        # and offset of the cloud center in the central pad
        i_central, dx, dy = self.pad_and_offset(x_center, y_center)

        # Draw induction waveform templates for the active pads.
        # results is a (n_hits, apgw, apgw) array (for now)
        results = self.induction_waveforms(n_electrons, dx, dy, sigma_z, sigma_r)

        # Set indices of the active pad grid, (n_hits, apgw, apgw) array
        results["pad"] = self.pads_around(i_central, n=apgw)

        # Set results['time'] to the start time of the first sample
        # TODO: think about round/int, and/or make template depend on t%dt
        _dt = strexo.CHARGE_READOUT_DT
        t_start = hits["time"] + drift_time - self.template_rise_time
        t_start = (np.round(t_start / _dt) * _dt).astype(np.int64)
        results["time"] = t_start[:, None, None] * np.ones(
            (1, apgw, apgw), dtype=np.int64
        )

        # Simulate collection spikes on top of the induction waveforms
        self.add_collection_spikes(results, n_electrons, dx, dy, sigma_t, sigma_r)

        # Flatten results (from 5x5 pad grid -> 25 waveforms) and send them out
        results = results.ravel()
        return self.simulation_results(results, start=start, end=end)

    def induction_waveforms(self, n_electrons, dx, dy, sigma_z, sigma_r):
        """Return (n_hits, apgw, apgw) array of induction waveforms
        generated by one hit

        Arguments:
         - n_electrons: (n_hits) array of electrons/hit in the entire cloud
         - dx, dy: (n_hits) array of cloud center offsets wrt to the central pad
         - sigma_z, sigma_r: (n_hits) array of longitudinal and transverse
            diffusion 1 sigmas.

        Only data, length, and dt fields will be set; caller is responsible for
        pad and time.
        """
        c = self.config
        n_hits = len(n_electrons)
        assert all([q.shape == n_electrons.shape for q in (dx, dy, sigma_z, sigma_r)])

        apgw = c["active_pad_grid_width"]
        result = np.zeros((n_hits, apgw, apgw), dtype=self.dtype_for("pad_waveforms"))
        result["length"] = c["induction_waveform_n_samples"]
        result["dt"] = strexo.CHARGE_READOUT_DT

        if self.config["induction_waveform_databank"] is None:
            # Use a silly sine wave placeholder in all pads
            sine_wave = 1e-2 * np.sin(
                np.linspace(0, 2 * np.pi, c["induction_waveform_n_samples"])
            )
            # Result has to be (n_hits, apgw, apgw, samples)
            result["data"] = (
                # n_electrons is (n_hits)
                n_electrons[:, None, None, None]
                # sine_wave is (n_samples)
                * sine_wave[None, None, None, :]
            )
        else:
            raise NotImplementedError(
                "Load an appropriate waveform from the databank, "
                "depending on (dx, dy, sigma_z, sigma_r)"
            )

        return result

    def add_collection_spikes(self, results, n_electrons, dx, dy, sigma_t, sigma_r):
        """Add charge collection spikes to waveforms in results.
        Modifies results in-place.

        Arguments:
            results: waveforms of hits, as produced by induction_waveforms
                shape is (n_hits, apgw, apgw), centered on the time the cloud's
                center is collected.
         - n_electrons: (n_hits) array of electrons/hit in the entire cloud
         - dx, dy: (n_hits) array of cloud center offsets wrt to the central pad
         - sigma_r: (n_hits) array of transverse diffusion 1 sigmas.
         - sigma_t: (n_hits) array of drift time variation 1 sigmas.
        """
        apgw = self.config["active_pad_grid_width"]

        for hit_i in range(len(n_electrons)):
            # For each collected electron, draw a collection position wrt the
            # cloud center's collection position...
            x, y = np.random.rand(2, n_electrons[hit_i]) * sigma_r[hit_i]

            # ... then calculate by which pad (index into the active pad grid)
            # each electron is collected. Helpful plot:
            #   x = np.linspace(-apgw/2, apgw/2, 100)
            #   plt.plot(x, (0.5 + x) // 1 + apgw//2)
            xi = (0.5 + (x + dx[hit_i]) / strexo.ANODE_PITCH).astype(int) + apgw // 2
            yi = (0.5 + (y + dy[hit_i]) / strexo.ANODE_PITCH).astype(int) + apgw // 2

            # For each electron, draw a collection time wrt to the
            # cloud center's collection time...
            t = np.random.rand(n_electrons[hit_i]) * sigma_t[hit_i]
            # ... then calculate at which sample index in the waveform
            # the electron is collected.
            # (Note that the center collection time is template_rise_time
            #  far away in the waveform. TODO: well, +- 10 ns...)
            ti = ((t + self.template_rise_time) // strexo.CHARGE_READOUT_DT).astype(int)

            # Add electrons to the waveform. Hurray for numpy magic
            # See https://stackoverflow.com/questions/2004364
            # TODO: you get IndexError if the induction templates are too short
            # to accomodate electrons that diffuse a lot.
            np.add.at(results["data"], (hit_i, xi, yi, ti), 1)

        return results

    def pad_and_offset(self, x, y):
        """Return pad numbers and offsets corresponding to positions

        Arguments:
         - x: array-like, x-coordinates in TPC frame (0 at center)
         - y: array-like, y-coordinates in TPC frame (0 at center)

        Returns: (pad_index, dx, dy)
         - pad_number: array-like of ints, pad numbers
         - dx: array-like, x-coordinates in pad frame (0 at center)
         - dy: array-like, x-coordinates in pad frame (0 at cemter)

        For now, pad numbers are x // anode_pitch + 1000 * y // anode_pitch.
            Assuming the TPC center is right above a cross of pads, pad 0 has
            positive x and y coordinates and the TPC center at its lower left.
        """
        ap = strexo.ANODE_PITCH
        pad_number = (x // ap + 1000 * y // ap).astype(np.int32)
        return pad_number, x % ap, y % ap

    def pads_around(self, central_pad_number, n):
        """Return (m, n, n) array of ints of pad numbers, for nxn pad grid
        centered on pad_index. Here m = len(pad_index).
        """
        # See pads_and_offset for pad number definition
        grid = np.arange(-n // 2 + 1, n // 2 + 1, dtype=np.int32)
        assert len(grid) == n
        return (
            central_pad_number[:, None, None]
            + grid[None, :, None,]
            + 1000 * grid[None, None, :]
        )
