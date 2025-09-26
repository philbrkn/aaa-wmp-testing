# aaa_wmp/io/njoy_interface.py
import pickle
from pathlib import Path

import numpy as np
from openmc.data import IncidentNeutron, ResonanceRange


class NJOYProcessor:
    """Handles NJOY data processing and caching."""

    def __init__(self, cache_dir="./data/NJOY_pickles", log=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = log

    def get_point_wise_xs(self, endf_file, name, njoy_error=5e-4, njoy_input=None):
        """Get 0K point-wise cross sections from NJOY."""
        if njoy_input:
            # Load from existing pickle
            with open(njoy_input, "rb") as f:
                return pickle.load(f)

        if self.log:
            print(f"Running NJOY to get 0K point-wise data (error={njoy_error})...")

        # Run NJOY
        nuc_ce = IncidentNeutron.from_njoy(
            endf_file,
            temperatures=[0.0],
            error=njoy_error,
            broadr=False,
            heatr=False,
            purr=False,
        )

        # Cache result
        cache_file = self.cache_dir / f"{name}_NJOY.pickle"
        with open(cache_file, "wb") as f:
            pickle.dump(nuc_ce, f)

        return nuc_ce

    def determine_energy_bounds(self, nuc_ce, endf_file):
        """Determine appropriate energy bounds from resonance data."""
        # Get ENDF resonance info
        endf_res = IncidentNeutron.from_endf(endf_file).resonances

        # Default to full range
        E_max = nuc_ce.energy["0K"][-1]
        E_max_idx = len(nuc_ce.energy["0K"]) - 1

        # Check resolved resonance range
        if (
            hasattr(endf_res, "resolved")
            and hasattr(endf_res.resolved, "energy_max")
            and type(endf_res.resolved) is not ResonanceRange
        ):
            E_max = endf_res.resolved.energy_max
        elif hasattr(endf_res, "unresolved") and hasattr(
            endf_res.unresolved, "energy_min"
        ):
            E_max = endf_res.unresolved.energy_min

        E_max_idx = np.searchsorted(nuc_ce.energy["0K"], E_max, side="right") - 1

        # Check for threshold reactions
        for mt in nuc_ce.reactions:
            if hasattr(nuc_ce.reactions[mt].xs["0K"], "_threshold_idx"):
                threshold_idx = nuc_ce.reactions[mt].xs["0K"]._threshold_idx
                if 0 < threshold_idx < E_max_idx:
                    E_max_idx = threshold_idx

        return E_max_idx

    def extract_cross_sections(self, nuc_ce, E_max_idx, bounds=None):
        """Extract cross sections within energy bounds."""
        energy = nuc_ce.energy["0K"][: E_max_idx + 1]

        # Apply user-specified bounds if provided
        if bounds:
            E_min = bounds.get("E_min", energy[0])
            E_max = bounds.get("E_max", energy[-1])
        else:
            E_min, E_max = energy[0], energy[-1]

        # Extract cross sections
        total_xs = nuc_ce[1].xs["0K"](energy)
        elastic_xs = nuc_ce[2].xs["0K"](energy)

        try:
            absorption_xs = nuc_ce[27].xs["0K"](energy)
        except KeyError:
            absorption_xs = np.zeros_like(total_xs)

        fissionable = False
        fission_xs = None
        try:
            fission_xs = nuc_ce[18].xs["0K"](energy)
            fissionable = True
        except KeyError:
            pass

        # Create cross section matrix
        if fissionable:
            ce_xs = np.vstack((elastic_xs, absorption_xs, fission_xs))
            mts = [2, 27, 18]
        else:
            ce_xs = np.vstack((elastic_xs, absorption_xs))
            mts = [2, 27]

        return {
            "energy": energy,
            "ce_xs": ce_xs,
            "mts": mts,
            "E_min": E_min,
            "E_max": E_max,
            "total_xs": total_xs,
            "elastic_xs": elastic_xs,
            "absorption_xs": absorption_xs,
            "fission_xs": fission_xs,
            "fissionable": fissionable,
        }
