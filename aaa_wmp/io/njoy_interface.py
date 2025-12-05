# aaa_wmp/io/njoy_interface.py
"""NJOY data processing and caching with temperature support."""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from openmc.data import IncidentNeutron, ResonanceRange


class NJOYProcessor:
    """Handles NJOY data processing and caching."""

    def __init__(self, cache_dir: str = "./data/input/NJOY_pickles", log: int = 0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = log

    def _get_cache_filename(self, name: str, temperatures: List[float]) -> Path:
        """Generate cache filename based on nuclide and temperatures.

        Convention:
            - 0K only: {name}_NJOY.pickle (backwards compatible)
            - With temps: {name}_NJOY_{T1}K_{T2}K_...pickle
        """
        if temperatures == [0.0]:
            return self.cache_dir / f"{name}_NJOY.pickle"

        temp_str = "_".join(f"{int(t)}K" for t in sorted(temperatures))
        return self.cache_dir / f"{name}_NJOY_{temp_str}.pickle"

    def get_point_wise_xs(
        self,
        endf_file: str,
        name: str,
        njoy_error: float = 5e-4,
        temperatures: Optional[List[float]] = None,
        njoy_input: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> IncidentNeutron:
        """Get point-wise cross sections from NJOY at specified temperatures.
        Parameters
        ----------
        endf_file : str
        Path to ENDF file
        name : str
        Nuclide name (e.g., "U238")
        njoy_error : float
        NJOY linearization tolerance
        temperatures : list of float, optional
        Temperatures in Kelvin. Default is [0.0] for 0K data.
        njoy_input : str, optional
        Path to existing pickle file to load instead of running NJOY
        force_regenerate : bool
        If True, regenerate even if cache exists

        Returns
        -------
        IncidentNeutron
        OpenMC nuclear data object with cross sections at requested temps
        """
        if temperatures is None:
            temperatures = [0.0]
        # Sort temperatures for consistent cache naming
        temperatures = sorted(temperatures)

        # Check for explicit input file
        if njoy_input and Path(njoy_input).exists():
            if self.log:
                print(f"Loading cached NJOY data from {njoy_input}")
            with open(njoy_input, "rb") as f:
                return pickle.load(f)

        # Check cache
        cache_file = self._get_cache_filename(name, temperatures)
        if cache_file.exists() and not force_regenerate:
            if self.log:
                print(f"Loading cached NJOY data from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Run NJOY
        if self.log:
            temp_str = ", ".join(f"{t}K" for t in temperatures)
            print(f"Running NJOY for {name} at [{temp_str}] (error={njoy_error})...")

        # For 0K, disable broadening; otherwise enable it
        if temperatures == [0.0]:
            nuc_ce = IncidentNeutron.from_njoy(
                endf_file,
                temperatures=temperatures,
                error=njoy_error,
                broadr=False,
                heatr=False,
                purr=False,
            )
        else:
            # For finite temperatures, we need broadr
            # Include 0K if not present (needed for comparison)
            temps_with_0k = (
                temperatures if 0.0 in temperatures else [0.0] + temperatures
            )
            nuc_ce = IncidentNeutron.from_njoy(
                endf_file,
                temperatures=temps_with_0k,
                error=njoy_error,
                broadr=True,
                heatr=False,
                purr=False,
            )

        # Cache result
        with open(cache_file, "wb") as f:
            pickle.dump(nuc_ce, f)
        if self.log:
            print(f"Cached NJOY data to {cache_file}")

        return nuc_ce

    def determine_energy_bounds(self, nuc_ce: IncidentNeutron, endf_file: str) -> int:
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

    def extract_cross_sections(
        self,
        nuc_ce: IncidentNeutron,
        E_max_idx: int,
        bounds: Optional[dict] = None,
        temperature: float = 0.0,
    ) -> dict:
        """Extract cross sections within energy bounds at a specific temperature.

        Parameters
        ----------
        nuc_ce : IncidentNeutron
            OpenMC nuclear data object
        E_max_idx : int
            Index of maximum energy
        bounds : dict, optional
            Dictionary with 'E_min' and/or 'E_max' keys
        temperature : float
            Temperature in Kelvin to extract (default 0.0)

        Returns
        -------
        dict
            Dictionary containing energy grid and cross sections
        """
        # Temperature key for OpenMC data
        temp_key = f"{int(temperature)}K"

        # Fall back to 0K if requested temp not available
        if temp_key not in nuc_ce.energy:
            if self.log:
                print(f"Warning: {temp_key} not available, using 0K")
            temp_key = "0K"

        energy = nuc_ce.energy[temp_key][: E_max_idx + 1]

        # Apply user-specified bounds if provided
        if bounds:
            E_min = bounds.get("E_min", energy[0])
            E_max = bounds.get("E_max", energy[-1])
        else:
            E_min, E_max = energy[0], energy[-1]

        # Extract cross sections
        total_xs = nuc_ce[1].xs[temp_key](energy)
        elastic_xs = nuc_ce[2].xs[temp_key](energy)

        try:
            absorption_xs = nuc_ce[27].xs[temp_key](energy)
        except KeyError:
            absorption_xs = np.zeros_like(total_xs)

        fissionable = False
        fission_xs = None
        try:
            fission_xs = nuc_ce[18].xs[temp_key](energy)
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
            "temperature": temperature,
        }


# Convenience function for generating temperature reference data
def generate_temperature_references(
    endf_file: str,
    name: str,
    temperatures: List[float],
    cache_dir: str = "./data/input/NJOY_pickles",
    njoy_error: float = 5e-4,
    log: int = 1,
) -> dict:
    """Generate NJOY reference data at multiple temperatures.

    This is a convenience function for Doppler broadening validation.

    Parameters
    ----------
    endf_file : str
        Path to ENDF file
    name : str
        Nuclide name
    temperatures : list of float
        Temperatures in Kelvin (e.g., [293.6, 600, 900, 1200])
    cache_dir : str
        Directory for caching
    njoy_error : float
        NJOY tolerance
    log : int
        Verbosity level

    Returns
    -------
    dict
        Dictionary mapping temperature -> cross section data dict
    """
    processor = NJOYProcessor(cache_dir=cache_dir, log=log)

    # Always include 0K for baseline
    all_temps = [0.0] + [t for t in temperatures if t != 0.0]

    # Generate all temperatures in one NJOY run (more efficient)
    nuc_ce = processor.get_point_wise_xs(
        endf_file=endf_file,
        name=name,
        njoy_error=njoy_error,
        temperatures=all_temps,
    )

    # Determine bounds (use 0K grid as reference)
    E_max_idx = processor.determine_energy_bounds(nuc_ce, endf_file)

    # Extract cross sections at each temperature
    results = {}
    for temp in all_temps:
        results[temp] = processor.extract_cross_sections(
            nuc_ce, E_max_idx, temperature=temp
        )

    # Add metadata
    results["nuc_ce"] = nuc_ce
    results["name"] = name
    results["AWR"] = nuc_ce.atomic_weight_ratio
    results["temperatures"] = all_temps

    return results
