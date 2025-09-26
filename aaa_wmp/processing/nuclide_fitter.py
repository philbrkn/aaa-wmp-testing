# aaa_wmp/processing/nuclide_fitter.py

import numpy as np
from scipy.signal import find_peaks

from ..constants import K_BOLTZMANN, TEMPERATURE_LIMIT
from ..io.njoy_interface import NJOYProcessor
from ..io.output_writer import OutputWriter
from ..processing.piece_fitting import fit_piece

# from .pole_cleanup import spurious_cleanup


class NuclideFitter:
    """Handles the complete nuclide fitting pipeline."""

    def __init__(self, config=None):
        self.config = config or {}
        self.njoy_processor = NJOYProcessor(log=self.config["log"])
        self.output_writer = OutputWriter()

    def fit_nuclide(self, endf_file, name, **kwargs):
        """Main entry point for fitting - current fit_nuclide logic"""
        # Extract cross sections
        data = self._prepare_data(endf_file, name, **kwargs)

        # Perform fitting
        results = self._perform_fitting(data, **kwargs)

        # Write output
        output_path = self.output_writer.write_output(results, data, name, **kwargs)

        return {"data": data, "results": results, "output_path": output_path}

    def _prepare_data(self, endf_file, name, **kwargs):
        log = kwargs.get("log", False)
        njoy_error = kwargs.get("njoy_error", 5e-4)
        njoy_input = kwargs.get("njoy_input", None)
        bounds = kwargs.get("bounds", None)

        # Get NJOY data
        nuc_ce = self.njoy_processor.get_point_wise_xs(
            endf_file, name, njoy_error, njoy_input
        )

        if log:
            print("Parsing cross sections within resolved resonance range...")

        # Determine energy bounds
        E_max_idx = self.njoy_processor.determine_energy_bounds(nuc_ce, endf_file)

        # Extract cross sections
        xs_data = self.njoy_processor.extract_cross_sections(nuc_ce, E_max_idx, bounds)

        # Add nuclide metadata
        xs_data["nuc_ce"] = nuc_ce
        xs_data["name"] = name
        xs_data["AWR"] = nuc_ce.atomic_weight_ratio

        # Calculate derived quantities for fitting
        i0 = np.searchsorted(xs_data["energy"], xs_data["E_min"], side="left")
        i1 = np.searchsorted(xs_data["energy"], xs_data["E_max"], side="right")
        xs_data["bound_pts"] = max(0, i1 - i0)
        xs_data["total_xs_bound"] = xs_data["total_xs"][i0:i1]

        if log:
            self._log_data_summary(xs_data)

        return xs_data

    def _log_data_summary(self, data):
        """Log summary of prepared data."""
        print(f"  MTs: {data['mts']}")
        print(
            f"  Energy range: {data['E_min']:.3e} to {data['E_max']:.3e} eV "
            f"({data['bound_pts']} points)"
        )
        peaks, _ = find_peaks(data["total_xs_bound"])
        print(f"  There are {peaks.size} peaks in this range.")

    def _perform_fitting(self, data, **kwargs):
        """Execute the fitting algorithm."""
        vf_pieces = self._determine_pieces(data, kwargs.get("vf_pieces"))
        space = kwargs.get("space", "E")
        log = kwargs.get("log", False)

        # Calculate piece width
        if space == "sqrt_E":
            piece_width = (np.sqrt(data["E_max"]) - np.sqrt(data["E_min"])) / vf_pieces
        else:
            piece_width = (data["E_max"] - data["E_min"]) / vf_pieces

        # Calculate alpha for Doppler
        alpha = data["AWR"] / (K_BOLTZMANN * TEMPERATURE_LIMIT)

        # Initialize results containers
        results = {
            "poles": [],
            "residues": [],
            "remainder_list": [],
            "energy_indices_list": [],
            "bcf_list": [],
            "poly_info_list": [],
            "err_hist_list": [],
            "vf_pieces": vf_pieces,
            "space": space,
            "piece_data": [],
        }

        # VF piece by piece
        for i_piece in range(vf_pieces):
            if log:
                print(f"Fitting piece {i_piece + 1}/{vf_pieces}...")

            piece_result = self._fit_piece(i_piece, data, piece_width, alpha, **kwargs)

            results["poles"].append(piece_result["poles"])
            results["residues"].append(piece_result["residues"])
            results["remainder_list"].append(piece_result["remainder"])
            results["energy_indices_list"].append(piece_result["energy_indices"])
            results["bcf_list"].append(piece_result["bcf"])
            results["poly_info_list"].append(piece_result["poly_info"])
            results["err_hist_list"].append(piece_result["err_hist"])
            results["piece_data"].append(piece_result)

        # Calculate total poles
        results["total_poles"] = sum(len(p) for p in results["poles"])

        if log:
            print(f"Total number of poles: {results['total_poles']}")

        return results

    def _determine_pieces(self, data, vf_pieces):
        """Determine number of fitting pieces."""
        if vf_pieces is not None:
            return vf_pieces

        # Auto-determine based on complexity
        peaks, _ = find_peaks(data["total_xs_bound"])
        n_peaks = peaks.size
        bound_pts = data["bound_pts"]

        if n_peaks > 200 or bound_pts > 30000 or n_peaks * bound_pts > 100 * 10000:
            return max(5, n_peaks // 10, bound_pts // 2000)
        return 1

    def _fit_piece(self, i_piece, data, piece_width, alpha, space, **kwargs):
        """Fit a single piece of the energy range."""
        return fit_piece(i_piece, data, piece_width, alpha, space, **kwargs)
