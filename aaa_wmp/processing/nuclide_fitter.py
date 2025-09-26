# aaa_wmp/processing/nuclide_fitter.py
import os
import pickle

import numpy as np
from scipy.signal import find_peaks

from ..constants import K_BOLTZMANN, TEMPERATURE_LIMIT
from ..core.conversion import proper_rational
from ..core.fitting import evaluate_miaaa, miaaa_xs
from ..core.plotting import (
    plot_aaa_results,
    plot_miaaa_convergence,
    plot_reconstruction,
)

# from ..core.aaa_fitting import miaaa_xs
# from ..core.evaluation import evaluate_miaaa
# from ..core.rational_conversion import proper_rational
from ..io.njoy_interface import NJOYProcessor

# from ..io.wmp_writer import WMPWriter
# from ..visualization.reconstruction_plots import plot_aaa_results
# from .pole_cleanup import spurious_cleanup


class NuclideFitter:
    """Handles the complete nuclide fitting pipeline."""

    def __init__(self, config=None):
        self.config = config or {}
        self.njoy_processor = NJOYProcessor(log=self.config["log"])
        # self.wmp_writer = WMPWriter()

    def fit_nuclide(self, endf_file, name, **kwargs):
        """Main entry point for fitting - current fit_nuclide logic"""
        # Extract cross sections
        data = self._prepare_data(endf_file, name, **kwargs)

        # Perform fitting
        results = self._perform_fitting(data, **kwargs)

        # Write output
        output_path = self._write_output(results, data, name, **kwargs)

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

    def _write_output(self, results, data, name, **kwargs):
        """Write fitting results to file."""
        log = kwargs.get("log", False)
        output_format = kwargs.get("output_format", "wmp")
        path_out = kwargs.get("path_out", "./output")

        # Extract data from results
        poles = results["poles"]
        # residues = results["residues"]
        vf_pieces = results["vf_pieces"]
        space = results["space"]

        # Log total poles
        n_poles = sum(len(p) for p in poles)
        if log:
            print(f"Total number of poles: {n_poles}")

        # Special handling for single-piece fits (detailed plots)
        if vf_pieces == 1 and kwargs.get("plot_single_piece", True):
            self._plot_single_piece_results(results, data, name, path_out, space, log)

        if output_format == "mp_data":
            return self._write_mp_data(results, data, name, path_out, log)
        elif output_format == "wmp":
            return self._write_wmp_format(results, data, name, path_out, log)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

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
        log = kwargs.get("log", False)
        cleanup = kwargs.get("cleanup", False)
        cleanup_tol = kwargs.get("cleanup_tol", 1e-6)
        plot_each_slice = kwargs.get("plot_each_slice", False)
        path_out = kwargs.get("path_out", "./output")

        energy = data["energy"]
        ce_xs = data["ce_xs"]
        fissionable = data["fissionable"]
        E_min = data["E_min"]
        E_max = data["E_max"]
        n_points = len(energy)

        # Calculate piece boundaries
        if space == "sqrt_E":
            sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
            sqrt_E_right = min(np.sqrt(E_max), sqrt_E_left + piece_width)
            E_left = sqrt_E_left**2
            E_right = sqrt_E_right**2

            # Doppler broadening extension
            if i_piece == 0 or np.sqrt(alpha) * sqrt_E_left < 4.0:
                e_start = E_left
            else:
                e_start = max(E_min, (np.sqrt(alpha) * sqrt_E_left - 4.0) ** 2 / alpha)
            e_end = min(E_max, (np.sqrt(alpha) * sqrt_E_right + 4.0) ** 2 / alpha)
        else:  # space == "E"
            E_left = E_min + i_piece * piece_width
            E_right = min(E_max, E_left + piece_width)
            sqrt_E_left = np.sqrt(E_left)
            sqrt_E_right = np.sqrt(E_right)

            if i_piece == 0 or np.sqrt(alpha) * sqrt_E_left < 4.0:
                e_start = E_left
            else:
                e_start = max(E_min, (np.sqrt(alpha) * sqrt_E_left - 4.0) ** 2 / alpha)
            e_end = min(E_max, (np.sqrt(alpha) * sqrt_E_right + 4.0) ** 2 / alpha)

        # Get energy indices
        e_start_idx = max(0, np.searchsorted(energy, e_start, side="right") - 1)
        e_end_idx = min(n_points, np.searchsorted(energy, e_end, side="left") + 1)

        if e_end_idx <= e_start_idx + 1:
            e_start_idx = max(0, e_start_idx - 1)
            e_end_idx = min(n_points, e_end_idx + 1)

        e_idx = range(e_start_idx, e_end_idx)
        E_piece = energy[e_idx]

        # Extract piece data
        sig_s_piece = ce_xs[0, e_idx]
        sig_a_piece = ce_xs[1, e_idx]
        sig_f_piece = ce_xs[2, e_idx] if fissionable else None

        channels = [sig_s_piece, sig_a_piece]
        if fissionable:
            channels.append(sig_f_piece)

        # Perform MIAAA fitting
        w, z, fz, R, err_hist = miaaa_xs(
            E_piece,
            channels,
            method=kwargs.get("method", "full_svd"),
            rtol=kwargs.get("rtol", 1e-13),
            mmax=kwargs.get("mmax", 100),
            greedy_metric="relative",
            log=log,
            space=space,
            normalize=True,
            lawson_iter=kwargs.get("lawson_iter", 0),
        )

        Z = np.sqrt(E_piece) if space == "sqrt_E" else E_piece

        # Optional cleanup
        if cleanup:
            pol, res, _, _ = proper_rational(z, w, w, fz, R, Z)
            z, fz, w = spurious_cleanup(
                pol, res.T, z, fz, w, E_piece, R.T, cleanup_tol=cleanup_tol
            )

        # Extract poles and residues
        if len(w) == 2 * len(z):  # Lawson succeeded
            m = len(z)
            w_num = w[m : 2 * m]
            w_den = w[:m]
            poles_piece, residues_piece, remainder, poly_info = proper_rational(
                z,
                w_num,
                w_den,
                fz,
                R,
                Z,
                pole_extraction=kwargs.get("pole_extraction", None),
                max_poly_degree=kwargs.get("max_poly_degree", 0),
            )
        else:  # No Lawson
            poles_piece, residues_piece, remainder, poly_info = proper_rational(
                z,
                w,
                w,
                fz,
                R,
                Z,
                pole_extraction=kwargs.get("pole_extraction", None),
                max_poly_degree=kwargs.get("max_poly_degree", 0),
            )

        if log:
            print(f"    Piece {i_piece + 1}: {len(poles_piece)} poles")

        # Optional plotting
        if plot_each_slice:
            self._plot_piece(
                E_piece, Z, channels, w, z, fz, fissionable, space, path_out, i_piece
            )

        return {
            "poles": poles_piece,
            "residues": residues_piece,
            "remainder": remainder,
            "poly_info": poly_info,
            "energy_indices": [e_start_idx, e_end_idx],
            "bcf": R,
            "err_hist": err_hist,
            "E_piece": E_piece,
            "channels": channels,
        }

    def _plot_piece(
        self, E_piece, Z, channels, w, z, fz, fissionable, space, path_out, i_piece
    ):
        """Plot fitting results for a single piece."""

        # Extract channel data
        sig_s_piece = channels[0]
        sig_a_piece = channels[1]
        sig_f_piece = channels[2] if fissionable else None

        # Handle Lawson weights
        if len(w) == 2 * len(z):  # Lawson succeeded
            m = len(z)
            w_num = w[m : 2 * m]
            w_den = w[:m]
            R_pieces = evaluate_miaaa(
                Z, w, z, fz, space=space, w_den=w_den, w_num=w_num
            )
        else:  # No Lawson
            R_pieces = evaluate_miaaa(Z, w, z, fz, space=space)

        # Extract reconstructed data
        R_s = R_pieces[0]
        R_a = R_pieces[1]
        R_f = R_pieces[2] if fissionable else None

        # Create plot directory for this piece
        piece_plot_dir = os.path.join(path_out, "piece_plots")
        os.makedirs(piece_plot_dir, exist_ok=True)

        # Call the plotting function
        plot_aaa_results(
            Z,
            sig_s_piece,
            sig_a_piece,
            R_s,
            R_a,
            sigma_f=sig_f_piece,
            R_f=R_f,
            path_out=piece_plot_dir,
            title_prefix=f"Piece {i_piece + 1}",
        )

    def _write_mp_data(self, results, data, name, path_out, log):
        """Write multipole data in pickle format."""
        mp_data = {
            "name": name,
            "AWR": data["AWR"],
            "E_min": data["E_min"],
            "E_max": data["E_max"],
            "poles": results["poles"],
            "residues": results["residues"],
        }

        mp_path_out = os.path.join(path_out, "mp_data")
        os.makedirs(mp_path_out, exist_ok=True)
        mp_filename = os.path.join(mp_path_out, f"{name}_mp.pickle")

        with open(mp_filename, "wb") as f:
            pickle.dump(mp_data, f)

        if log:
            print(f"Dumped multipole data to file: {mp_filename}")

        return mp_filename

    def _plot_single_piece_results(self, results, data, name, path_out, space, log):
        """Generate detailed plots for single-piece fits."""
        # Get piece data
        piece_data = results["piece_data"][0]
        E_piece = piece_data["E_piece"]
        channels = piece_data["channels"]
        poles = results["poles"][0]
        residues = results["residues"][0]
        poly_info = results["poly_info_list"][0]
        err_hist = results["err_hist_list"][0]

        # Prepare channel data
        channels_data = {
            "elastic": channels[0],
            "absorption": channels[1],
            "fission": channels[2] if data["fissionable"] else None,
        }

        # Print pole details if requested
        if log:
            for p in np.sort(np.sqrt(poles)):
                print(f"pole real: {p.real:.2e} | imag: {p.imag:.2e}")

        # Create plots directory
        plot_path = os.path.join(path_out, "plots")
        os.makedirs(plot_path, exist_ok=True)

        # Relative error plot
        plot_reconstruction(
            E_piece,
            channels_data,
            poles,
            residues,
            name=name,
            path_out=plot_path,
            plot_type="loglog",
            show_error=True,
            error_type="relative",
            poly_info=poly_info,
            fit_space=space,
        )

        # Absolute error plot
        plot_reconstruction(
            E_piece,
            channels_data,
            poles,
            residues,
            name=name,
            path_out=plot_path,
            plot_type="loglog",
            show_error=True,
            error_type="absolute",
            poly_info=poly_info,
            fit_space=space,
        )

        # Convergence plot
        plot_miaaa_convergence(err_hist, path_out=plot_path)

    def _write_wmp_format(self, results, data, name, path_out, log):
        """Write WMP format HDF5 file."""
        import h5py

        from ..constants import WMP_VERSION

        poles = results["poles"]
        residues = results["residues"]
        vf_pieces = results["vf_pieces"]
        space = results["space"]
        fissionable = data["fissionable"]

        # Stack poles and residues in WMP format
        wmp_data_pieces = []
        for ip in range(vf_pieces):
            if len(poles[ip]) > 0:
                if fissionable:
                    piece_data = np.column_stack(
                        [
                            poles[ip],
                            residues[ip][0],  # elastic
                            residues[ip][1],  # absorption
                            residues[ip][2],  # fission
                        ]
                    )
                else:
                    piece_data = np.column_stack(
                        [
                            poles[ip],
                            residues[ip][0],  # elastic
                            residues[ip][1],  # absorption
                        ]
                    )
                wmp_data_pieces.append(piece_data)

        # Concatenate all pieces
        wmp_data_array = np.vstack(wmp_data_pieces) if wmp_data_pieces else np.array([])

        # Build windows array
        windows = []
        pole_count = 0
        for iw in range(vf_pieces):
            n_poles_piece = len(poles[iw]) if iw < len(poles) else 0
            if n_poles_piece > 0:
                windows.append([pole_count + 1, pole_count + n_poles_piece])
                pole_count += n_poles_piece
            else:
                windows.append([pole_count + 1, pole_count])

        # Calculate spacing
        spacing = (np.sqrt(data["E_max"]) - np.sqrt(data["E_min"])) / vf_pieces

        # Prepare output path
        wmp_path_out = os.path.join(path_out, "wmp_files")
        os.makedirs(wmp_path_out, exist_ok=True)
        filename = os.path.join(wmp_path_out, f"{name}_wmp.h5")

        # Write HDF5 file
        with h5py.File(filename, "w", libver="earliest") as f:
            f.attrs["filetype"] = np.bytes_("data_wmp")
            f.attrs["version"] = np.array(WMP_VERSION)

            g = f.create_group(name)

            # Write scalars
            g.create_dataset("version", data=np.array(WMP_VERSION))
            g.create_dataset("spacing", data=np.array(spacing))
            g.create_dataset("sqrtAWR", data=np.array(np.sqrt(data["AWR"])))
            g.create_dataset("E_min", data=np.array(data["E_min"]))
            g.create_dataset("E_max", data=np.array(data["E_max"]))

            # Write arrays
            g.create_dataset("data", data=wmp_data_array)
            g.create_dataset("windows", data=np.array(windows))
            g.create_dataset("broaden_poly", data=np.ones(vf_pieces, dtype=np.int8))
            g.create_dataset("curvefit", data=[])  # Empty for now

            # Write remainder data
            remainder_group = g.create_group("remainder_data")
            for i, remainder in enumerate(results["remainder_list"]):
                if remainder is not None:
                    remainder_group.create_dataset(f"window_{i}", data=remainder)

            # Write energy indices
            g.create_dataset(
                "energy_indices", data=np.array(results["energy_indices_list"])
            )
            g.create_dataset("energy_grid", data=data["energy"])

            # Write BCF data if available
            if "bcf_list" in results:
                bcf_group = g.create_group("bcf_data")
                for i, bcf in enumerate(results["bcf_list"]):
                    if bcf is not None:
                        bcf_group.create_dataset(f"window_{i}", data=bcf)

            # Add metadata
            g.attrs["fit_space"] = np.bytes_(space)
            g.attrs["poly_format"] = np.bytes_("polyval")

        if log:
            print(f"Wrote WMP file: {filename}")

        return filename
