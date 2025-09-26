import os
import pickle

import numpy as np

from ..visualization.plotting import plot_miaaa_convergence, plot_reconstruction


class OutputWriter:
    """Handles output writing (mp)"""

    def write_output(self, results, data, name, **kwargs):
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
