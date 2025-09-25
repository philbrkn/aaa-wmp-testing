import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from openmc.data import K_BOLTZMANN
from openmc.data.neutron import IncidentNeutron
from openmc.data.resonance import ResonanceRange
from scipy.signal import find_peaks

from .cleanup import spurious_cleanup
from .conversion import proper_rational
from .fitting import evaluate_miaaa, miaaa_xs
from .plotting import (
    plot_aaa_results,
    plot_miaaa_convergence,
    plot_reconstruction,
)

# Constants that determine which value to access
_MP_EA = 0  # Pole

# Residue indices
_MP_RS = 1  # Residue scattering
_MP_RA = 2  # Residue absorption
_MP_RF = 3  # Residue fission

# Polynomial fit indices
_FIT_S = 0  # Scattering
_FIT_A = 1  # Absorption
_FIT_F = 2  # Fission

# Upper temperature limit (K)
TEMPERATURE_LIMIT = 3000

# Logging control
DETAILED_LOGGING = 2


def fit_nuclide(
    endf_file,
    name,
    njoy_error=5e-4,
    vf_pieces=None,
    log=False,
    path_out=None,
    mp_filename=None,
    njoy_input=None,
    bounds=None,
    plot_each_slice=True,
    fit_mask_guard=0,
    cleanup=False,
    cleanup_tol=1e-6,
    output_format=None,
    **kwargs,
):
    r"""Generate multipole data for a nuclide from ENDF.

    Parameters
    ----------
    endf_file : str
        Path to ENDF evaluation
    njoy_error : float, optional
        Fractional error tolerance for processing point-wise data with NJOY
    vf_pieces : integer, optional
        Number of equal-in-momentum spaced energy pieces for data fitting
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to write out mutipole data file and vector fitting figures
    mp_filename : str, optional
        File name to write out multipole data
    **kwargs
        Keyword arguments passed to :func:`openmc.data.multipole._vectfit_xs`

    Returns
    -------
    mp_data
        Dictionary containing necessary multipole data of the nuclide

    """

    wmp_path_out = os.path.join(path_out, "aaa_in_h5wmp_format")
    base = Path(__file__).parent
    njoy_path_out = os.path.join(base, "data/input/NJOY_pickles")
    aaa_plot_loc = os.path.join(path_out, "plots/aaa_bary_plot")
    plot_path = os.path.join(path_out, "plots/reconstruction_plots")
    mp_path_out = os.path.join(path_out, "mp_data_output")
    # ======================================================================
    # PREPARE POINT-WISE XS
    # make 0K ACE data using njoy
    if njoy_input is None:
        if log:
            print(f"Running NJOY to get 0K point-wise data (error={njoy_error})...")

        nuc_ce = IncidentNeutron.from_njoy(
            endf_file,
            temperatures=[0.0],
            error=njoy_error,
            broadr=False,
            heatr=False,
            purr=False,
        )
        # dump the NJOY input for later use
        njoy_path_out.mkdir(parents=True, exist_ok=True)
        with open(njoy_path_out / f"{name}_NJOY.pickle", "wb") as f:
            pickle.dump(nuc_ce, f)
    else:
        # pickle in
        nuc_ce = pickle.load(open(njoy_input, "rb"))

    if log:
        print("Parsing cross sections within resolved resonance range...")

    # Determine upper energy: the lower of RRR upper bound and first threshold
    endf_res = IncidentNeutron.from_endf(endf_file).resonances
    if (
        hasattr(endf_res, "resolved")
        and hasattr(endf_res.resolved, "energy_max")
        and type(endf_res.resolved) is not ResonanceRange
    ):
        E_max = endf_res.resolved.energy_max
    elif hasattr(endf_res, "unresolved") and hasattr(endf_res.unresolved, "energy_min"):
        E_max = endf_res.unresolved.energy_min
    else:
        E_max = nuc_ce.energy["0K"][-1]
    E_max_idx = np.searchsorted(nuc_ce.energy["0K"], E_max, side="right") - 1
    for mt in nuc_ce.reactions:
        if hasattr(nuc_ce.reactions[mt].xs["0K"], "_threshold_idx"):
            threshold_idx = nuc_ce.reactions[mt].xs["0K"]._threshold_idx
            if 0 < threshold_idx < E_max_idx:
                E_max_idx = threshold_idx

    # parse energy and cross sections
    energy = nuc_ce.energy["0K"][: E_max_idx + 1]
    if bounds:
        E_min = bounds["E_min"]
        E_max = bounds["E_max"]
    else:
        E_min, E_max = energy[0], energy[-1]

    n_points = energy.size
    total_xs = nuc_ce[1].xs["0K"](energy)
    elastic_xs = nuc_ce[2].xs["0K"](energy)

    try:
        absorption_xs = nuc_ce[27].xs["0K"](energy)
    except KeyError:
        absorption_xs = np.zeros_like(total_xs)

    fissionable = False
    try:
        fission_xs = nuc_ce[18].xs["0K"](energy)
        fissionable = True
    except KeyError:
        pass

    # make vectors
    if fissionable:
        ce_xs = np.vstack((elastic_xs, absorption_xs, fission_xs))
        mts = [2, 27, 18]
    else:
        ce_xs = np.vstack((elastic_xs, absorption_xs))
        mts = [2, 27]

    i0 = np.searchsorted(energy, E_min, side="left")
    i1 = np.searchsorted(energy, E_max, side="right")
    bound_pts = max(0, i1 - i0)  # number of points in [E_min, E_max]
    total_xs_bound = total_xs[i0:i1]
    if log:
        print(f"  MTs: {mts}")
        print(f"  Energy range: {E_min:.3e} to {E_max:.3e} eV ({bound_pts} points)")
        peaks, _ = find_peaks(total_xs_bound)
        print(f"  There are {peaks.size} peaks in this range.")

    # ======================================================================
    # PERFORM VECTOR FITTING
    if vf_pieces is None:
        # divide into pieces for complex nuclides
        peaks, _ = find_peaks(total_xs_bound)
        n_peaks = peaks.size
        if n_peaks > 200 or bound_pts > 30000 or n_peaks * bound_pts > 100 * 10000:
            vf_pieces = max(5, n_peaks // 10, bound_pts // 2000)
        else:
            vf_pieces = 1

    space = kwargs.get("space", "E")
    if space == "sqrt_E":
        piece_width = (np.sqrt(E_max) - np.sqrt(E_min)) / vf_pieces
    else:
        piece_width = (E_max - E_min) / vf_pieces
    # print(f"Piece width {piece_width}")
    alpha = nuc_ce.atomic_weight_ratio / (K_BOLTZMANN * TEMPERATURE_LIMIT)

    poles, residues, remainder_list, energy_indices_list, bcf_list = [], [], [], [], []
    # VF piece by piece
    for i_piece in range(vf_pieces):
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{vf_pieces}...")
        # Calculate piece boundaries based on space
        if space == "sqrt_E":
            # Boundaries in sqrt(E) space
            sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
            sqrt_E_right = min(np.sqrt(E_max), sqrt_E_left + piece_width)
            E_left = sqrt_E_left**2
            E_right = sqrt_E_right**2

            # Doppler broadening extension (matching WMP logic)
            if i_piece == 0 or np.sqrt(alpha) * sqrt_E_left < 4.0:
                e_start = E_left
            else:
                e_start = max(E_min, (np.sqrt(alpha) * sqrt_E_left - 4.0) ** 2 / alpha)
            e_end = min(E_max, (np.sqrt(alpha) * sqrt_E_right + 4.0) ** 2 / alpha)
        else:  # space == "E"
            # Boundaries in E space
            E_left = E_min + i_piece * piece_width
            E_right = min(E_max, E_left + piece_width)

            # For E space, convert to sqrt for Doppler calculation
            sqrt_E_left = np.sqrt(E_left)
            sqrt_E_right = np.sqrt(E_right)

            # Same Doppler logic
            if i_piece == 0 or np.sqrt(alpha) * sqrt_E_left < 4.0:
                e_start = E_left
            else:
                e_start = max(E_min, (np.sqrt(alpha) * sqrt_E_left - 4.0) ** 2 / alpha)
            e_end = min(E_max, (np.sqrt(alpha) * sqrt_E_right + 4.0) ** 2 / alpha)

        # start E of this piece
        # E_left = E_min + i_piece * piece_width
        # E_right = min(E_max, E_left + piece_width)

        # s_left = np.sqrt(alpha * max(E_left, 0.0))
        # s_right = np.sqrt(alpha * max(E_right, 0.0))
        # if i_piece == 0 or s_left < 4:
        #     e_start = E_min
        # else:
        #     e_start = max(E_min, (s_left - 4) ** 2 / alpha)

        # e_end = min(E_max, (s_right + 4) ** 2 / alpha)

        # --- 3) Slice indices on the *E* grid ---
        e_start_idx = max(0, np.searchsorted(energy, e_start, side="right") - 1)
        e_end_idx = min(n_points, np.searchsorted(energy, e_end, side="left") + 1)
        if e_end_idx <= e_start_idx + 1:
            # ensure at least two points; mildly expand if needed
            e_start_idx = max(0, e_start_idx - 1)
            e_end_idx = min(n_points, e_end_idx + 1)

        e_idx = range(e_start_idx, e_end_idx)

        if log:
            print(
                f"  Piece {i_piece + 1}: E={energy[e_start_idx]:.3e} to "
                f"{energy[e_end_idx - 1]:.3e} eV"
            )

        if fit_mask_guard > 0:
            g = 0.1 * (e_end - e_start)
            mask_fit = (energy >= e_start - g) & (energy <= e_end + g)
            mask_core = (energy >= e_start) & (energy <= e_end)
            E_piece = energy[mask_fit]
            sig_s_piece = elastic_xs[mask_fit]
            sig_a_piece = absorption_xs[mask_fit]
            sig_f_piece = fission_xs[mask_fit] if fissionable else None

            w, z, fsz, faz, *rest = aaa_xs(
                E_piece,
                sig_s_piece,
                sig_a_piece,
                sigma_f=sig_f_piece,
                rtol=kwargs.get("rtol", 1e-13),
                mmax=kwargs.get("mmax", 100),
                log=log,
                fit_mask=np.ones_like(E_piece, dtype=bool),
                core_mask=mask_core[mask_fit],
                space=space,
            )
        else:
            E_piece = energy[e_idx]
            sig_s_piece = ce_xs[0, e_idx]
            sig_a_piece = ce_xs[1, e_idx]
            sig_f_piece = ce_xs[2, e_idx] if fissionable else None
            if fissionable:
                channels = [sig_s_piece, sig_a_piece, sig_f_piece]
            else:
                channels = [sig_s_piece, sig_a_piece]
            w, z, fz, R, err_hist = miaaa_xs(
                E_piece,
                channels,
                method=kwargs.get("method", "full_svd"),
                rtol=kwargs.get("rtol", 1e-13),
                mmax=kwargs.get("mmax", 100),
                greedy_metric="relative",  # relative or absolute_sum
                # greedy_metric="absolute_sum",  # relative or absolute_sum
                log=log,
                space=space,
                normalize=True,
                lawson_iter=0,
            )
            if space == "sqrt_E":
                Z = np.sqrt(E_piece)
            else:
                Z = E_piece

            if cleanup:
                pol, res, _, _ = proper_rational(
                    z,
                    w,
                    w,
                    fz,
                    R,
                    Z,
                    # pole_extraction=kwargs.get("pole_extraction", None),
                    # max_poly_degree=kwargs.get("max_poly_degree", 0),
                )
                z, fz, w = spurious_cleanup(
                    pol, res.T, z, fz, w, E_piece, R.T, cleanup_tol=cleanup_tol
                )
            if len(w) == 2 * len(z):  # YES LAWSON
                m = len(z)
                w_num = w[m : 2 * m]
                w_den = w[:m]
                poles_s, residues_list, info = proper_rational(
                    z,
                    w_num,
                    w_den,
                    fz,
                    R,
                    E_piece,
                    # pole_extraction="polynomial", max_poly_degree=2,
                    # pole_extraction="pseudo_pole", n_pseudo_poles=2,
                )
            else:  # NO LAWSON
                poles_s, residues_list, remainder, poly_info = proper_rational(
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
                print(f"Piece number of poles: {len(poles_s)}")

        poles.append(poles_s)
        residues.append(residues_list)  # because of errors in WMP
        # Store the polynomial info for THIS piece
        remainder_list.append(remainder)
        energy_indices_list.append([e_start_idx, e_end_idx])  # Store indices
        bcf_list.append(R)

        if plot_each_slice:
            if len(w) == 2 * len(z):  # YES LAWSON
                R_pieces = evaluate_miaaa(
                    Z, w, z, fz, space=space, w_num=w_num, w_den=w_den
                )
            else:  # NO LAWSON
                R_pieces = evaluate_miaaa(Z, w, z, fz, space=space)
            R_fission = R_pieces[2] if fissionable else None
            plot_aaa_results(
                Z,
                sig_s_piece,
                sig_a_piece,
                R_pieces[0],
                R_pieces[1],
                sigma_f=sig_f_piece,
                R_f=R_fission,
                path_out=aaa_plot_loc,
            )

    # print number of poles
    n_poles = sum([p.size for p in poles])
    if log:
        print(f"Total number of poles: {n_poles}")

    if vf_pieces == 1:
        channels_data = {
            "elastic": sig_s_piece,
            "absorption": sig_a_piece,
            "fission": sig_f_piece,
        }
        # Main reconstruction plot with remainder in subplot
        poles = poles[0]
        residues = residues[0]

        for p in np.sort(np.sqrt(poles)):
            print(f"pole real:  {p.real:.2e}  | imag: {p.imag:.2e} ")

        plot_reconstruction(
            E_piece,
            channels_data,
            poles,
            residues,
            name="U238",
            path_out=plot_path,
            plot_type="loglog",
            show_error=True,
            error_type="relative",
            poly_info=poly_info,
            fit_space=space,
        )
        plot_reconstruction(
            E_piece,
            channels_data,
            poles,
            residues,
            name="U238",
            path_out=plot_path,
            plot_type="loglog",
            show_error=True,
            error_type="absolute",
            poly_info=poly_info,
            fit_space=space,
        )
        plot_miaaa_convergence(err_hist, rtol=None, path_out=aaa_plot_loc)

    if output_format == "mp_data":
        # collect multipole data into a dictionary
        mp_data = {
            "name": nuc_ce.name,
            "AWR": nuc_ce.atomic_weight_ratio,
            "E_min": E_min,
            "E_max": E_max,
            "poles": poles,
            "residues": residues,
        }

        # dump multipole data to file
        if path_out:
            if not os.path.exists(mp_path_out):
                os.makedirs(mp_path_out)
            if not mp_filename:
                mp_filename = f"{nuc_ce.name}_mp.pickle"
            mp_filename = os.path.join(mp_path_out, mp_filename)
            with open(mp_filename, "wb") as f:
                pickle.dump(mp_data, f)
            if log:
                print(f"Dumped multipole data to file: {mp_filename}")
    elif output_format == "wmp":
        # Collect all poles, residues, and polynomial info
        all_poles = []
        all_residues = []
        for i_piece in range(vf_pieces):
            poles_piece = poles[i_piece]
            res_piece = residues[i_piece]

            all_poles.append(poles_piece)
            all_residues.append(res_piece)

        # Stack poles and residues in WMP format
        wmp_data = []
        for ip in range(vf_pieces):
            if len(all_poles[ip]) > 0:
                # Stack as [pole, res_s, res_a, res_f] per row
                if fissionable:
                    piece_data = np.column_stack(
                        [
                            all_poles[ip],
                            all_residues[ip][0],  # elastic
                            all_residues[ip][1],  # absorption
                            all_residues[ip][2],  # fission
                        ]
                    )
                else:
                    piece_data = np.column_stack(
                        [
                            all_poles[ip],
                            all_residues[ip][0],  # elastic
                            all_residues[ip][1],  # absorption
                        ]
                    )
                wmp_data.append(piece_data)

        # Concatenate all pieces
        data = np.vstack(wmp_data) if wmp_data else np.array([])
        # Windows array
        n_windows = vf_pieces
        spacing = (np.sqrt(E_max) - np.sqrt(E_min)) / n_windows

        windows = []
        pole_count = 0
        for iw in range(n_windows):
            n_poles_piece = len(all_poles[iw]) if iw < len(all_poles) else 0
            if n_poles_piece > 0:
                windows.append([pole_count + 1, pole_count + n_poles_piece])
                pole_count += n_poles_piece
            else:
                windows.append([pole_count + 1, pole_count])

        # For curvefit, store your polynomial coefficients directly
        # Create a custom format that stores your polyval-compatible coefficients
        # curvefit = []
        # for iw in range(n_windows):
        #     poly = all_poly_coeffs[iw] if iw < len(all_poly_coeffs) else None

        #     if poly is not None and isinstance(poly, list):
        #         # Store coefficients for each channel as-is
        #         n_channels = 3 if fissionable else 2
        #         cf = []
        #         for ch in range(n_channels):
        #             if ch < len(poly) and poly[ch] is not None:
        #                 # Store the polynomial coefficients directly
        #                 cf.append(poly[ch])
        #             else:
        #                 # No polynomial for this channel
        #                 cf.append([0.0])  # Just a zero constant
        #         curvefit.append(cf)
        #     else:
        #         # No polynomials for this window
        #         n_channels = 3 if fissionable else 2
        #         curvefit.append([[0.0]] * n_channels)

        # Convert curvefit to a padded array for HDF5 storage
        # Find max polynomial degree
        # max_poly_len = 0
        # for window_cf in curvefit:
        #     for channel_cf in window_cf:
        #         max_poly_len = max(max_poly_len, len(channel_cf))

        # # Pad all to same length
        # n_channels = 3 if fissionable else 2
        # curvefit_array = np.zeros((n_windows, max_poly_len, n_channels))
        # for iw, window_cf in enumerate(curvefit):
        #     for ch, channel_cf in enumerate(window_cf):
        #         curvefit_array[iw, : len(channel_cf), ch] = channel_cf

        # Write to HDF5
        wmp_filename = f"{nuc_ce.name}_wmp.h5"
        filename = os.path.join(wmp_path_out, wmp_filename)

        with h5py.File(filename, "w", libver="earliest") as f:
            f.attrs["filetype"] = np.bytes_("data_wmp")
            WMP_VERSION_MAJOR = 1
            WMP_VERSION_MINOR = 3  # 1 for old, 2 for new
            WMP_VERSION = (WMP_VERSION_MAJOR, WMP_VERSION_MINOR)
            f.attrs["version"] = np.array(WMP_VERSION)
            g = f.create_group(nuc_ce.name)
            # Write scalars.
            g.create_dataset("version", data=np.array(WMP_VERSION))
            #
            g.create_dataset("spacing", data=np.array(spacing))
            g.create_dataset(
                "sqrtAWR", data=np.array(np.sqrt(nuc_ce.atomic_weight_ratio))
            )
            g.create_dataset("E_min", data=np.array(E_min))
            g.create_dataset("E_max", data=np.array(E_max))

            # Write arrays
            g.create_dataset("data", data=data)
            g.create_dataset("windows", data=np.array(windows))
            g.create_dataset("broaden_poly", data=np.ones(n_windows, dtype=np.int8))
            g.create_dataset("curvefit", data=[])

            remainder_group = g.create_group("remainder_data")
            for i, remainder_piece in enumerate(remainder_list):
                remainder_group.create_dataset(f"window_{i}", data=remainder_piece)
            g.create_dataset("energy_indices", data=np.array(energy_indices_list))
            # Also store the full energy grid so you can reconstruct
            g.create_dataset("energy_grid", data=energy)
            bcf_group = g.create_group("bcf_data")
            for i, R_piece in enumerate(bcf_list):
                bcf_group.create_dataset(f"window_{i}", data=R_piece)

            # Add metadata about the fit space so you know how to evaluate later
            g.attrs["fit_space"] = np.bytes_(space)
            # Indicates numpy.polyval format
            g.attrs["poly_format"] = np.bytes_("polyval")
