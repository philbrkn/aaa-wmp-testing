# aaa_wmp/processing/piece_fitting.py
import os

import numpy as np

from ..core.aaa_fitting import evaluate_miaaa, miaaa_xs
from ..core.conversion import proper_rational
from ..core.plotting import plot_aaa_results


def calculate_piece_boundaries(i_piece, piece_width, data, alpha, space):
    energy = data["energy"]
    E_min = data["E_min"]
    E_max = data["E_max"]
    n_points = len(energy)
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
    return e_start_idx, e_end_idx


def fit_piece(i_piece, data, piece_width, alpha, space, **kwargs):
    """Fit a single piece of the energy range."""
    log = kwargs.get("log", False)
    cleanup = kwargs.get("cleanup", False)
    cleanup_tol = kwargs.get("cleanup_tol", 1e-6)
    plot_each_slice = kwargs.get("plot_each_slice", False)
    path_out = kwargs.get("path_out", "./output")

    energy = data["energy"]
    ce_xs = data["ce_xs"]
    fissionable = data["fissionable"]

    e_start_idx, e_end_idx = calculate_piece_boundaries(
        i_piece, piece_width, data, alpha, space
    )
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
        plot_piece(
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


def plot_piece(E_piece, Z, channels, w, z, fz, fissionable, space, path_out, i_piece):
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
        R_pieces = evaluate_miaaa(Z, w, z, fz, space=space, w_den=w_den, w_num=w_num)
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
