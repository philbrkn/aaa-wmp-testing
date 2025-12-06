import os
import pickle
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def apply_polyfit_background(poles, res, Z, remainder, max_poly_degree=0):
    k = remainder.shape[0]
    # Fit polynomial to remainder
    poly_coeffs = []
    for i in range(k):
        if np.max(np.abs(remainder[i, :])) > 1e-12:
            # Fit polynomial
            p = np.polyfit(
                Z.real if np.allclose(Z.imag, 0) else Z,
                remainder[i, :],
                max_poly_degree,
            )
            poly_coeffs.append(p)
        else:
            poly_coeffs.append(None)
    return poly_coeffs


def read_and_evaluate_wmp_debug(filename, E_eval, background_method="poly", deg=1):
    """Read WMP file and evaluate using evaluate_simple"""
    with h5py.File(filename, "r") as f:
        g = f[list(f.keys())[0]]

        # Read data
        data = g["data"][:]
        windows = g["windows"][:]
        fit_space = g.attrs.get("fit_space", b"sqrt_E")
        if isinstance(fit_space, bytes):
            fit_space = fit_space.decode()
        spacing = g["spacing"][()]
        E_min = g["E_min"][()]
        E_max = g["E_max"][()]
        sqrtAWR = g["sqrtAWR"][()]

        # Read energy indices and remainder data
        energy_indices = g["energy_indices"][:]
        energy_grid = g["energy_grid"][:]

        # Read remainder data
        remainder_data = []
        if "remainder_data" in g:
            remainder_group = g["remainder_data"]
            for i in range(len(windows)):
                if f"window_{i}" in remainder_group:
                    remainder_data.append(remainder_group[f"window_{i}"][:])
                else:
                    remainder_data.append(None)
        else:
            remainder_data = [None] * len(windows)

        # Extract poles and residues
        poles = data[:, 0]
        fissionable = data.shape[1] == 4
        if fissionable:
            residues = data[:, 1:4].T
        else:
            residues = data[:, 1:3].T

        n_channels = 3 if fissionable else 2
        n_windows = len(windows)

        # Calculate piece width
        if fit_space == "sqrt_E":
            piece_width = (np.sqrt(E_max) - np.sqrt(E_min)) / n_windows
        else:
            piece_width = (E_max - E_min) / n_windows

        total_xs = np.zeros((n_channels, len(E_eval)))

        for iw in range(n_windows):
            # Calculate NOMINAL window boundaries
            if fit_space == "sqrt_E":
                sqrt_E_left = np.sqrt(E_min) + iw * piece_width
                sqrt_E_right = min(np.sqrt(E_max), sqrt_E_left + piece_width)
                E_left_nominal = sqrt_E_left**2
                E_right_nominal = sqrt_E_right**2
            else:
                E_left_nominal = E_min + iw * piece_width
                E_right_nominal = min(E_max, E_left_nominal + piece_width)

            # Find E_eval points in NOMINAL window
            if iw == n_windows - 1:
                mask = (E_eval >= E_left_nominal) & (E_eval <= E_right_nominal)
            else:
                mask = (E_eval >= E_left_nominal) & (E_eval < E_right_nominal)

            if not np.any(mask):
                continue

            E_window = E_eval[mask]

            # Get poles and residues for this window
            start, end = windows[iw]
            if end >= start:
                window_poles = poles[start - 1 : end]
                window_residues = residues[:, start - 1 : end]
            else:
                window_poles = np.array([])
                window_residues = np.zeros((n_channels, 0))

            # Transform to correct space
            if fit_space == "sqrt_E":
                s = np.sqrt(E_window)
            else:
                s = E_window

            # Initialize output for this window
            xs_window = np.zeros((n_channels, len(E_window)))

            # Compute pole contributions
            if len(window_poles) > 0:
                denominators = s[:, np.newaxis] - window_poles[np.newaxis, :]
                for ch in range(n_channels):
                    xs_window[ch] = np.sum(
                        (window_residues[ch] / denominators).real, axis=1
                    )

            # Fit and add polynomial background
            if background_method == "poly" and remainder_data[iw] is not None:
                e_start_idx_stored, e_end_idx_stored = energy_indices[iw]
                E_remainder_full = energy_grid[e_start_idx_stored:e_end_idx_stored]

                # Find remainder points in NOMINAL window
                remainder_mask = (E_remainder_full >= E_left_nominal) & (
                    E_remainder_full <= E_right_nominal
                )
                E_remainder_nominal = E_remainder_full[remainder_mask]
                remainder_nominal = remainder_data[iw][:, remainder_mask]

                if fit_space == "sqrt_E":
                    s_remainder = np.sqrt(E_remainder_nominal)
                else:
                    s_remainder = E_remainder_nominal

                # Fit polynomial for each channel
                for ch in range(n_channels):
                    if (
                        remainder_nominal.shape[1] > deg
                        and np.max(np.abs(remainder_nominal[ch, :])) > 1e-12
                    ):
                        p = np.polyfit(s_remainder, remainder_nominal[ch, :], deg=deg)
                        xs_window[ch] += np.polyval(p, s).real

            # Store results
            total_xs[:, mask] = xs_window

            if iw % 100 == 0:
                print(f"Processed window {iw}/{n_windows}")

        return total_xs


def create_reference_from_njoy(njoy_pickle_path, E_grid=None, bounds=None):
    """
    Extract reference cross sections from NJOY pickle for WMP comparison.
    This follows the same logic as your fit_nuclide function.

    Parameters
    ----------
    njoy_pickle_path : str or Path
        Path to the NJOY pickle file
    E_grid : array-like, optional
        Energy grid for evaluation. If None, uses NJOY's natural grid
    bounds : dict, optional
        Energy bounds with 'E_min' and 'E_max' keys

    Returns
    -------
    dict
        Reference cross sections and energy grid
    """

    # Load NJOY data
    with open(njoy_pickle_path, "rb") as f:
        nuc_ce = pickle.load(f)

    # Determine energy bounds using same logic as fit_nuclide
    # First get the natural NJOY energy grid
    njoy_energy = nuc_ce.energy["0K"]

    # Find appropriate upper bound (same as fit_nuclide logic)
    E_max = njoy_energy[-1]  # default to full range
    E_max_idx = len(njoy_energy) - 1

    # Check for thresholds that would limit our range
    for mt in nuc_ce.reactions:
        if hasattr(nuc_ce.reactions[mt].xs["0K"], "_threshold_idx"):
            threshold_idx = nuc_ce.reactions[mt].xs["0K"]._threshold_idx
            if 0 < threshold_idx < E_max_idx:
                E_max_idx = threshold_idx
                E_max = njoy_energy[threshold_idx]

    # Apply user bounds if provided
    if bounds:
        E_min = max(bounds.get("E_min", njoy_energy[0]), njoy_energy[0])
        E_max = min(bounds.get("E_max", E_max), E_max)
    else:
        E_min = njoy_energy[0]

    # If no energy grid provided, use NJOY's natural grid in the range
    if E_grid is None:
        mask = (njoy_energy >= E_min) & (njoy_energy <= E_max)
        energy_eval = njoy_energy[mask]
    else:
        # Filter provided grid to valid range
        mask = (E_grid >= E_min) & (E_grid <= E_max)
        energy_eval = E_grid[mask]

    print(f"Reference energy range: {E_min:.2e} to {E_max:.2e} eV")
    print(f"Reference points: {len(energy_eval)}")

    # Extract cross sections using same logic as fit_nuclide
    reference_data = {}

    # Total cross section (MT=1)
    try:
        total_xs = nuc_ce[1].xs["0K"](energy_eval)
        reference_data["total"] = total_xs
    except KeyError:
        pass

    # Elastic scattering (MT=2)
    try:
        elastic_xs = nuc_ce[2].xs["0K"](energy_eval)
        reference_data["elastic"] = elastic_xs
    except KeyError:
        reference_data["elastic"] = np.zeros_like(energy_eval)

    # Absorption (MT=27)
    try:
        absorption_xs = nuc_ce[27].xs["0K"](energy_eval)
        reference_data["absorption"] = absorption_xs
    except KeyError:
        reference_data["absorption"] = np.zeros_like(energy_eval)

    # Fission (MT=18)
    try:
        fission_xs = nuc_ce[18].xs["0K"](energy_eval)
        reference_data["fission"] = fission_xs
    except KeyError:
        reference_data["fission"] = None

    # Return both cross sections and the energy grid used
    reference_data["energy"] = energy_eval
    reference_data["bounds"] = {"E_min": E_min, "E_max": E_max}

    return reference_data


def plot_wmp_comparison(
    wmp_data, reference_data, E_grid, name="U238", path_out="./plots", T=0.0
):
    """
    Simple function to plot WMP vs reference data with errors.

    Parameters
    ----------
    wmp_data : WindowedMultipole
        Your WMP data object
    reference_data : dict
        Reference data with keys 'elastic', 'absorption', 'fission'
    E_grid : array-like
        Energy grid for comparison
    name : str
        Nuclide name for plot titles
    path_out : str
        Directory to save plots
    T : float
        Temperature in K
    """

    # Extract WMP cross sections
    print("Evaluating WMP cross sections...")

    wmp_elastic = np.array(wmp_data[0])
    wmp_absorption = np.array(wmp_data[1])
    wmp_fission = np.array(wmp_data[2])

    # Plot each channel
    channels = [
        ("elastic", "σ_s", wmp_elastic, reference_data.get("elastic")),
        ("absorption", "σ_a", wmp_absorption, reference_data.get("absorption")),
        ("fission", "σ_f", wmp_fission, reference_data.get("fission")),
    ]

    os.makedirs(path_out, exist_ok=True)

    for channel_name, symbol, wmp_xs, ref_xs in channels:
        if ref_xs is None:
            continue

        # Create figure with error subplot
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
        )

        # Main plot
        ax1.semilogy(E_grid, ref_xs, "b-", label=f"Reference {symbol}", linewidth=2)
        ax1.semilogy(E_grid, wmp_xs, "r--", label=f"WMP {symbol}", linewidth=2)
        ax1.set_ylabel("Cross section (b)")
        ax1.set_title(f"{name} - {channel_name.capitalize()} Channel")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)
        ax1.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
        ax1.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

        # Error subplot
        mask = ref_xs != 0
        rel_error = np.full_like(ref_xs, np.nan)
        rel_error[mask] = np.abs(wmp_xs[mask] - ref_xs[mask]) / ref_xs[mask] * 100
        abs_err = np.abs(wmp_xs - ref_xs)
        ax2.semilogy(E_grid, abs_err, "k-", linewidth=1.5)
        # ax2.semilogy(E_grid, rel_error, "k-", linewidth=1.5)
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Relative Error (%)")

        # grid lines and such
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_ylim(1e-7, 1e1)
        # ax2.set_ylim(1e0, 1e3)
        ax2.grid(True, which="both", alpha=0.3)
        # horizontal lines
        ax2.grid(which="major", axis="y", linestyle="--", linewidth=1)
        # vertical lines.
        ax2.grid(which="major", axis="x", linestyle="-", linewidth=0.8, alpha=0.7)
        ax2.grid(which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
        from matplotlib.ticker import LogLocator

        ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
        # ax2.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))

        plt.tight_layout()
        plt.savefig(
            os.path.join(path_out, f"{name}_{channel_name}_NEWAAA_wmp_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Print error statistics
        if np.any(mask):
            max_err = np.max(np.abs(rel_error[mask]))
            rms_err = np.sqrt(np.mean(rel_error[mask] ** 2))
            print(
                f"{channel_name.capitalize()}: Max error = {max_err:.2e}%, RMS error = {rms_err:.2e}%"
            )


if __name__ == "__main__":
    # filepath = "aaa_analyze_constant/U238_wmp.h5"
    # filepath = "aaa_test/U238_wmp_5e-4_328p.h5"
    # filepath = "data/output/aaa_in_h5wmp_format/U238_wmp_5e-4_2000p_sqrtE.h5"
    name = "U238"
    njoy_pickle_path = f"data/input/NJOY_pickles/{name}_NJOY.pickle"
    # bounds={"E_min": 17400, "E_max": 17600}
    # bounds = {"E_min": 0, "E_max": 2e4}
    # bounds = {"E_min": 1, "E_max": 60}
    bounds = {"E_min": 0.1, "E_max": 19999}
    # bounds = {"E_min": 0, "E_max": 9999}

    reference_data = create_reference_from_njoy(njoy_pickle_path, bounds=bounds)
    E_grid = reference_data["energy"]

    # TODO: modular to fissionable
    xs_refs = np.asarray(
        [
            reference_data["elastic"],
            reference_data["absorption"],
            reference_data["fission"],
        ]
    )
    background_method = "poly"
    results = read_and_evaluate_wmp_debug(filepath, E_grid, background_method, deg=3)
    plot_wmp_comparison(
        results,
        reference_data,
        E_grid,
        name="U238",
        path_out="data/output/plotsdoppler",
        T=0.0,
    )
