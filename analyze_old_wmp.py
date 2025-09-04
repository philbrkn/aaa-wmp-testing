import h5py
from openmc.data.multipole_old import WindowedMultipole
import numpy as np
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt


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
    wmp_elastic = []
    wmp_absorption = []
    wmp_fission = []

    for E in E_grid:
        sig_s, sig_a, sig_f = wmp_data._evaluate(E, T)
        wmp_elastic.append(sig_s)
        wmp_absorption.append(sig_a)
        wmp_fission.append(sig_f)

    wmp_elastic = np.array(wmp_elastic)
    wmp_absorption = np.array(wmp_absorption)
    wmp_fission = np.array(wmp_fission)

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
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Main plot
        ax1.loglog(E_grid, ref_xs, "b-", label=f"Reference {symbol}", linewidth=2)
        ax1.loglog(E_grid, wmp_xs, "r--", label=f"WMP {symbol}", linewidth=2)
        ax1.set_ylabel("Cross section (b)")
        ax1.set_title(f"{name} - {channel_name.capitalize()} Channel")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error subplot
        mask = ref_xs != 0
        rel_error = np.full_like(ref_xs, np.nan)
        rel_error[mask] = np.abs(wmp_xs[mask] - ref_xs[mask]) / ref_xs[mask] * 100

        # ax2.semilogx(E_grid, rel_error, "k-", linewidth=1.5)
        ax2.loglog(E_grid, rel_error, "k-", linewidth=1.5)
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Relative Error (%)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(path_out, f"{name}_{channel_name}_wmp_comparison.png"),
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


def count_wmp_poles_in_range(wmp_data, bounds):
    """
    Count how many poles are in a given energy range.

    Parameters
    ----------
    wmp_data : WindowedMultipole
        The WMP data object
    E_min : float
        Lower energy bound in eV
    E_max : float
        Upper energy bound in eV

    Returns
    -------
    int
        Number of poles in the energy range
    """
    # Convert pole energies from sqrt(E) to E
    # wmp_data.data[:, 0] contains poles in sqrt(E) format
    pole_energies = np.real(wmp_data.data[:, 0]) ** 2

    # Count poles in the range
    in_range = (pole_energies >= bounds['E_min']) & (pole_energies <= bounds['E_max'])
    n_poles = np.sum(in_range)

    print(f"Energy range: {bounds['E_min']:.2e} to {bounds['E_max']:.2e} eV")
    print(f"Number of poles: {n_poles}")
    print(f"Total poles in WMP: {len(pole_energies)}")

    return n_poles


# wmp_file = Path(__file__).parent / "windowing_h5_output" / "U238-vf-wmp-VIII.h5"
wmp_file = Path(__file__).parent / "ENDF-VIII-data" / "officialWMP-U238.h5"
njoy_pickle_path = Path(__file__).parent / "NJOY_pickles" / "U238_NJOY.pickle"


name = "U238"
wmp = WindowedMultipole(name)
wmp_data = wmp.from_hdf5(wmp_file)

# bounds = {'E_min': wmp_data.E_min, 'E_max': wmp_data.E_max}
bounds = {"E_min": 785, "E_max": 861}
reference_data = create_reference_from_njoy(njoy_pickle_path, bounds=bounds)
E_grid = reference_data["energy"]

plot_wmp_comparison(wmp_data, reference_data, E_grid, name="U238", path_out="./plots")
count_wmp_poles_in_range(wmp_data, bounds)
# print(wmp_data.windows[1:20])
# print(wmp_data.data.shape[0])
# sig_s, sig_a, sig_f = wmp_data._evaluate(E, T=0)
