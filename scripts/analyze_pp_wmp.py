import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from multipole_deplete_v3 import WindowedMultipole


def plot_wmp_comparison(
    wmp_data,
    reference_data,
    E_grid,
    name="U238",
    path_out="./plots",
    T=0.0,
    method="VF",
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
    method : str
        "VF" or "AAA" - determines which evaluate function to use
    """

    print("Evaluating WMP cross sections...")
    wmp_elastic = []
    wmp_absorption = []
    wmp_fission = []

    # Only apply -1j conversion for VF method
    if method == "VF":
        wmp_data.data[:, 1:] *= -1j

    for E in E_grid:
        if method == "AAA":
            sig_s, sig_a, sig_f = wmp_data._evaluate_aaa(E, T)
        else:
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

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
        )

        ax1.semilogy(E_grid, ref_xs, "b-", label=f"Reference {symbol}", linewidth=2)
        ax1.semilogy(E_grid, wmp_xs, "r--", label=f"WMP {symbol}", linewidth=2)
        ax1.set_ylabel("Cross section (b)")
        ax1.set_title(f"{name} - {channel_name.capitalize()} Channel ({method})")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)

        mask = ref_xs != 0
        rel_error = np.full_like(ref_xs, np.nan)
        rel_error[mask] = np.abs(wmp_xs[mask] - ref_xs[mask]) / ref_xs[mask] * 100

        ax2.semilogy(E_grid, rel_error, "k-", linewidth=1.5)
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Relative Error (%)")
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.set_ylim(1e-7, 1e1)
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                path_out, f"{name}_{channel_name}_wmp_comparison_{method}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        if np.any(mask):
            max_err = np.max(np.abs(rel_error[mask]))
            rms_err = np.sqrt(np.mean(rel_error[mask] ** 2))
            print(
                f"{channel_name.capitalize()}: Max error = {max_err:.2e}%, RMS error = {rms_err:.2e}%"
            )


def create_reference_from_njoy(njoy_pickle_path, E_grid=None, bounds=None):
    """Extract reference cross sections from NJOY pickle."""
    with open(njoy_pickle_path, "rb") as f:
        nuc_ce = pickle.load(f)

    njoy_energy = nuc_ce.energy["0K"]
    E_max = njoy_energy[-1]
    E_max_idx = len(njoy_energy) - 1

    for mt in nuc_ce.reactions:
        if hasattr(nuc_ce.reactions[mt].xs["0K"], "_threshold_idx"):
            threshold_idx = nuc_ce.reactions[mt].xs["0K"]._threshold_idx
            if 0 < threshold_idx < E_max_idx:
                E_max_idx = threshold_idx
                E_max = njoy_energy[threshold_idx]

    if bounds:
        E_min = max(bounds.get("E_min", njoy_energy[0]), njoy_energy[0])
        E_max = min(bounds.get("E_max", E_max), E_max)
    else:
        E_min = njoy_energy[0]

    if E_grid is None:
        mask = (njoy_energy >= E_min) & (njoy_energy <= E_max)
        energy_eval = njoy_energy[mask]
    else:
        mask = (E_grid >= E_min) & (E_grid <= E_max)
        energy_eval = E_grid[mask]

    print(f"Reference energy range: {E_min:.2e} to {E_max:.2e} eV")
    print(f"Reference points: {len(energy_eval)}")

    reference_data = {}

    try:
        reference_data["elastic"] = nuc_ce[2].xs["0K"](energy_eval)
    except KeyError:
        reference_data["elastic"] = np.zeros_like(energy_eval)

    try:
        reference_data["absorption"] = nuc_ce[27].xs["0K"](energy_eval)
    except KeyError:
        reference_data["absorption"] = np.zeros_like(energy_eval)

    try:
        reference_data["fission"] = nuc_ce[18].xs["0K"](energy_eval)
    except KeyError:
        reference_data["fission"] = None

    reference_data["energy"] = energy_eval
    reference_data["bounds"] = {"E_min": E_min, "E_max": E_max}

    return reference_data


def load_aaa_wmp(wmp_file, pseudo_file=None):
    """
    Load AAA WMP data including pseudopoles.

    Parameters
    ----------
    wmp_file : str
        Path to HDF5 file with physical poles
    pseudo_file : str, optional
        Path to pickle file with pseudopoles. If None, looks for
        wmp_file.replace('.h5', '_pseudo.pickle')

    Returns
    -------
    WindowedMultipole
        WMP object with pseudo_poles and pseudo_residues loaded
    """
    # Load main WMP data
    wmp_data = WindowedMultipole.from_hdf5(wmp_file)

    # Try to load pseudopoles
    if pseudo_file is None:
        pseudo_file = wmp_file.replace(".h5", "_pseudo.pickle")

    if os.path.exists(pseudo_file):
        with open(pseudo_file, "rb") as f:
            pseudo_data = pickle.load(f)
        wmp_data.pseudo_poles = pseudo_data["pseudo_poles"]
        wmp_data.pseudo_residues = pseudo_data["pseudo_residues"]
        print(f"Loaded pseudopoles from {pseudo_file}")
        n_pseudo = sum(len(pp) for pp in wmp_data.pseudo_poles)
        print(f"Total pseudopoles: {n_pseudo}")
    else:
        print(f"No pseudopole file found at {pseudo_file}")
        wmp_data.pseudo_poles = None
        wmp_data.pseudo_residues = None

    return wmp_data


# ============== MAIN ==============

name = "U238"
METHOD = "AAA"  # or "VF"

# wmp_file = "data/output/WMP_Lib_viii.0/U238/U238_4kphys_6kpp.h5"
wmp_file = "data/output/WMP_Lib_viii.0/U238/U238.h5"
njoy_pickle_path = f"data/input/NJOY_pickles/{name}_NJOY.pickle"

# Load WMP data
if METHOD == "AAA":
    wmp_data = load_aaa_wmp(wmp_file)
else:
    wmp_data = WindowedMultipole.from_hdf5(wmp_file)

print(f"Total number of physical poles: {len(wmp_data.data)}")
print(f"Number of windows: {len(wmp_data.windows)}")

if METHOD == "AAA" and wmp_data.pseudo_poles is not None:
    n_pseudo = sum(len(pp) for pp in wmp_data.pseudo_poles)
    print(f"Total pseudopoles: {n_pseudo}")
    print(f"Average pseudopoles per window: {n_pseudo / len(wmp_data.windows):.2f}")

bounds = {"E_min": 1, "E_max": 20000}
reference_data = create_reference_from_njoy(njoy_pickle_path, bounds=bounds)
E_grid = reference_data["energy"]

### DEBUG ###
E_test = 6.67  # first big U-238 resonance
sqrtE = np.sqrt(E_test)

# Find window
i_window = int((sqrtE - np.sqrt(wmp_data.E_min)) / wmp_data.spacing)
print(f"E={E_test}, window={i_window}")

# Physical poles in this window
startw = wmp_data.windows[i_window, 0] - 1
endw = wmp_data.windows[i_window, 1]
print(f"Physical poles: {startw} to {endw} ({endw - startw} poles)")

# Pseudopoles in this window
if wmp_data.pseudo_poles is not None:
    pp = wmp_data.pseudo_poles[i_window]
    print(f"Pseudopoles in window: {len(pp)}")
    if len(pp) > 0:
        print(f"Pseudopole values: {pp}")

# Evaluate
sig_s, sig_a, sig_f = wmp_data._evaluate_aaa(E_test, 0.0)
print(f"WMP: sig_s={sig_s:.4f}, sig_a={sig_a:.4f}, sig_f={sig_f:.4f}")

# Reference
ref_idx = np.argmin(np.abs(reference_data["energy"] - E_test))
print(
    f"Ref: sig_s={reference_data['elastic'][ref_idx]:.4f}, sig_a={reference_data['absorption'][ref_idx]:.4f}"
)
###########

plot_wmp_comparison(
    wmp_data,
    reference_data,
    E_grid,
    name=name,
    path_out="data/output/U238/wmp_plots",
    method=METHOD,
)
