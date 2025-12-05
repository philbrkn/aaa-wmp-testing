#!/usr/bin/env python
"""
Plot NJOY cross section data for U238 showing RRR, URR, and continuum regions.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openmc.data import IncidentNeutron


def load_njoy_data(njoy_file):
    """Load NJOY data from pickle file."""
    with open(njoy_file, "rb") as f:
        return pickle.load(f)


def get_resonance_boundaries(endf_file):
    """Extract resonance region boundaries from ENDF file."""
    endf_data = IncidentNeutron.from_endf(endf_file)

    boundaries = {"rrr_min": None, "rrr_max": None, "urr_min": None, "urr_max": None}

    if hasattr(endf_data, "resonances"):
        res = endf_data.resonances

        # Resolved resonance region
        if hasattr(res, "resolved") and res.resolved is not None:
            if hasattr(res.resolved, "energy_min"):
                boundaries["rrr_min"] = res.resolved.energy_min
            if hasattr(res.resolved, "energy_max"):
                boundaries["rrr_max"] = res.resolved.energy_max

        # Unresolved resonance region
        if hasattr(res, "unresolved") and res.unresolved is not None:
            if hasattr(res.unresolved, "energy_min"):
                boundaries["urr_min"] = res.unresolved.energy_min
            if hasattr(res.unresolved, "energy_max"):
                boundaries["urr_max"] = res.unresolved.energy_max

    return boundaries


def add_resonance_regions(ax, boundaries, energy):
    """Add shaded regions for RRR, URR, and continuum."""

    y_limits = ax.get_ylim()

    # RRR region
    if boundaries["rrr_min"] is not None and boundaries["rrr_max"] is not None:
        ax.axvspan(
            boundaries["rrr_min"],
            boundaries["rrr_max"],
            alpha=0.15,
            color="blue",
            label="RRR",
        )
        ax.axvline(boundaries["rrr_max"], color="black", linestyle="--", alpha=0.5)

    # URR region
    if boundaries["urr_min"] is not None and boundaries["urr_max"] is not None:
        ax.axvspan(
            boundaries["urr_min"],
            boundaries["urr_max"],
            alpha=0.15,
            color="orange",
            label="URR",
        )
        ax.axvline(boundaries["urr_max"], color="black", linestyle="--", alpha=0.5)

    # Continuum region (fast range)
    if boundaries["urr_max"] is not None:
        # Continuum starts where URR ends
        ax.axvspan(
            boundaries["urr_max"],
            energy[-1],
            alpha=0.15,
            color="green",
            label="Continuum",
        )
    elif boundaries["rrr_max"] is not None:
        # If no URR, continuum starts where RRR ends
        ax.axvspan(
            boundaries["rrr_max"],
            energy[-1],
            alpha=0.2,
            color="green",
            label="Continuum",
        )


def plot_total_view(nuc_ce, boundaries, output_dir=None):
    from pathlib import Path

    e_min = 1000.0
    energy = nuc_ce.energy["293K"]
    total_xs = nuc_ce[1].xs["293K"](energy)

    # apply lower energy cutoff for the total curve
    mask = energy >= e_min
    energy = energy[mask]
    total_xs = total_xs[mask]

    fig, ax = plt.subplots(figsize=(7, 5))

    # prepare URR
    urr = nuc_ce.urr["293K"]
    n_energy = urr.table.shape[0]
    n_band = urr.table.shape[2]

    print(f"Found URR tables: {n_energy} energy points, {n_band} probability bands")

    # Determine whether urr provides sigma0 (background sigma) to convert table values
    sigma0 = getattr(urr, "sigma0", None)

    # collect y extents from URR and total to set limits later
    y_mins = []
    y_maxs = []

    for i in range(n_energy):
        E_i = urr.energy[i]
        if E_i < e_min:
            continue

        # energy cell bounds
        if i > 0:
            e_left = E_i - 0.5 * (E_i - urr.energy[i - 1])
        else:
            e_left = E_i - 0.5 * (urr.energy[i + 1] - E_i)
        if i < n_energy - 1:
            e_right = E_i + 0.5 * (urr.energy[i + 1] - E_i)
        else:
            e_right = E_i + 0.5 * (E_i - urr.energy[i - 1])

        # physical anchor sigma at this energy
        if sigma0 is not None:
            sigma_anchor = sigma0[i]
        else:
            # use pointwise total xs at the URR energy (fallback)
            sigma_anchor = nuc_ce[1].xs["293K"](np.array([E_i]))[0]

        for j in range(n_band):
            # compute band bottom/top from table values and map to physical sigma
            # urr.table[i,1,j] is the band sigma in the table (often relative). Multiply by anchor.
            band_val = urr.table[i, 1, j]
            if j < n_band - 1:
                band_val_next = urr.table[i, 1, j + 1]
            else:
                band_val_next = band_val

            xs_bottom = sigma_anchor * band_val
            xs_top = sigma_anchor * band_val_next

            # skip non-positive or zero-height bands (not plottable on log scale)
            if xs_top <= 0 or xs_bottom <= 0 or xs_top <= xs_bottom:
                continue

            # compute grayscale from probability (normalized)
            max_prob = np.diff(urr.table[i, 0, :]).max()
            if j > 0:
                prob = (
                    (urr.table[i, 0, j] - urr.table[i, 0, j - 1]) / max_prob
                    if max_prob > 0
                    else 0.0
                )
            else:
                prob = urr.table[i, 0, j] / max_prob if max_prob > 0 else 0.0
            gray_value = max(0.0, min(1.0, 1.0 - prob * 0.7))

            rect = Rectangle(
                (e_left, xs_bottom),
                e_right - e_left,
                xs_top - xs_bottom,
                facecolor=str(gray_value),
                edgecolor="none",
                linewidth=0,
                transform=ax.transData,
            )
            ax.add_patch(rect)

            y_mins.append(xs_bottom)
            y_maxs.append(xs_top)

    # now plot total curve
    ax.plot(energy, total_xs, "black", linewidth=0.07)  # , label="Total")

    # thicken fast range
    if boundaries.get("urr_max") is not None:
        fast_mask = energy >= boundaries["urr_max"]
    elif boundaries.get("rrr_max") is not None:
        fast_mask = energy >= boundaries["rrr_max"]
    else:
        fast_mask = np.zeros_like(energy, dtype=bool)

    ax.plot(energy[fast_mask], total_xs[fast_mask], color="black", linewidth=0.7)
    # add resonance regions (ensure this function respects e_min if needed)
    add_resonance_regions(ax, boundaries, energy)

    # set log scales and limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(e_min, energy[-1])
    ax.set_xlim(e_min, 3e6)

    # choose y limits that include URR bands and the total curve
    if y_mins and y_maxs:
        ymin = min(min(y_mins), np.nanmin(total_xs)) * 0.5
        ymax = max(max(y_maxs), np.nanmax(total_xs)) * 2.0
        # protect against non-positive
        ymin = max(ymin, 1e-6)
        # ax.set_ylim(ymin, ymax)
        ax.set_ylim(1e-3, ymax)
    else:
        # fallback sensible limits
        ax.set_ylim(np.nanmin(total_xs) * 0.1, np.nanmax(total_xs) * 2.0)

    # region labels: compute y_pos after limits are set
    # y_pos = ax.get_ylim()[0] * 10.0

    # if boundaries.get("rrr_min") and boundaries.get("rrr_max"):
    #     rrr_center = np.sqrt(boundaries["rrr_min"] * boundaries["rrr_max"])
    #     if rrr_center >= e_min:
    #         ax.text(rrr_center, y_pos, "RRR", ha="center", fontsize=12, fontweight="bold")

    # if boundaries.get("urr_min") and boundaries.get("urr_max"):
    #     urr_center = np.sqrt(boundaries["urr_min"] * boundaries["urr_max"])
    #     if urr_center >= e_min:
    #         ax.text(urr_center, y_pos, "URR", ha="center", fontsize=12, fontweight="bold")

    # if boundaries.get("urr_max"):
    #     cont_center = np.sqrt(boundaries["urr_max"] * energy[-1])
    #     if cont_center >= e_min:
    #         ax.text(cont_center, y_pos, "Continuum", ha="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("Cross Section (barns)", fontsize=12)
    ax.set_title("U238 Total Cross Section at 293K", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "u238_combined.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_dir / 'u238_combined.png'}")


def print_region_info(boundaries):
    """Print information about resonance regions."""
    print("\n" + "=" * 60)
    print("U238 Resonance Region Information")
    print("=" * 60)

    if boundaries["rrr_min"] is not None and boundaries["rrr_max"] is not None:
        print("Resolved Resonance Region (RRR):")
        print(
            f"  Energy range: {boundaries['rrr_min']:.2e} - {boundaries['rrr_max']:.2e} eV"
        )
        print(f"  ({boundaries['rrr_min']:.2f} - {boundaries['rrr_max']:.2f} eV)")

    if boundaries["urr_min"] is not None and boundaries["urr_max"] is not None:
        print("\nUnresolved Resonance Region (URR):")
        print(
            f"  Energy range: {boundaries['urr_min']:.2e} - {boundaries['urr_max']:.2e} eV"
        )
        print(f"  ({boundaries['urr_min']:.2f} - {boundaries['urr_max']:.2f} eV)")

    if boundaries["urr_max"] is not None:
        print("\nContinuum Region (Fast Range):")
        print(f"  Starts at: {boundaries['urr_max']:.2e} eV")
        print(f"  ({boundaries['urr_max']:.2f} eV)")
    elif boundaries["rrr_max"] is not None:
        print("\nContinuum Region (Fast Range):")
        print(f"  Starts at: {boundaries['rrr_max']:.2e} eV")
        print(f"  ({boundaries['rrr_max']:.2f} eV)")

    print("=" * 60 + "\n")


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    name = "U238"

    # ENDF and NJOY file paths
    endf_file = base_dir / "data/input/ENDF/ENDF-VIII-data" / "n-092_U_238.endf"
    njoy_file = base_dir / "data/input/NJOY_pickles" / f"{name}_NJOY.pickle"
    output_dir = base_dir / "data/output" / name / "plots"

    # Check if NJOY pickle exists
    if not njoy_file.exists():
        print(f"Error: NJOY pickle file not found at {njoy_file}")
        print("Please run the fitting script first to generate NJOY data.")
        sys.exit(1)

    # Check if ENDF file exists
    if not endf_file.exists():
        print(f"Error: ENDF file not found at {endf_file}")
        sys.exit(1)

    print(f"Loading NJOY data from {njoy_file}")
    nuc_ce = load_njoy_data(njoy_file)

    print(f"Extracting resonance boundaries from {endf_file}")
    boundaries = get_resonance_boundaries(endf_file)

    # Print region information
    print_region_info(boundaries)

    # Create plots
    print("\nGenerating plots...")

    # Individual subplot view
    # plot_cross_sections_by_region(nuc_ce, boundaries, output_dir)

    # Combined single plot view
    plot_total_view(nuc_ce, boundaries, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
