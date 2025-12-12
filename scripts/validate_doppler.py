# scripts/validate_doppler.py
"""
Validate AAA-WMP Doppler broadening against NJOY reference data.

This script:
1. Loads AAA poles/residues from a fitted nuclide
2. Generates (or loads) NJOY reference data at multiple temperatures
3. Reconstructs cross sections using Faddeeva-based broadening
4. Computes error metrics and generates comparison plots

Usage:
    python scripts/validate_doppler.py
"""

import pickle
import sys
from math import pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aaa_wmp.io.njoy_interface import generate_temperature_references

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "name": "U238",
    "endf_file": "data/input/ENDF/ENDF-VIII-data/n-092_U_238.endf",
    "mp_data_file": "data/output/U238/mp_data/U238_mp.pickle",
    "output_dir": "data/output/U238/validation",
    # Validation temperatures (K)
    # Quick test: [293.6, 900, 1500]
    # Full validation: [293.6, 600, 900, 1200, 1500, 1800, 2400]
    # "temperatures": [294],
    "temperatures": [294, 600, 900, 1200, 1500],
    # Energy bounds for validation (None = use full fitted range)
    "E_min": 1,
    "E_max": 200,
    # NJOY settings
    "njoy_error": 5e-4,
    # Error thresholds for pass/fail
    "rtol_threshold": 1e-2,  # 1% relative error
    "atol_threshold": 1e-5,  # barns
    "space": "sqrt_E",
}

# =============================================================================
# Faddeeva-based reconstruction (from OpenMC multipole.py)
# =============================================================================


def faddeeva(z):
    """Evaluate the complex Faddeeva function (integral form).

    This matches OpenMC's _faddeeva implementation.
    """
    if np.angle(z) > 0:
        return wofz(z)
    else:
        return -np.conj(wofz(z.conjugate()))


def evaluate_with_temperature(
    E, poles, residues, AWR, temperature, poly_coeffs=None, fit_space="sqrt_E"
):
    """
    Evaluate cross sections with Doppler broadening.

    Same as evaluate_simple but includes Faddeeva-based broadening
    for finite temperatures.
    """

    K_BOLTZMANN = 8.617333262e-5  # eV/K

    E = np.atleast_1d(E)
    sqrtE = np.sqrt(E)

    # Z is the variable used for pole evaluation
    if fit_space == "sqrt_E":
        Z = sqrtE
    else:
        Z = E

    n_channels = residues.shape[0]
    xs = np.zeros((n_channels, len(E)))

    if temperature == 0.0:
        # Your existing 0K logic
        for i, s_val in enumerate(Z):
            denominators = s_val - poles
            for ch in range(n_channels):
                xs[ch, i] = np.sum((residues[ch] / denominators).real)
    else:
        # Finite temperature: Faddeeva
        sqrtkT = sqrt(K_BOLTZMANN * temperature)
        dopp = sqrt(AWR) / sqrtkT

        for i, z_val in enumerate(Z):
            for j, pole in enumerate(poles):
                # z_val is sqrt(E) if fit_space="sqrt_E"
                z_arg = (z_val - pole) * dopp
                # Faddeeva gives the broadened version of 1/(z-p)
                # Factor of sqrt(pi)*dopp comes from the broadening integral
                w_val = faddeeva(z_arg) * sqrt(pi) * dopp
                for ch in range(n_channels):
                    xs[ch, i] += (residues[ch, j] * w_val * (-1j)).real

    # Add polynomial (needs broadening too for finite T, but skip for now)
    # if poly_coeffs is not None:
    #     for ch, p in enumerate(poly_coeffs):
    #         if p is not None:
    #             xs[ch] += np.real(np.polyval(p, Z))

    return xs


def compute_error_metrics(xs_ref, xs_test, energy, atol=1e-10):
    """Compute error metrics between reference and test cross sections.

    Returns
    -------
    dict
        Dictionary with various error metrics
    """
    abs_err = np.abs(xs_test - xs_ref)

    # Relative error (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(xs_ref > atol, abs_err / xs_ref, 0.0)

    # Metrics
    metrics = {
        "max_abs_err": np.max(abs_err),
        "mean_abs_err": np.mean(abs_err),
        "rms_abs_err": np.sqrt(np.mean(abs_err**2)),
        "max_rel_err": np.max(rel_err),
        "mean_rel_err": np.mean(rel_err[xs_ref > atol]),
        "rms_rel_err": np.sqrt(np.mean(rel_err[xs_ref > atol] ** 2)),
        # Fraction of points within tolerance
        "frac_within_1pct": np.mean((rel_err < 0.01) | (xs_ref < atol)),
        "frac_within_5pct": np.mean((rel_err < 0.05) | (xs_ref < atol)),
    }

    return metrics


def print_metrics(metrics, channel_name, temperature):
    """Pretty print error metrics."""
    print(f"\n  {channel_name} at {temperature}K:")
    print(f"    Max relative error: {metrics['max_rel_err'] * 100:.3f}%")
    print(f"    RMS relative error: {metrics['rms_rel_err'] * 100:.3f}%")
    print(f"    Within 1%: {metrics['frac_within_1pct'] * 100:.1f}%")
    print(f"    Within 5%: {metrics['frac_within_5pct'] * 100:.1f}%")


# =============================================================================
# Plotting
# =============================================================================


def plot_comparison(
    energy, xs_ref, xs_test, channel_name, temperature, output_path, E_range=None
):
    """Generate comparison plot for a single channel/temperature."""

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Cross section comparison
    ax1.loglog(energy, xs_ref, "b-", label="NJOY reference", linewidth=1)
    ax1.loglog(
        energy, xs_test, "r--", label="AAA reconstruction", linewidth=0.8, alpha=0.8
    )
    ax1.set_ylabel("Cross section (b)")
    ax1.set_title(f"{channel_name} at {temperature}K")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative error
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(xs_test - xs_ref) / np.where(xs_ref > 1e-10, xs_ref, 1)
    ax2.semilogy(energy, rel_err, "k-", linewidth=0.5)
    ax2.axhline(0.01, color="blue", linestyle="--", label="1%")
    ax2.axhline(0.05, color="purple", linestyle="--", label="5%")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("Relative error")
    ax2.set_ylim(1e-6, 1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    if E_range:
        ax1.set_xlim(E_range)
        ax2.set_xlim(E_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_summary(all_metrics, temperatures, output_path):
    """Generate summary plot of errors across temperatures."""

    channels = ["elastic", "absorption", "fission"]
    colors = {"elastic": "blue", "absorption": "green", "fission": "red"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for channel in channels:
        if channel not in all_metrics:
            continue
        max_errs = [
            all_metrics[channel].get(T, {}).get("rms_rel_err", np.nan) * 100
            for T in temperatures
        ]
        ax.plot(
            temperatures,
            max_errs,
            "o-",
            label=channel,
            color=colors[channel],
            markersize=8,
        )

    ax.axhline(1.0, color="gray", linestyle="--", label="1% threshold")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Max relative error (%)")
    ax.set_title("Doppler Broadening Validation: RMS Rel Error vs Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Main validation routine
# =============================================================================


def load_mp_data(mp_file):
    """Load multipole data from pickle file."""
    with open(mp_file, "rb") as f:
        mp_data = pickle.load(f)

    # Handle multi-piece format (concatenate if needed)
    poles_list = mp_data["poles"]
    residues_list = mp_data["residues"]

    if isinstance(poles_list, list):
        # Multi-piece: concatenate
        poles = np.concatenate(poles_list)
        # Residues: each element is (n_channels, n_poles_in_piece)
        residues = np.hstack(residues_list)
    else:
        poles = poles_list
        residues = residues_list

    # Get poly_info if available
    poly_info_list = mp_data.get("poly_info_list", [None])
    if poly_info_list and poly_info_list[0] is not None:
        # For single piece, just get the poly_coeffs
        poly_coeffs = poly_info_list[0].get("poly_coeffs", None)
    else:
        poly_coeffs = None

    return {
        "poles": poles,
        "residues": residues,
        "poly_coeffs": poly_coeffs,
        "AWR": mp_data["AWR"],
        "E_min": mp_data["E_min"],
        "E_max": mp_data["E_max"],
        "name": mp_data["name"],
        "space": mp_data.get("space", "sqrt_E"),
    }


def validate_doppler(config):
    """Run the full Doppler broadening validation."""

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AAA-WMP Doppler Broadening Validation")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load AAA multipole data
    # -------------------------------------------------------------------------
    print("\n[1] Loading AAA multipole data...")
    mp_file = base_dir / config["mp_data_file"]
    mp_data = load_mp_data(mp_file)

    poles = mp_data["poles"]
    residues = mp_data["residues"]
    AWR = mp_data["AWR"]

    print(f"    Nuclide: {mp_data['name']}")
    print(f"    Number of poles: {len(poles)}")
    print(f"    Energy range: {mp_data['E_min']:.3e} to {mp_data['E_max']:.3e} eV")
    print(f"    AWR: {AWR:.4f}")

    # -------------------------------------------------------------------------
    # Step 2: Generate/load NJOY reference data
    # -------------------------------------------------------------------------
    print("\n[2] Generating NJOY reference data...")
    temperatures = config["temperatures"]

    ref_data = generate_temperature_references(
        endf_file=str(base_dir / config["endf_file"]),
        name=config["name"],
        temperatures=temperatures,
        cache_dir=str(base_dir / "data/input/NJOY_pickles"),
        njoy_error=config["njoy_error"],
        log=1,
    )

    print(f"    Temperatures: {ref_data['temperatures']}")

    # -------------------------------------------------------------------------
    # Step 3: Reconstruct and compare at each temperature
    # -------------------------------------------------------------------------
    print("\n[3] Validating at each temperature...")

    all_metrics = {"elastic": {}, "absorption": {}, "fission": {}}
    channel_names = ["elastic", "absorption", "fission"]

    for temp in [0.0] + temperatures:
        print(f"\n  Temperature: {temp}K")

        # Get reference data
        ref = ref_data[temp]
        energy = ref["energy"]

        # Apply energy bounds if specified
        E_min = config["E_min"] or mp_data["E_min"]
        E_max = config["E_max"] or mp_data["E_max"]
        mask = (energy >= E_min) & (energy <= E_max)
        energy = energy[mask]

        # Reference cross sections
        xs_ref = {
            "elastic": ref["elastic_xs"][mask]
            if len(ref["elastic_xs"]) > sum(mask)
            else ref["elastic_xs"],
            "absorption": ref["absorption_xs"][mask]
            if len(ref["absorption_xs"]) > sum(mask)
            else ref["absorption_xs"],
        }
        if ref["fissionable"]:
            xs_ref["fission"] = (
                ref["fission_xs"][mask]
                if len(ref["fission_xs"]) > sum(mask)
                else ref["fission_xs"]
            )

        # Reconstruct from poles
        xs_recon = evaluate_with_temperature(
            energy,
            poles,
            residues,
            AWR,
            temp,
            poly_coeffs=mp_data["poly_coeffs"],  # Add if you have them
            fit_space=mp_data["space"],  # Match your fitting space
        )

        # Compare each channel
        for i_ch, channel in enumerate(channel_names):
            if channel not in xs_ref:
                continue

            metrics = compute_error_metrics(xs_ref[channel], xs_recon[i_ch], energy)
            all_metrics[channel][temp] = metrics
            print_metrics(metrics, channel, temp)

            # Generate plot
            plot_path = output_dir / f"{config['name']}_{channel}_{int(temp)}K.png"
            plot_comparison(
                energy,
                xs_ref[channel],
                xs_recon[i_ch],
                channel,
                temp,
                plot_path,
                E_range=(E_min, E_max),
            )

    # -------------------------------------------------------------------------
    # Step 4: Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Generate summary plot
    plot_error_summary(
        all_metrics,
        [0.0] + temperatures,
        output_dir / f"{config['name']}_error_summary.png",
    )

    # Check pass/fail
    passed = True
    for channel, temp_metrics in all_metrics.items():
        for temp, metrics in temp_metrics.items():
            if metrics["rms_rel_err"] > config["rtol_threshold"]:
                print(
                    f"FAIL: {channel} at {temp}K exceeds RMS {config['rtol_threshold'] * 100}% threshold"
                )
                passed = False

    if passed:
        print("\nAll channels PASS validation thresholds!")
    else:
        print("\nSome channels FAILED validation thresholds.")

    print(f"\nPlots saved to: {output_dir}")

    return all_metrics


if __name__ == "__main__":
    validate_doppler(CONFIG)
