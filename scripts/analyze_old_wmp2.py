"""
Debug script to decompose a single WMP window evaluation.
Similar to the post-AAA analysis but for the windowed output.

Usage:
1. Set the paths and test energy
2. Run to see decomposition of poles, polynomial, c0 contributions
3. Compare against reference to identify where the error comes from
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

# Adjust path as needed
# sys.path.insert(0, str(Path(__file__).parent.parent))

# You'll need to import your WMP class
# from multipole_deplete_v3 import WindowedMultipole
# import openmc.data.vectfit as vf

K_BOLTZMANN = 8.617333262e-5  # eV/K
TEMPERATURE_LIMIT = 3000  # K


def decompose_window(wmp_data, E_test, T=0.0, channel_names=None):
    """
    Decompose WMP evaluation at a single energy point.
    Shows contribution from poles, polynomial, and c0 separately.
    """
    if channel_names is None:
        channel_names = ["elastic", "absorption", "fission"]

    sqrtE = np.sqrt(E_test)
    invE = 1.0 / E_test

    # Find window
    i_window = int((sqrtE - np.sqrt(wmp_data.E_min)) / wmp_data.spacing)
    i_window = max(0, min(i_window, len(wmp_data.windows) - 1))

    # Window boundaries
    inbegin = np.sqrt(wmp_data.E_min) + wmp_data.spacing * i_window
    inend = inbegin + wmp_data.spacing

    # Doppler extended boundaries
    awr = wmp_data.sqrtAWR**2
    alpha = awr / (K_BOLTZMANN * TEMPERATURE_LIMIT)
    if i_window == 0 or np.sqrt(alpha) * inbegin < 4.0:
        e_start = inbegin**2
    else:
        e_start = max(wmp_data.E_min, (np.sqrt(alpha) * inbegin - 4.0) ** 2 / alpha)
    e_end = min(wmp_data.E_max, (np.sqrt(alpha) * inend + 4.0) ** 2 / alpha)

    print("=" * 70)
    print(f"WMP WINDOW DECOMPOSITION AT E = {E_test:.4f} eV")
    print("=" * 70)

    print(f"\nWindow {i_window}:")
    print(f"  Inner bounds: [{inbegin**2:.2f}, {inend**2:.2f}] eV")
    print(f"  Doppler extended: [{e_start:.2f}, {e_end:.2f}] eV")

    # Get poles for this window
    start_pole = wmp_data.windows[i_window, 0] - 1  # Convert to 0-based
    end_pole = wmp_data.windows[i_window, 1]
    n_poles = end_pole - start_pole

    print(f"  Poles: indices [{start_pole}, {end_pole}) = {n_poles} poles")

    # Extract poles and residues
    window_poles = wmp_data.data[start_pole:end_pole, 0]
    n_channels = wmp_data.data.shape[1] - 1
    window_residues = wmp_data.data[start_pole:end_pole, 1 : 1 + n_channels]

    print(f"  Number of channels: {n_channels}")

    # Compute pole contributions (0K)
    xs_poles = np.zeros(n_channels)
    for i_pole in range(n_poles):
        pole = window_poles[i_pole]
        psi_chi = -1j / (pole - sqrtE)
        c_temp = psi_chi * invE
        for i_ch in range(n_channels):
            residue = window_residues[i_pole, i_ch]
            xs_poles[i_ch] += np.real(residue * c_temp)

    # Compute polynomial contribution
    fit_order = wmp_data.curvefit.shape[1] - 1
    xs_poly = np.zeros(n_channels)
    temp = invE
    for i_poly in range(fit_order + 1):
        for i_ch in range(n_channels):
            xs_poly[i_ch] += wmp_data.curvefit[i_window, i_poly, i_ch] * temp
        temp *= sqrtE

    # Compute c0 contribution (if stored)
    xs_c0 = np.zeros(n_channels)
    has_pr_constant = (
        hasattr(wmp_data, "pr_constant") and wmp_data.pr_constant is not None
    )
    if has_pr_constant:
        for i_ch in range(n_channels):
            xs_c0[i_ch] = wmp_data.pr_constant[i_window, i_ch] * invE

    # Total
    xs_total = xs_poles + xs_poly + xs_c0

    # Also get the result from _evaluate for comparison
    xs_evaluate = np.array(wmp_data._evaluate(E_test, T))

    print("\nDECOMPOSITION (barns):")
    print(
        f"{'Channel':<12} {'Poles':>14} {'Polynomial':>14} {'c0/E':>14} {'Total':>14} {'_evaluate':>14}"
    )
    print("-" * 84)
    for i_ch in range(min(n_channels, len(channel_names))):
        ch = channel_names[i_ch]
        print(
            f"{ch:<12} {xs_poles[i_ch]:>14.6e} {xs_poly[i_ch]:>14.6e} {xs_c0[i_ch]:>14.6e} {xs_total[i_ch]:>14.6e} {xs_evaluate[i_ch]:>14.6e}"
        )

    # Check if manual matches _evaluate
    print(
        f"\nManual vs _evaluate match: {np.allclose(xs_total[: len(xs_evaluate)], xs_evaluate)}"
    )

    # Print curvefit coefficients
    print(f"\nCURVEFIT COEFFICIENTS (window {i_window}, order {fit_order}):")
    for i_poly in range(fit_order + 1):
        if i_poly == 0:
            term = "1/E      "
        elif i_poly == 1:
            term = "1/sqrt(E)"
        elif i_poly == 2:
            term = "const    "
        else:
            term = f"E^{(i_poly - 2) / 2:<6.1f}"
        coeffs = wmp_data.curvefit[i_window, i_poly, :]
        print(f"  {term}: {coeffs}")

    # Print c0 if stored
    if has_pr_constant:
        print(f"\nPR_CONSTANT (window {i_window}):")
        print(f"  {wmp_data.pr_constant[i_window]}")
    else:
        print("\nPR_CONSTANT: Not stored in WMP")

    # Show nearest poles
    print(f"\nNEAREST POLES to sqrt(E)={sqrtE:.4f}:")
    distances = np.abs(window_poles.real - sqrtE)
    sorted_idx = np.argsort(distances)
    for i in range(min(5, n_poles)):
        idx = sorted_idx[i]
        pole = window_poles[idx]
        dist = distances[idx]
        res = window_residues[idx]
        print(f"  Pole {idx}: {pole:.6f} (dist={dist:.4f}), residues={res}")

    print("=" * 70)

    return {
        "i_window": i_window,
        "n_poles": n_poles,
        "xs_poles": xs_poles,
        "xs_poly": xs_poly,
        "xs_c0": xs_c0,
        "xs_total": xs_total,
        "xs_evaluate": xs_evaluate,
        "curvefit": wmp_data.curvefit[i_window],
        "pr_constant": wmp_data.pr_constant[i_window] if has_pr_constant else None,
    }


def plot_window_decomposition(
    wmp_data,
    i_window,
    reference_data=None,
    n_points=500,
    channel_idx=0,
    channel_name="elastic",
    output_path=None,
):
    """
    Plot the decomposition across an entire window's energy range.
    """
    # Window boundaries
    inbegin = np.sqrt(wmp_data.E_min) + wmp_data.spacing * i_window
    inend = inbegin + wmp_data.spacing

    awr = wmp_data.sqrtAWR**2
    alpha = awr / (K_BOLTZMANN * TEMPERATURE_LIMIT)
    if i_window == 0 or np.sqrt(alpha) * inbegin < 4.0:
        e_start = inbegin**2
    else:
        e_start = max(wmp_data.E_min, (np.sqrt(alpha) * inbegin - 4.0) ** 2 / alpha)
    e_end = min(wmp_data.E_max, (np.sqrt(alpha) * inend + 4.0) ** 2 / alpha)

    E_grid = np.linspace(e_start, e_end, n_points)

    # Get poles for this window
    start_pole = wmp_data.windows[i_window, 0] - 1
    end_pole = wmp_data.windows[i_window, 1]
    n_poles = end_pole - start_pole
    window_poles = wmp_data.data[start_pole:end_pole, 0]
    window_residues = wmp_data.data[start_pole:end_pole, 1 + channel_idx]

    fit_order = wmp_data.curvefit.shape[1] - 1
    has_pr_constant = (
        hasattr(wmp_data, "pr_constant") and wmp_data.pr_constant is not None
    )

    # Compute decomposition across energy range
    xs_poles = np.zeros(n_points)
    xs_poly = np.zeros(n_points)
    xs_c0 = np.zeros(n_points)
    xs_total = np.zeros(n_points)
    xs_evaluate = np.zeros(n_points)

    for i, E in enumerate(E_grid):
        sqrtE = np.sqrt(E)
        invE = 1.0 / E

        # Poles
        for j in range(n_poles):
            pole = window_poles[j]
            residue = window_residues[j]
            psi_chi = -1j / (pole - sqrtE)
            xs_poles[i] += np.real(residue * psi_chi * invE)

        # Polynomial
        temp = invE
        for i_poly in range(fit_order + 1):
            xs_poly[i] += wmp_data.curvefit[i_window, i_poly, channel_idx] * temp
            temp *= sqrtE

        # c0
        if has_pr_constant:
            xs_c0[i] = wmp_data.pr_constant[i_window, channel_idx] * invE

        xs_total[i] = xs_poles[i] + xs_poly[i] + xs_c0[i]

        # _evaluate result
        result = wmp_data._evaluate(E, 0.0)
        xs_evaluate[i] = result[channel_idx]

    # Get reference if provided
    xs_ref = None
    if reference_data is not None:
        ref_E = reference_data.get("energy")
        ref_xs = reference_data.get(channel_name)
        if ref_E is not None and ref_xs is not None:
            xs_ref = np.interp(E_grid, ref_E, ref_xs)

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Cross sections
    ax1 = axes[0]
    if xs_ref is not None:
        ax1.semilogy(
            E_grid, np.maximum(xs_ref, 1e-10), "k-", label="Reference", linewidth=2
        )
    ax1.semilogy(
        E_grid,
        np.maximum(np.abs(xs_total), 1e-10),
        "r--",
        label="WMP total (manual)",
        linewidth=1.5,
    )
    ax1.semilogy(
        E_grid,
        np.maximum(np.abs(xs_evaluate), 1e-10),
        "b:",
        label="WMP _evaluate",
        linewidth=1.5,
    )
    ax1.set_ylabel(f"{channel_name} XS (barns)")
    ax1.set_title(
        f"Window {i_window}: [{inbegin**2:.2f}, {inend**2:.2f}] eV (inner), {n_poles} poles"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Decomposition
    ax2 = axes[1]
    ax2.plot(E_grid, xs_poles, "g-", label="Poles", linewidth=1.5)
    ax2.plot(E_grid, xs_poly, "m-", label="Polynomial", linewidth=1.5)
    if has_pr_constant and np.any(xs_c0 != 0):
        ax2.plot(E_grid, xs_c0, "c-", label="c0/E", linewidth=1.5)
    ax2.plot(E_grid, xs_total, "r--", label="Total", linewidth=1.5)
    ax2.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax2.set_ylabel("Cross section (barns)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Relative error
    ax3 = axes[2]
    if xs_ref is not None:
        mask = xs_ref > 1e-10
        rel_err = np.full_like(E_grid, np.nan)
        rel_err[mask] = np.abs(xs_total[mask] - xs_ref[mask]) / xs_ref[mask] * 100
        ax3.semilogy(E_grid, rel_err, "r-", label="Relative error", linewidth=1.5)
        ax3.axhline(0.1, color="k", linestyle="--", alpha=0.5, label="0.1%")
        ax3.axhline(1.0, color="k", linestyle=":", alpha=0.5, label="1%")
        ax3.set_ylabel("Relative error (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    ax3.set_xlabel("Energy (eV)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()

    return {
        "E_grid": E_grid,
        "xs_poles": xs_poles,
        "xs_poly": xs_poly,
        "xs_c0": xs_c0,
        "xs_total": xs_total,
        "xs_evaluate": xs_evaluate,
        "xs_ref": xs_ref,
    }


def compare_with_reference(wmp_data, reference_data, E_test, channel_names=None):
    """
    Compare WMP result with reference at a single energy.
    """
    if channel_names is None:
        channel_names = ["elastic", "absorption", "fission"]

    # Get WMP result
    result = decompose_window(wmp_data, E_test, T=0.0, channel_names=channel_names)

    # Get reference
    ref_E = reference_data.get("energy")
    idx = np.argmin(np.abs(ref_E - E_test))

    print("\nCOMPARISON WITH REFERENCE:")
    print(f"{'Channel':<12} {'WMP':>14} {'Reference':>14} {'Error %':>12}")
    print("-" * 54)

    for i_ch, ch in enumerate(channel_names):
        ref_xs = reference_data.get(ch)
        if ref_xs is not None and i_ch < len(result["xs_total"]):
            ref_val = ref_xs[idx]
            wmp_val = result["xs_total"][i_ch]
            if ref_val != 0:
                err = (wmp_val - ref_val) / ref_val * 100
            else:
                err = float("inf") if wmp_val != 0 else 0
            print(f"{ch:<12} {wmp_val:>14.6e} {ref_val:>14.6e} {err:>12.3f}")

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ADJUST THESE PATHS
    wmp_h5_path = "data/output/WMP_Lib_viii.0/U238/U238.h5"
    njoy_pickle_path = "data/input/NJOY_pickles/U238_NJOY.pickle"
    name = "U238"

    # Test energies
    E_test = 19065.0  # The problematic energy from your debug

    # Load WMP
    from multipole_deplete_v3 import WindowedMultipole

    wmp = WindowedMultipole(name)
    wmp_data = wmp.from_hdf5(wmp_h5_path)

    # Load reference
    with open(njoy_pickle_path, "rb") as f:
        njoy_data = pickle.load(f)
    reference_data = {
        "energy": njoy_data.energy["0K"],
        "elastic": njoy_data[2].xs["0K"](njoy_data.energy["0K"]),
        "absorption": njoy_data[27].xs["0K"](njoy_data.energy["0K"]),
        "fission": njoy_data[18].xs["0K"](njoy_data.energy["0K"])
        if 18 in njoy_data.reactions
        else None,
    }

    # Run decomposition
    result = decompose_window(wmp_data, E_test)

    # Compare with reference
    compare_with_reference(wmp_data, reference_data, E_test)

    # Plot the window containing E_test
    i_window = int((np.sqrt(E_test) - np.sqrt(wmp_data.E_min)) / wmp_data.spacing)
    plot_window_decomposition(
        wmp_data,
        i_window,
        reference_data,
        channel_idx=0,
        channel_name="elastic",
        output_path=f"window_{i_window}_decomposition.png",
    )
