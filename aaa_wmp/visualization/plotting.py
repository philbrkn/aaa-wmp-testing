import os

import matplotlib.pyplot as plt
import numpy as np

# from ..core.conversion import evaluate_simple


def plot_single_channel(
    E,
    original,
    reconstructed,
    channel_name,
    symbol,
    name,
    path_out=None,
    plot_type="loglog",
    show_error=False,
    error_type="relative",
    poles=None,
):
    """
    Plot a single reaction channel comparison.

    Parameters
    ----------
    E : array-like
        Energy grid
    original : array-like
        Original cross section data
    reconstructed : array-like
        Reconstructed cross section data
    channel_name : str
        Name of the channel (e.g., 'elastic', 'absorption', 'fission')
    symbol : str
        Symbol for the cross section (e.g., 'σ_s', 'σ_a', 'σ_f')
    color : str
        Color for the original data line
    name : str
        Nuclide name
    path_out : str, optional
        Directory to save plot
    plot_type : str
        Plot scale type ('loglog', 'semilogx', 'semilogy', 'linear')
    show_error : bool
        Whether to show error on secondary y-axis
    error_type : str
        Type of error to plot: 'relative', 'absolute', or 'remainder'
    """
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    # Choose plotting function
    if plot_type == "loglog":
        plot_func = ax1.loglog
    elif plot_type == "semilogx":
        plot_func = ax1.semilogx
    elif plot_type == "semilogy":
        plot_func = ax1.semilogy
    else:  # linear
        plot_func = ax1.plot

    # Plot data
    plot_func(E, original, "b-", label=f"Original {symbol}", linewidth=2)
    plot_func(E, reconstructed, "r--", label="Reconstructed", linewidth=2)

    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("Cross section (b)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    # ax1.grid(which="major", axis="x", linestyle=":", linewidth=1, zorder=0)
    # ax1.grid(which="major", axis="y", linestyle="-", linewidth=0.3, zorder=0)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax1.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    # Add pole vertical lines if provided
    if poles is not None:
        poles = np.array(poles)
        # Extract real parts of poles
        pole_energies = np.real(poles)

        # Filter poles to only show those within the energy range
        E_min, E_max = np.min(E), np.max(E)
        visible_poles = pole_energies[
            (pole_energies >= E_min) & (pole_energies <= E_max)
        ]
        pole_color = "black"
        pole_alpha = 0.7
        pole_linewidth = 1.5
        pole_linestyle = "--"
        for i, pole_energy in enumerate(visible_poles):
            # Add vertical line on main plot
            ax1.axvline(
                pole_energy,
                color=pole_color,
                alpha=pole_alpha,
                linewidth=pole_linewidth,
                linestyle=pole_linestyle,
                label="Poles" if i == 0 else "",  # Only label first pole for legend
            )

    # Handle error plotting on secondary axis
    if show_error:
        # Calculate error based on type
        original = np.array(original)
        reconstructed = np.array(reconstructed)

        if error_type == "relative":
            # Avoid division by zero
            mask = original != 0
            error = np.full_like(original, np.nan)
            error[mask] = (
                np.abs((reconstructed[mask] - original[mask]) / original[mask]) * 100
            )
            error_label = "Relative Error (%)"
            error_color = "black"
            if np.any(mask):
                max_err = np.max(np.abs(error[mask]))
                rms_err = np.sqrt(np.mean(error[mask] ** 2))
                print(
                    f"{channel_name:<12} | Max rel error = {max_err:.2e}%  | RMS rel error = {rms_err:.2e}%"
                )
            ax2.set_ylim(1e-7, 1e1)
        elif error_type == "absolute":
            error = np.abs(reconstructed - original)
            error_label = "Absolute Error (b)"
            error_color = "black"

            max_err = np.max(np.abs(error))
            rms_err = np.sqrt(np.mean(error**2))
            print(
                f"{channel_name:<12} | Max abs error = {max_err:.2e}  | RMS abs error = {rms_err:.2e}"
            )
            ax2.set_ylim(1e-10, 1e2)
        elif error_type == "remainder":
            error = reconstructed - original
            error_label = "Remainder (b)"
            error_color = "black"
        else:
            raise ValueError(
                "error_type must be 'relative', 'absolute', or 'remainder'"
            )
        # Plot error with appropriate scale
        if plot_type in ["loglog", "semilogy"]:
            # For log y-scale, we need positive values
            if error_type == "remainder":
                # For remainder, we might have negative values, so use regular plot
                ax2.plot(
                    E,
                    error,
                    color=error_color,
                    linewidth=1.5,
                    label=error_label,
                )
            else:
                # For absolute and relative errors (always positive), we can use semilogy
                ax2.semilogy(
                    E,
                    error,
                    color=error_color,
                    linewidth=1.5,
                    label=error_label,
                )
        else:
            ax2.plot(
                E,
                error,
                color=error_color,
                linewidth=1.5,
                label=error_label,
            )

        ax2.set_ylabel(error_label, color=error_color)
        # ax2.tick_params(axis="y", labelcolor=error_color)
        ax2.grid(True, which="both", alpha=0.3)
        # horizontal lines
        ax2.grid(which="major", axis="y", linestyle="--", linewidth=1)
        # vertical lines.
        ax2.grid(which="major", axis="x", linestyle="-", linewidth=0.8, alpha=0.7)
        ax2.grid(which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
        from matplotlib.ticker import LogLocator

        ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
        ax2.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend()

    if path_out:
        filename = f"{name}_{channel_name}_reconstruction"
        if show_error:
            filename += f"_{error_type}_error"
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_out, f"{filename}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def plot_reconstruction(
    E,
    original_data,
    poles,
    residues,
    name="Nuclide",
    path_out=None,
    plot_type="loglog",
    show_error=False,
    error_type="relative",
    poly_info=None,
    fit_space="sqrt_E",
):
    """
    Plot original ACE data vs reconstructed from poles/residues.
    Creates separate plots for elastic, absorption, and fission.

    Parameters
    ----------
    E : array-like
        Energy grid (eV)
    original_data : dict
        Original cross sections with keys:
        - 'sigma_s' or 'elastic': elastic scattering
        - 'sigma_a' or 'absorption': absorption
        - 'sigma_f' or 'fission': fission (optional)
    poles : array-like
        Complex poles from proper_rational or other source
    residues : array-like
        Residues from proper_rational (can be 2D array or list)
    name : str
        Nuclide name for plot titles
    path_out : str, optional
        Directory to save plots (if None, shows plots)
    plot_type : str
        Plotting scale: "loglog", "semilogx", "semilogy", or "linear"
    poly_info : dict or list, optional
        Polynomial coefficients info from proper_rational.
        Can be either:
        - dict with 'poly_coeffs' key containing list of polynomial coefficients
        - list of polynomial coefficients directly

    Returns
    -------
    dict
        Reconstructed cross sections
    """
    # temp load NJOY data
    # temp = 1500
    temp = 0
    from ..io.njoy_interface import generate_temperature_references

    ref_data = generate_temperature_references(
        endf_file="data/input/ENDF/ENDF-VIII-data/n-092_U_238.endf",
        name="U238",
        temperatures=[294, 600, 900, 1200, 1500],
        cache_dir="data/input/NJOY_pickles",
        njoy_error=5e-4,
        log=1,
    )
    ref = ref_data[temp]
    energy = ref["energy"]
    # Apply energy bounds if specified
    E_min, E_max = np.min(E), np.max(E)
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

    Z = np.sqrt(energy) if fit_space == "sqrt_E" else energy

    # USe vf.evaluate instaed?
    # from openmc.data.vectfit import evaluate
    #
    # xs_recon = evaluate(np.sqrt(E), poles, residues)

    from ..core.conversion import (
        # build_wmp_poles,
        evaluate_openmc_T,
        fit_pseudopoles_adaptive_0K,
        # refit_residues_openmc,
        # refit_residues_realpart,
        to_wmp_form,
    )

    mp_poles, mp_residues = to_wmp_form(poles, residues)
    print(f"Number of poles is {len(mp_poles)}")
    awr = np.sqrt(ref_data["AWR"])
    # Pole-only reconstruction at 0 K
    xs_poles_0K = evaluate_openmc_T(
        energy, 0.0, mp_poles, mp_residues / 1j, sqrtAWR=awr, poly_coeffs=None
    )

    def interp_xs(energy_src, xs_src, energy_dst):
        return np.interp(energy_dst, energy_src, xs_src)

    energy_0K = ref_data[0]["energy"]
    energy_eval = energy
    xs_ref_0K_interp = np.vstack(
        [
            interp_xs(energy_0K, ref_data[0]["elastic_xs"], energy_eval),
            interp_xs(energy_0K, ref_data[0]["absorption_xs"], energy_eval),
            interp_xs(energy_0K, ref_data[0]["fission_xs"], energy_eval),
        ]
    )
    # Background remainder (pointwise)
    xs_poles_0K = np.asarray(xs_poles_0K)
    remainder = xs_ref_0K_interp - xs_poles_0K

    pp, res_pp = fit_pseudopoles_adaptive_0K(
        energy,
        remainder,
        xs_0K_recon=xs_poles_0K,  # <-- "0K reconstruction" denominator
        max_poles=2,
        rtol=1e-8,
        verbose=False,
    )
    # Evaluate pseudo background using the SAME kernel used in the fit: 1/(Z - p)
    if pp.size > 0:
        Cpp = 1.0 / (Z[:, None] - pp[None, :])  # (n, npp)
        bg_0K = res_pp @ Cpp.T  # (k, n)
    else:
        bg_0K = np.zeros_like(remainder)

    # def print_avg_background(xs_bg, names=("elastic", "absorption", "fission")):
    #     xs_bg = np.asarray(xs_bg)
    #     avg = xs_bg.mean(axis=1)
    #     for i, name in enumerate(names):
    #         print(f"avg background {name:10s} = {avg[i]:.6e}")
    #
    # print_avg_background(xs_bg)

    # Reconstruction at any temperature + same background
    xs_recon = evaluate_openmc_T(
        energy, temp, mp_poles, mp_residues / 1j, sqrtAWR=awr, poly_coeffs=None
    )
    xs_recon = np.asarray(xs_recon) + bg_0K

    # xs_recon += remainder
    # xs_recon += poly_vals

    # Define channels to plot
    channels = [
        {
            "name": "elastic",
            "symbol": "σ_s",
            "original": xs_ref["elastic"],
            "reconstructed": xs_recon[0],
        },
        {
            "name": "absorption",
            "symbol": "σ_a",
            "original": xs_ref["absorption"],
            "reconstructed": xs_recon[1],
        },
        {
            "name": "fission",
            "symbol": "σ_f",
            "original": xs_ref["fission"],
            "reconstructed": xs_recon[2],
        },
    ]

    if path_out:
        os.makedirs(path_out, exist_ok=True)

    # Plot each channel using the modular function
    for channel in channels:
        # Skip if no original data or reconstructed data
        if channel["original"] is None or channel["reconstructed"] is None:
            continue

        plot_single_channel(
            Z,
            channel["original"],
            channel["reconstructed"],
            channel["name"],
            channel["symbol"],
            name,
            path_out,
            plot_type,
            show_error=show_error,
            error_type=error_type,
            poles=poles,
        )


def plot_wmp_validation(
    E,
    original,
    reconstructed,
    c0,
    channel_name,
    symbol,
    name,
    path_out=None,
    rtol=1e-3,
    atol=1e-5,
    window_bounds=None,
):
    """
    Plot WMP reconstruction validation for a single channel.

    Parameters
    ----------
    E : array-like
        Energy grid (eV)
    original : array-like
        Reference cross section (barns)
    reconstructed : array-like
        WMP reconstruction (barns)
    c0 : float
        Constant term in E*sigma space for this channel
    channel_name : str
        Name of the channel (elastic, absorption, fission)
    symbol : str
        LaTeX symbol for the cross section
    name : str
        Nuclide name
    path_out : str, optional
        Directory to save plot
    rtol : float
        Relative tolerance (default 1e-3 = 0.1%)
    atol : float
        Absolute tolerance in barns (default 1e-5)
    window_bounds : list of tuples, optional
        List of (E_left, E_right) for each window
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute errors
    abs_err = np.abs(reconstructed - original)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = abs_err / np.abs(original)
        rel_err[~np.isfinite(rel_err)] = 0

    # Points satisfying tolerance
    satisfied = (rel_err < rtol) | (abs_err < atol)
    satisfaction_pct = 100 * np.sum(satisfied) / len(satisfied)

    # Compute contributions
    c0_contribution = c0 / E  # c0 is in E*sigma space
    pole_only = reconstructed - c0_contribution
    remainder = original - pole_only  # What the constant needs to capture

    # ========== Top Left: Original vs Reconstruction ==========
    ax = axes[0, 0]
    ax.loglog(E, original, "b-", label=f"Reference {symbol}", linewidth=1.5)
    ax.loglog(
        E, reconstructed, "r--", label="WMP Reconstruction", linewidth=1.0, alpha=0.8
    )

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(f"{name} {channel_name.capitalize()}: Reference vs WMP")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # ========== Top Right: Remainder vs c0/E ==========
    ax = axes[0, 1]
    ax.semilogx(E, remainder, "b-", label="Remainder (ref - poles)", linewidth=1.0)
    ax.semilogx(E, c0_contribution, "r--", label=f"c₀/E (c₀={c0:.3e})", linewidth=1.5)

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(f"{channel_name.capitalize()}: Remainder vs Polynomial Background")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # ========== Bottom Left: Relative Error ==========
    ax = axes[1, 0]
    ax.semilogy(E, rel_err * 100, "b-", linewidth=0.8, label="Relative error")
    ax.axhline(
        rtol * 100,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"rtol = {rtol * 100:.1f}%",
    )

    # Shade where absolute tolerance saves us
    abs_dominated = (abs_err < atol) & (rel_err >= rtol)
    if np.any(abs_dominated):
        for start, end in _get_contiguous_regions(abs_dominated):
            ax.axvspan(E[start], E[end - 1], alpha=0.2, color="green")
        ax.fill_between([], [], alpha=0.2, color="green", label="Within atol only")

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(
        f"{channel_name.capitalize()}: Relative Error ({satisfaction_pct:.1f}% within tolerance)"
    )
    ax.set_xscale("log")
    ax.set_ylim(1e-6, 1e2)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # ========== Bottom Right: Absolute Error ==========
    ax = axes[1, 1]
    ax.semilogy(E, abs_err, "b-", linewidth=0.8, label="Absolute error")
    ax.axhline(
        atol, color="r", linestyle="--", linewidth=1.5, label=f"atol = {atol:.0e} barns"
    )

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absolute error (barns)")
    ax.set_title(f"{channel_name.capitalize()}: Absolute Error")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"{name} {channel_name.capitalize()} WMP Validation",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if path_out:
        filename = f"{name}_{channel_name}_wmp_validation.png"
        filepath = os.path.join(path_out, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()
    else:
        plt.show()

    return {
        "satisfaction_pct": satisfaction_pct,
        "max_rel_err": np.max(rel_err[abs_err > atol]) if np.any(abs_err > atol) else 0,
        "max_abs_err": np.max(abs_err),
    }


def _get_contiguous_regions(mask):
    """Helper to find contiguous True regions in a boolean mask."""
    regions = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions


def plot_wmp_temperature_validation(
    E,
    ref_0K,
    ref_T,
    recon_0K,
    recon_T,
    temperature,
    channel_name,
    symbol,
    name,
    path_out=None,
    rtol=1e-3,
    atol=1e-5,
    window_bounds=None,
):
    """
    Plot WMP reconstruction validation at finite temperature.

    Parameters
    ----------
    E : array-like
        Energy grid (eV)
    ref_0K : array-like
        Reference cross section at 0K (barns)
    ref_T : array-like
        Reference cross section at temperature T (barns)
    recon_0K : array-like
        WMP reconstruction at 0K (barns)
    recon_T : array-like
        WMP reconstruction at temperature T (barns)
    temperature : float
        Temperature in Kelvin
    channel_name : str
        Name of the channel
    symbol : str
        LaTeX symbol for the cross section
    name : str
        Nuclide name
    path_out : str, optional
        Directory to save plot
    rtol : float
        Relative tolerance (default 1e-3 = 0.1%)
    atol : float
        Absolute tolerance in barns (default 1e-5)
    window_bounds : list of tuples, optional
        List of (E_left, E_right) for each window
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute errors at temperature T
    abs_err_T = np.abs(recon_T - ref_T)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_T = abs_err_T / np.abs(ref_T)
        rel_err_T[~np.isfinite(rel_err_T)] = 0

    # Compute errors at 0K for comparison
    abs_err_0K = np.abs(recon_0K - ref_0K)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_0K = abs_err_0K / np.abs(ref_0K)
        rel_err_0K[~np.isfinite(rel_err_0K)] = 0

    satisfied_T = (rel_err_T < rtol) | (abs_err_T < atol)
    satisfaction_pct_T = 100 * np.sum(satisfied_T) / len(satisfied_T)

    satisfied_0K = (rel_err_0K < rtol) | (abs_err_0K < atol)
    satisfaction_pct_0K = 100 * np.sum(satisfied_0K) / len(satisfied_0K)

    # ========== Top Left: Reference vs Reconstruction at T ==========
    ax = axes[0, 0]
    ax.loglog(
        E, ref_T, "b-", label=f"Reference {symbol} @ {temperature}K", linewidth=1.5
    )
    ax.loglog(
        E, recon_T, "r--", label=f"WMP @ {temperature}K", linewidth=1.0, alpha=0.8
    )

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(f"{channel_name.capitalize()}: Reference vs WMP @ {temperature}K")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # ========== Top Right: Doppler Broadening Effect ==========
    ax = axes[0, 1]
    ax.loglog(
        E, ref_0K, color="black", label="Reference @ 0K", linewidth=1.0, alpha=0.6
    )
    ax.loglog(E, ref_T, "b--", label=f"Reference @ {temperature}K", linewidth=1.5)
    ax.loglog(E, recon_T, "r:", label=f"WMP @ {temperature}K", linewidth=1.5)

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(f"{channel_name.capitalize()}: Doppler Broadening Effect")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # ========== Bottom Left: Relative Error Comparison ==========
    ax = axes[1, 0]
    ax.semilogy(
        E,
        rel_err_0K * 100,
        color="gray",
        linewidth=0.8,
        alpha=0.7,
        label="Rel error @ 0K",
    )
    ax.semilogy(
        E, rel_err_T * 100, "b-", linewidth=0.8, label=f"Rel error @ {temperature}K"
    )
    ax.axhline(
        rtol * 100,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"rtol = {rtol * 100:.1f}%",
    )

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(
        f"{channel_name.capitalize()}: Relative Error (0K: {satisfaction_pct_0K:.1f}%, {temperature}K: {satisfaction_pct_T:.1f}% within tol)"
    )
    ax.set_xscale("log")
    ax.set_ylim(1e-6, 1e2)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # ========== Bottom Right: Absolute Error Comparison ==========
    ax = axes[1, 1]
    ax.semilogy(
        E, abs_err_0K, color="gray", linewidth=0.8, alpha=0.7, label="Abs error @ 0K"
    )
    ax.semilogy(E, abs_err_T, "b-", linewidth=0.8, label=f"Abs error @ {temperature}K")
    ax.axhline(
        atol, color="r", linestyle="--", linewidth=1.5, label=f"atol = {atol:.0e} barns"
    )

    if window_bounds is not None:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.8, linewidth=1.8)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.8, linewidth=1.8)

    # Points failing rtol but saved by atol
    saved_by_atol_T = (rel_err_T >= rtol) & (abs_err_T < atol)
    # Points failing both tolerances
    failing_both_T = (rel_err_T >= rtol) & (abs_err_T >= atol)
    if np.any(saved_by_atol_T):
        ax.scatter(
            E[saved_by_atol_T],
            abs_err_T[saved_by_atol_T],
            c="green",
            s=10,
            alpha=0.7,
            zorder=5,
            label="Saved by atol",
        )

    # Highlight points failing both tolerances (red)
    if np.any(failing_both_T):
        ax.scatter(
            E[failing_both_T],
            abs_err_T[failing_both_T],
            c="red",
            s=10,
            alpha=0.7,
            zorder=5,
            label="Failing both",
        )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absolute error (barns)")
    ax.set_title(f"{channel_name.capitalize()}: Absolute Error")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"{name} {channel_name.capitalize()} WMP Validation @ {temperature}K",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if path_out:
        filename = f"{name}_{channel_name}_wmp_validation_{temperature}K.png"
        filepath = os.path.join(path_out, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        # print(f'  Saved: {filepath}')
        plt.close()
    else:
        plt.show()

    return {
        "satisfaction_pct_0K": satisfaction_pct_0K,
        "satisfaction_pct_T": satisfaction_pct_T,
        "max_rel_err_T": np.max(rel_err_T[abs_err_T > atol])
        if np.any(abs_err_T > atol)
        else 0,
        "max_abs_err_T": np.max(abs_err_T),
    }


def plot_decomposition(
    E,
    original,
    pole_contribution,
    pseudo_contribution,
    remainder,
    channel_name,
    symbol,
    name,
    path_out=None,
    plot_type="loglog",
):
    """
    Plot the decomposition of cross section into pole, pseudo-pole, and remainder contributions.

    Parameters
    ----------
    E : array-like
        Energy grid
    original : array-like
        Original cross section data
    pole_contribution : array-like
        Contribution from physical poles
    pseudo_contribution : array-like
        Contribution from pseudo poles
    remainder : array-like
        Remaining difference (original - pole - pseudo)
    channel_name : str
        Name of the channel
    symbol : str
        Symbol for the cross section
    name : str
        Nuclide name
    path_out : str, optional
        Directory to save plot
    plot_type : str
        Plot scale type
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Choose plotting function based on type
    def get_plot_func(ax):
        if plot_type == "loglog":
            return ax.loglog
        elif plot_type == "semilogx":
            return ax.semilogx
        elif plot_type == "semilogy":
            return ax.semilogy
        else:
            return ax.plot

    # Top left: Original vs total reconstruction
    ax = axes[0, 0]
    plot_func = get_plot_func(ax)
    total_recon = pole_contribution + pseudo_contribution
    plot_func(E, original, "b-", label=f"Original {symbol}", linewidth=2)
    plot_func(E, total_recon, "r--", label="Total Reconstruction", linewidth=1.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (b)")
    ax.set_title(f"{channel_name.capitalize()}: Original vs Reconstruction")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Top right: Pole contribution only
    ax = axes[0, 1]
    plot_func = get_plot_func(ax)
    plot_func(E, original, "b-", label=f"Original {symbol}", linewidth=2, alpha=0.5)
    plot_func(
        E, pole_contribution, color="gray", label="Pole Contribution", linewidth=1.5
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (b)")
    ax.set_title(f"{channel_name.capitalize()}: Physical Pole Contribution")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Bottom left: Pseudo-pole contribution
    ax = axes[1, 0]
    y_min = min(pseudo_contribution.min(), remainder.min())
    y_max = max(pseudo_contribution.max(), remainder.max())
    # Pseudo contribution can be negative, so use semilogx for energy axis only
    if plot_type in ["loglog", "semilogx"]:
        ax.semilogx(
            E,
            pseudo_contribution,
            "m-",
            label="Pseudo-pole Contribution",
            linewidth=1.5,
        )
    else:
        ax.plot(
            E,
            pseudo_contribution,
            "m-",
            label="Pseudo-pole Contribution",
            linewidth=1.5,
        )
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (b)")
    ax.set_title(f"{channel_name.capitalize()}: Pseudo-pole Contribution (Background)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(y_min, y_max)

    # Bottom right: Final remainder
    ax = axes[1, 1]
    if plot_type in ["loglog", "semilogx"]:
        ax.semilogx(E, remainder, "k-", label="Remainder (Error)", linewidth=1.5)
    else:
        ax.plot(E, remainder, "k-", label="Remainder (Error)", linewidth=1.5)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (b)")
    ax.set_title(f"{channel_name.capitalize()}: Remainder (Original - Reconstruction)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(y_min, y_max)

    plt.suptitle(
        f"{name} {channel_name.capitalize()} Cross Section Decomposition", fontsize=14
    )
    plt.tight_layout()

    if path_out:
        filename = f"{name}_{channel_name}_decomposition.png"
        plt.savefig(os.path.join(path_out, filename), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def fit_background_poly(E, xs_ref, xs_poles, degree=0, mask=None):
    """
    Fit a temperature-independent background polynomial to cross sections.

    Fits:
        sigma_bg(E) ≈ sigma_ref(E) - sigma_poles(E)

    using a polynomial in sqrt(E).

    Parameters
    ----------
    E : ndarray (n,)
        Energy grid (eV)
    xs_ref : ndarray (k, n)
        Reference cross sections at 0 K
    xs_poles : ndarray (k, n)
        Pole-only reconstructed cross sections (0 K)
    degree : int
        Polynomial degree in sqrt(E)
        degree=0 -> constant background
        degree=1 -> a + b*sqrt(E)
    mask : ndarray, optional
        Boolean mask of points to include in fit

    Returns
    -------
    coeffs : ndarray (k, degree+1)
        Polynomial coefficients per channel,
        ordered from lowest degree to highest:
            coeffs[:,0] = constant term
            coeffs[:,1] = linear sqrt(E) term
            etc.
    """
    E = np.asarray(E)
    u = np.sqrt(E)

    xs_ref = np.asarray(xs_ref)
    xs_poles = np.asarray(xs_poles)

    if mask is None:
        mask = np.ones_like(E, dtype=bool)

    k = xs_ref.shape[0]
    coeffs = np.zeros((k, degree + 1))

    # Vandermonde matrix in sqrt(E)
    V = np.vstack([u**d for d in range(degree + 1)]).T  # (n, degree+1)
    V = V[mask]

    for i in range(k):
        y = (xs_ref[i] - xs_poles[i])[mask]
        c, *_ = np.linalg.lstsq(V, y, rcond=None)
        coeffs[i] = c

    return coeffs


def eval_background_poly(E, coeffs):
    """
    Evaluate background polynomial in sqrt(E).

    Parameters
    ----------
    E : ndarray (n,)
    coeffs : ndarray (k, degree+1)

    Returns
    -------
    xs_bg : ndarray (k, n)
    """
    E = np.asarray(E)
    u = np.sqrt(E)

    k, degp1 = coeffs.shape
    xs_bg = np.zeros((k, len(E)))

    for d in range(degp1):
        xs_bg += coeffs[:, d, None] * u**d

    return xs_bg


def evaluate_multipole_xs(E, data_dict, poly_info=None, fit_space="sqrt_E"):
    """
    Evaluate cross sections using the pole/residue representation.

    This implements the basic multipole evaluation without windowing,
    useful for prototyping and validation.

    Parameters
    ----------
    E : float or array-like
        Energy in eV
    data_dict : dict
        Output from poles_residues_to_openmc_data

    Returns
    -------
    tuple
        (elastic_xs, absorption_xs, fission_xs) where fission_xs is None
        if not fissionable
    """

    E_array = np.atleast_1d(E)
    if fit_space == "sqrt_E":
        s = np.sqrt(E_array)  # Poles are in sqrt_E space
    else:
        s = E_array  # Poles are in E space

    data = data_dict["data"]
    fissionable = data_dict["fissionable"]

    # Initialize cross sections
    elastic_xs = np.zeros_like(E, dtype=float)
    absorption_xs = np.zeros_like(E, dtype=float)
    fission_xs = np.zeros_like(E, dtype=float) if fissionable else None

    # Add pole contributions
    for i, s_val in enumerate(s):
        # Using a vectorized operation is much faster than a second for-loop
        poles = data[:, 0]
        denominators = s_val - poles

        # The core formula for WMP format is Re(i * R / (E - p))
        contributions = 1 / denominators

        # Elastic (column 1)
        elastic_xs[i] = np.sum((data[:, 1] * contributions).real)

        # Absorption (column 2)
        absorption_xs[i] = np.sum((data[:, 2] * contributions).real)

        # Fission (column 3, if present)
        if fissionable:
            fission_xs[i] = np.sum((data[:, 3] * contributions).real)

    # Add polynomial contribution if provided
    if poly_info is not None:
        # Handle different input formats
        if isinstance(poly_info, dict):
            poly_coeffs = poly_info.get(
                "poly_coeffs", poly_info.get("polycoeffs", None)
            )
        else:
            poly_coeffs = poly_info

        if poly_coeffs is not None:
            # Ensure it's a list
            if not isinstance(poly_coeffs, list):
                poly_coeffs = [poly_coeffs]

            # Add polynomial contribution to each channel
            # Take real part since cross sections should be real
            if len(poly_coeffs) >= 1 and poly_coeffs[0] is not None:
                poly_val = np.polyval(poly_coeffs[0], s)
                elastic_xs = elastic_xs + np.real(poly_val)
            if len(poly_coeffs) >= 2 and poly_coeffs[1] is not None:
                poly_val = np.polyval(poly_coeffs[1], s)
                absorption_xs = absorption_xs + np.real(poly_val)
            if len(poly_coeffs) >= 3 and poly_coeffs[2] is not None:
                poly_val = np.polyval(poly_coeffs[2], s)
                fission_xs = fission_xs + np.real(poly_val)

        if np.any(elastic_xs < 0):
            neg_indices = np.where(elastic_xs < 0)[0]
            print(
                f"WARNING: Negative elastic XS detected at {len(neg_indices)} points!"
            )
            print(f"  Range: [{elastic_xs.min():.3e}, {elastic_xs.max():.3e}]")
            print(f"  At energies: {E[neg_indices[:5]]}")  # Show first 5
            # elastic_xs[elastic_xs < 0] = 1e-10

        if absorption_xs is not None and np.any(absorption_xs < 0):
            print("WARNING: Negative absorption XS detected!")
            # absorption_xs[absorption_xs < 0] = 1e-10

        if fission_xs is not None and np.any(fission_xs < 0):
            print("WARNING: Negative fission XS detected!")
            # fission_xs[fission_xs < 0] = 1e-10

    return [elastic_xs, absorption_xs, fission_xs]


def plot_aaa_results(
    E,
    sigma_s,
    sigma_a,
    R_s,
    R_a,
    sigma_f=None,
    R_f=None,
    path_out=None,
    title_prefix=None,
):
    """
    Plot ACE vs reconstructed cross sections and relative error
    in the same figure (trainer style), one figure per MT.

    Parameters
    ----------
    E : (M,) energy grid (eV)
    sigma_s, sigma_a, sigma_f : ACE/CE cross sections on E
    R_s, R_a, R_f : reconstructed cross sections on E
    path_out : directory to save figures (created if needed)
    title_prefix : optional extra text in plot title
    """
    if path_out:
        os.makedirs(path_out, exist_ok=True)

    E = np.asarray(E)
    Emin, Emax = float(E[0]), float(E[-1])

    # Set up channels: MT numbers + (ACE, RECON) pairs
    channels = [(2, sigma_s, R_s), (27, sigma_a, R_a)]  # elastic  # absorption
    if sigma_f is not None and R_f is not None:
        channels.append((18, sigma_f, R_f))  # fission

    for mt, xs_true, xs_fit in channels:
        if xs_true is None or xs_fit is None:
            continue
        xs_true = np.asarray(xs_true, dtype=float)
        # xs_fit = np.asarray(xs_fit, dtype=float)
        xs_fit = np.real(xs_fit) if np.iscomplexobj(xs_fit) else xs_fit

        # Relative error exactly like trainer (no floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(xs_fit - xs_true) / xs_true

        fig, ax1 = plt.subplots()
        lns1 = ax1.semilogy(E, xs_true, "g", label="ACE xs")
        lns2 = ax1.semilogy(E, xs_fit, "b", label="Reconstructed xs")
        ax2 = ax1.twinx()
        lns3 = ax2.semilogy(E, rel, "r", label="Relative error", alpha=0.5)

        lns = lns1 + lns2 + lns3
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc="best")

        ax1.set_xlabel("energy (eV)")
        ax1.set_ylabel("cross section (b)", color="b")
        ax1.tick_params(axis="y", colors="b")
        ax2.set_ylabel("relative error", color="r")
        ax2.tick_params(axis="y", colors="r")

        title = f"MT {mt} — {Emin:.0f}–{Emax:.0f} eV"
        if title_prefix:
            title = f"{title_prefix} | " + title
        plt.title(title)
        fig.tight_layout()

        out = f"{Emin:.0f}-{Emax:.0f}_MT{mt}.png"
        if path_out:
            out = os.path.join(path_out, out)
        plt.savefig(out, dpi=200)
        plt.close()


def plot_miaaa_convergence(err_hist, rtol=None, path_out=None):
    """
    Plot the convergence of errors from miaaa_xs function.

    Parameters
    ----------
    err_hist : list or array
        Error history returned by miaaa_xs (5th element of the return tuple)
    rtol : float, optional
        Relative tolerance line to show on plot
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = np.arange(len(err_hist))

    # Plot error history
    ax.semilogy(
        iterations, err_hist, "b.-", linewidth=2, markersize=6, label="Max Error"
    )

    # Add tolerance line if provided
    if rtol is not None:
        ax.axhline(
            y=rtol, color="r", linestyle="--", linewidth=2, label=f"rtol = {rtol:.1e}"
        )

    ax.set_xlabel("Iteration (m)")
    ax.set_ylabel("Error")
    ax.set_title("MIAAA Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Make it look nice
    plt.tight_layout()
    out = "miaaa_convergence.png"
    if path_out:
        out = os.path.join(path_out, out)
        plt.savefig(out, dpi=200)
