import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Choose plotting function
    if plot_type == "loglog":
        plot_func = plt.loglog
    elif plot_type == "semilogx":
        plot_func = plt.semilogx
    elif plot_type == "semilogy":
        plot_func = plt.semilogy
    else:  # linear
        plot_func = plt.plot

    # Plot data
    plot_func(E, original, f"b-", label=f"Original {symbol}", linewidth=2)
    plot_func(E, reconstructed, "r--", label="Reconstructed", linewidth=2)

   
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("Cross section (b)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

# Handle error plotting on secondary axis
    if show_error:
        ax2 = ax1.twinx()
        
        # Calculate error based on type
        original = np.array(original)
        reconstructed = np.array(reconstructed)
        
        if error_type == "relative":
            # Avoid division by zero
            mask = original != 0
            error = np.full_like(original, np.nan)
            error[mask] = np.abs((reconstructed[mask] - original[mask]) / original[mask]) * 100
            error_label = "Relative Error (%)"
            error_color = 'black'
        elif error_type == "absolute":
            error = np.abs(reconstructed - original)
            error_label = "Absolute Error (b)"
            error_color = 'black'
        elif error_type == "remainder":
            error = reconstructed - original
            error_label = "Remainder (b)"
            error_color = 'black'
        else:
            raise ValueError("error_type must be 'relative', 'absolute', or 'remainder'")
        
        # Plot error with appropriate scale
        if plot_type in ["loglog", "semilogy"]:
            # For log y-scale, we need positive values
            if error_type == "remainder":
                # For remainder, we might have negative values, so use regular plot
                ax2.plot(E, error, color=error_color, alpha=0.7, linewidth=1.5, 
                        label=error_label, linestyle=':')
            else:
                # For absolute and relative errors (always positive), we can use semilogy
                ax2.semilogy(E, error, color=error_color, alpha=0.7, linewidth=1.5, 
                           label=error_label, linestyle=':')
        else:
            ax2.plot(E, error, color=error_color, alpha=0.7, linewidth=1.5, 
                    label=error_label, linestyle=':')
        
        ax2.set_ylabel(error_label, color=error_color)
        ax2.tick_params(axis='y', labelcolor=error_color)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax1.legend()

    if path_out:
        filename = f"{name}_{channel_name}_reconstruction"
        if show_error:
            filename += f"_{error_type}_error"
        plt.savefig(os.path.join(path_out, f"{filename}.png"), dpi=200)
        plt.close()
    else:
        plt.show()


def plot_reconstruction(
    E, original_data, poles, residues, name="Nuclide", path_out=None, plot_type="loglog",
    show_error=False, error_type="relative"
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

    Returns
    -------
    dict
        Reconstructed cross sections
    """

    # Convert to OpenMC format and evaluate
    mc_data = poles_residues_to_openmc_data(poles, residues, name=name)
    elastic_recon, absorption_recon, fission_recon = evaluate_multipole_xs(E, mc_data)

    # Define channels to plot
    channels = [
        {
            "name": "elastic",
            "symbol": "σ_s",
            "original": original_data.get("sigma_s", original_data.get("elastic")),
            "reconstructed": elastic_recon,
        },
        {
            "name": "absorption",
            "symbol": "σ_a",
            "original": original_data.get("sigma_a", original_data.get("absorption")),
            "reconstructed": absorption_recon,
        },
        {
            "name": "fission",
            "symbol": "σ_f",
            "original": original_data.get("sigma_f", original_data.get("fission")),
            "reconstructed": fission_recon,
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
            E,
            channel["original"],
            channel["reconstructed"],
            channel["name"],
            channel["symbol"],
            name,
            path_out,
            plot_type,
            show_error=show_error,
            error_type=error_type,
        )

    if path_out:
        print(f"Saved reconstruction plots to {path_out}")

    return {
        "elastic": elastic_recon,
        "absorption": absorption_recon,
        "fission": fission_recon,
    }


def evaluate_multipole_xs(E, data_dict, background_constants=None):
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

    E = np.atleast_1d(E)
    data = data_dict["data"]
    fissionable = data_dict["fissionable"]

    # Initialize cross sections
    elastic_xs = np.zeros_like(E, dtype=float)
    absorption_xs = np.zeros_like(E, dtype=float)
    fission_xs = np.zeros_like(E, dtype=float) if fissionable else None

    # Add pole contributions
    for i, energy in enumerate(E):
        for pole_idx in range(data.shape[0]):
            pole = data[pole_idx, 0]

            # Simple pole evaluation: residue / (E - pole)
            denominator = energy - pole

            # if abs(denominator) > 1e-12:  # Avoid division by zero
            contribution = 1.0 / (denominator)  # Include 1/E factor

            # Elastic (column 1)
            elastic_xs[i] += (data[pole_idx, 1] * contribution).real

            # Absorption (column 2)
            absorption_xs[i] += (data[pole_idx, 2] * contribution).real

            # Fission (column 3, if present)
            if fissionable:
                fission_xs[i] += (data[pole_idx, 3] * contribution).real

    # # Add constant background as per Gavin's insight
    # if background_constants:
    #     # These are rough estimates - in practice you'd determine these
    #     # from the nearly constant remainder after subtracting pole contributions
    #     elastic_xs += background_constants["elastic"]
    #     absorption_xs += background_constants["absorption"]
    #     if fissionable:
    #         fission_xs += background_constants["fission"]

    # Ensure non-negative cross sections
    # elastic_xs = np.maximum(elastic_xs, 0.0)
    # absorption_xs = np.maximum(absorption_xs, 0.0)
    # if fissionable:
    #     fission_xs = np.maximum(fission_xs, 0.0)

    return elastic_xs, absorption_xs, fission_xs


def poles_residues_to_openmc_data(poles, residues, name="test_nuclide", AWR=235.0):
    """
    Simple conversion of poles and residues to OpenMC multipole data format.

    Takes poles and residues from AAA and creates the basic data structure
    that OpenMC expects

    Parameters
    ----------
    poles : array-like
        Complex poles in energy space (eV)
    residues : list or array
        Residues for each reaction channel. Should be:
        - [elastic_residues, absorption_residues] for non-fissionable
        - [elastic_residues, absorption_residues, fission_residues] for fissionable
        Each element should be an array of complex residues matching poles length
    name : str, optional
        Nuclide name (default "test_nuclide")
    AWR : float, optional
        Atomic weight ratio (default 235.0)

    Returns
    -------
    dict
        Dictionary with OpenMC-compatible data:
        - 'data': 2D array [pole_energy, elastic_residue, absorption_residue, (fission_residue)]
        - 'name': nuclide name
        - 'sqrtAWR': sqrt of atomic weight ratio
        - 'fissionable': boolean indicating if fission channel present
        - 'n_poles': number of poles
    """
    poles = np.array(poles, dtype=complex)
    n_poles = len(poles)

    # Determine if fissionable and get residue arrays
    if isinstance(residues, list):
        n_reactions = len(residues)
        residue_arrays = [np.array(r, dtype=complex) for r in residues]
    else:
        # Assume it's a 2D array with shape (n_reactions, n_poles)
        # residue_arrays = [residues[i] for i in range(residues.shape[0])]
        residue_arrays = residues.T
        n_reactions = len(residue_arrays)

    fissionable = n_reactions > 2

    # Validate dimensions
    for i, res_array in enumerate(residue_arrays):
        if len(res_array) != n_poles:
            raise ValueError(
                f"Residue array {i} length ({len(res_array)}) "
                f"doesn't match poles length ({n_poles})"
            )

    # Create the data array: [pole, elastic_residue, absorption_residue, (fission_residue)]
    data_cols = 1 + n_reactions
    data = np.zeros((n_poles, data_cols), dtype=complex)

    # Fill in poles (first column)
    data[:, 0] = poles

    # Fill in residues
    for i, res_array in enumerate(residue_arrays):
        data[:, i + 1] = res_array

    # Sort by pole energy (real part)
    sort_idx = np.argsort(data[:, 0].real)
    data = data[sort_idx]

    return {
        "data": data,
        "name": name,
        "sqrtAWR": np.sqrt(AWR),
        "fissionable": fissionable,
        "n_poles": n_poles,
        "n_reactions": n_reactions,
    }


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
