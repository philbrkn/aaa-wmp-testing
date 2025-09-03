from math import sqrt
from pathlib import Path

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import svd
import scipy.linalg as la

import openmc.checkvalue as cv

from .data import K_BOLTZMANN
from .neutron import IncidentNeutron
from .resonance import ResonanceRange
from .aaa import (
    evaluate_aaa,
    aaa_xs,
    extract_poles_and_residues,
    apply_cleanup2_to_aaa,
    plot_aaa_results,
)
from .miaaa import (
    miaaa_xs,
    evaluate_miaaa,
    proper_rational,
    extract_poles_residues
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


def vectfit_nuclide(
    endf_file,
    njoy_error=5e-4,
    vf_pieces=None,
    log=False,
    fitter="aaa",
    path_out=None,
    mp_filename=None,
    njoy_input=None,
    bounds=None,
    cleanup=False,
    plot_each_slice=True,
    fit_mask_guard=0,
    cleanup_tol=1e-6,
    analyze_constant=False,
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
        base_dir = Path(
            path_out
        ).parent  # TODO: this assumes path_out is a subdirectory
        njoy_path_out = base_dir / "NJOY_pickles"
        njoy_path_out.mkdir(parents=True, exist_ok=True)
        with open(njoy_path_out / "U238_NJOY.pickle", "wb") as f:
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

    if log:
        print(f"  MTs: {mts}")
        i0 = np.searchsorted(energy, E_min, side="left")
        i1 = np.searchsorted(energy, E_max, side="right")
        bound_pts = max(0, i1 - i0)  # number of points in [E_min, E_max]
        print(f"  Energy range: {E_min:.3e} to {E_max:.3e} eV ({bound_pts} points)")
        peaks, _ = find_peaks(total_xs)
        print(f"  There are {peaks.size} peaks in this range.")

    # ======================================================================
    # PERFORM VECTOR FITTING
    if vf_pieces is None:
        # divide into pieces for complex nuclides
        peaks, _ = find_peaks(total_xs)
        n_peaks = peaks.size
        if n_peaks > 200 or n_points > 30000 or n_peaks * n_points > 100 * 10000:
            vf_pieces = max(5, n_peaks // 10, n_points // 2000)
        else:
            vf_pieces = 1

    piece_width = (E_max - E_min) / vf_pieces
    # print(f"Piece width {piece_width}")
    alpha = nuc_ce.atomic_weight_ratio / (K_BOLTZMANN * TEMPERATURE_LIMIT)
    space = kwargs.get("space", "E")

    poles, residues = [], []
    # VF piece by piece
    for i_piece in range(vf_pieces):
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{vf_pieces}...")
        # start E of this piece
        E_left = E_min + i_piece * piece_width
        E_right = min(E_max, E_left + piece_width)

        s_left = np.sqrt(alpha * max(E_left, 0.0))
        s_right = np.sqrt(alpha * max(E_right, 0.0))
        if i_piece == 0 or s_left < 4:
            e_start = E_min
        else:
            e_start = max(E_min, (s_left - 4) ** 2 / alpha)

        e_end = min(E_max, (s_right + 4) ** 2 / alpha)

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

            if fitter == "aaa":
                w, z, fsz, faz, *rest = aaa_xs(
                    E_piece,
                    sig_s_piece,
                    sig_a_piece,
                    sigma_f=sig_f_piece,
                    method=kwargs.get("method", "full_svd"),
                    rtol=kwargs.get("rtol", 1e-13),
                    mmax=kwargs.get("mmax", 100),
                    log=log,
                    space=space,
                )
                # EXTRACT
                ffz = rest[0] if fissionable else None
                fvals_piece = [fsz, faz] + ([ffz] if fissionable else [])
                poles_s, residues_list = extract_poles_and_residues(
                    w.astype(complex),
                    z.astype(complex),
                    fvals_piece,
                    log=log,
                    space=space
                )
            elif fitter == "miaaa":
                w, z, fz, R, err_hist = miaaa_xs(
                    E_piece,
                    [sig_s_piece, sig_a_piece, sig_f_piece],
                    method=kwargs.get("method", "full_svd"),
                    rtol=kwargs.get("rtol", 1e-13),
                    mmax=kwargs.get("mmax", 100),
                    greedy_metric="relative",  # relative or absolute_sum
                    log=log,
                    space=space,
                    normalize=True,
                    lawson_iter=0
                )
                # w_num = w_den.copy()  # Same if no Lawson iteration
                # poles_s, residues_list = extract_poles_residues(w, z, fz)
                poles_s, residues_list, pra, pr_handles, polycoeffs = proper_rational(
                    z, w, w, fz, R, E_piece, maxpolydegree=1
                )
                print(polycoeffs)
            else:
                raise ValueError("Unknown fitter passed in.")

        if cleanup:
            # Define the actual fitting window (not the extended data range)
            z_clean, fs_clean, fa_clean, ff_clean, w_clean = apply_cleanup2_to_aaa(
                E_piece,
                sig_s_piece,
                sig_a_piece,
                z,
                fsz,
                faz,
                w,
                sigma_f=sig_f_piece,
                ff=(ffz if fissionable else None),
                cleanup_tol=cleanup_tol,  # Only remove if pole-zero distance < 1e-6
                space=space,
                log=log,
            )

            # Update the values
            z = z_clean
            fsz = fs_clean
            faz = fa_clean
            w = w_clean
            if fissionable:
                ffz = ff_clean

            # Then extract poles and residues with cleaned values
            fvals_piece = [fsz, faz] + ([ffz] if fissionable else [])
            poles_s, residues_list = extract_poles_and_residues(
                w.astype(complex),
                z.astype(complex),
                fvals_piece,
                log=log,
                space=space,
            )

        poles.append(poles_s)
        residues.append(residues_list)

        if plot_each_slice and fitter == "aaa":
            R_s_piece = evaluate_aaa(E_piece, w, z, fsz, space=space)
            R_a_piece = evaluate_aaa(E_piece, w, z, faz, space=space)
            R_f_piece = (
                evaluate_aaa(E_piece, w, z, ffz, space=space) if fissionable else None
            )
            plot_aaa_results(
                E_piece,
                sig_s_piece,
                sig_a_piece,
                R_s_piece,
                R_a_piece,
                sigma_f=sig_f_piece,
                R_f=R_f_piece,
                path_out=path_out,
            )
        elif plot_each_slice and fitter == "miaaa":
            R_pieces = evaluate_miaaa(E_piece, w, z, fz, space=space)
            plot_aaa_results(
                E_piece,
                sig_s_piece,
                sig_a_piece,
                R_pieces[0],
                R_pieces[1],
                sigma_f=sig_f_piece,
                R_f=R_pieces[2],
                path_out=path_out,
            )

    # print number of poles
    n_poles = sum([p.size for p in poles])
    if log:
        print(f"Total number of poles: {n_poles}")

    if vf_pieces == 1 and analyze_constant:
        background_analysis = analyze_constant_background(
            E_piece,
            sig_s_piece,
            sig_a_piece,
            poles,
            residues,
            sig_f_piece,
            path_out,
            name="U238",
            background_constants=kwargs.get("background_constants", None)
        )

        # verify_residues(
        #     poles,
        #     residues,
        #     energy,  # The full energy grid
        #     elastic_xs,
        #     absorption_xs,
        #     fission_xs if fissionable else None,
        # )

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
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if not mp_filename:
            mp_filename = f"{nuc_ce.name}_mp.pickle"
        mp_filename = os.path.join(path_out, mp_filename)
        with open(mp_filename, "wb") as f:
            pickle.dump(mp_data, f)
        if log:
            print(f"Dumped multipole data to file: {mp_filename}")

    return mp_data


def analyze_constant_background(
    E,
    sigma_s,
    sigma_a,
    poles,
    residues,
    sigma_f=None,
    path_out=None,
    background_constants=None,
    name="U238",
):
    """
    Analyze the constant background by evaluating pole contributions
    and examining the remainder

    Parameters
    ----------
    E : array-like
        Energy grid (eV)
    sigma_s, sigma_a : array-like
        Original elastic and absorption cross sections
    poles : array-like
        Complex poles in E space
    residues : list
        [elastic_residues, absorption_residues, (fission_residues)]
    sigma_f : array-like, optional
        Original fission cross sections
    fission_residues : array-like, optional
        Fission residues
    path_out : str, optional
        Directory to save plots
    name : str, optional
        Nuclide name for plots

    Returns
    -------
    dict
        Analysis results with background constants and pole contributions
    """
    poles = poles[0]
    # print(residues)
    # print(len(residues[0]))
    residues = residues[0]
    E = np.array(E)
    poles = np.array(poles, dtype=complex)

    mc_data = poles_residues_to_openmc_data(poles, residues, name=name)
    # Evaluate pole contributions only (no background)
    elastic_pole, absorption_pole, fission_pole = evaluate_multipole_xs(
        E,
        mc_data,
        background_constants=background_constants
    )

    # Calculate remainders (original - pole contributions)
    elastic_remainder = sigma_s - elastic_pole
    absorption_remainder = sigma_a - absorption_pole
    fission_remainder = None
    if sigma_f is not None and fission_pole is not None:
        fission_remainder = sigma_f - fission_pole

    # Analyze the remainder to find the "nearly constant" contribution
    elastic_mean = np.mean(elastic_remainder)
    elastic_std = np.std(elastic_remainder)
    elastic_median = np.median(elastic_remainder)
    from scipy import stats

    elastic_modes = stats.mode(elastic_remainder)
    absorption_mean = np.mean(absorption_remainder)
    absorption_std = np.std(absorption_remainder)
    absorption_median = np.median(absorption_remainder)

    results = {
        "elastic_background": elastic_mean,
        "elastic_background_std": elastic_std,
        "absorption_background": absorption_mean,
        "absorption_background_std": absorption_std,
        "elastic_remainder": elastic_remainder,
        "absorption_remainder": absorption_remainder,
        "elastic_pole_contribution": elastic_pole,
        "absorption_pole_contribution": absorption_pole,
    }

    if fission_remainder is not None:
        fission_median = np.median(fission_remainder)
        fission_mean = np.mean(fission_remainder)
        fission_std = np.std(fission_remainder)
        results.update(
            {
                "fission_background": fission_mean,
                "fission_background_std": fission_std,
                "fission_remainder": fission_remainder,
                "fission_pole_contribution": fission_pole,
            }
        )

    print(f"\nBackground Analysis for {name}:")
    # print(f"Elastic background: {elastic_mean:.4f} ± {elastic_std:.4f} b")
    # print(f"Absorption background: {absorption_mean:.4f} ± {absorption_std:.4f} b")
    print(f"Elastic background median: {elastic_median:.8f} b")
    print(f"Elastic mode {elastic_modes.mode} and count {elastic_modes.count}")
    print(f"Absorption background median: {absorption_median:.8f} b")
    if fission_remainder is not None:
        print(f"Fission background: {fission_mean:.8f} ± {fission_std:.4f} b")
        print(f"Fission background median: {fission_median} b")

    # Create plots showing the decomposition
    if path_out:
        os.makedirs(path_out, exist_ok=True)

        # Plot elastic decomposition
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig = plt.figure(figsize=(8, 6))

        plt.semilogy(E, sigma_s, "g-", label="Original σ_s", linewidth=2)
        plt.semilogy(E, elastic_pole, "b--", label="Pole contribution", linewidth=2)
        plt.semilogy(E, elastic_remainder, "r:", label="Remainder", linewidth=2)
        plt.axhline(
            elastic_median,
            color="r",
            linestyle="-",
            alpha=0.7,
            label=f"Median remainder = {elastic_median:.4f} b",
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Cross section (b)")
        plt.title(f"{name} Elastic Scattering Decomposition")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(path_out, f"{name}_elastic_decomposition.png"), dpi=200
        )
        plt.close()

        fig2 = plt.figure(figsize=(8, 6))
        # Plot remainder detail
        plt.semilogy(E, elastic_remainder, "r-", linewidth=2, label="Remainder")
        plt.axhline(
            elastic_median,
            color="k",
            linestyle="--",
            alpha=0.7,
            label=f"Median = {elastic_median:.4f} b",
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Remainder (b)")
        plt.title("Elastic Remainder Detail (Should be Nearly Constant)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(path_out, f"{name}_elastic_remainder.png"), dpi=200)
        plt.close()

        # Similar plot for absorption
        fig = plt.figure(figsize=(8, 6))

        plt.semilogy(E, sigma_a, "g-", label="Original σ_a", linewidth=2)
        plt.semilogy(E, absorption_pole, "b--", label="Pole contribution", linewidth=2)
        plt.semilogy(
            E, np.abs(absorption_remainder), "r:", label="|Remainder|", linewidth=2
        )
        plt.axhline(
            abs(absorption_median),
            color="r",
            linestyle="-",
            alpha=0.7,
            label=f"Median |remainder| = {abs(absorption_median):.4f} b",
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Cross section (b)")
        plt.title(f"{name} Absorption Decomposition")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_out, f"{name}_absorption_decomposition.png"), dpi=200
        )
        plt.close()

        fig = plt.figure(figsize=(8, 6))

        plt.semilogy(E, absorption_remainder, "r-", linewidth=2, label="Remainder")
        plt.axhline(
            absorption_median,
            color="k",
            linestyle="--",
            alpha=0.7,
            label=f"Median = {absorption_median:.4f} b",
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Remainder (b)")
        plt.title("Absorption Remainder Detail")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(path_out, f"{name}_absorption_remainder.png"), dpi=200)
        plt.close()

        # Similar plot for absorption
        fig = plt.figure(figsize=(8, 6))

        plt.semilogy(E, sigma_f, "g-", label="Original σ_f", linewidth=2)
        plt.semilogy(E, fission_pole, "b--", label="Pole contribution", linewidth=2)
        plt.semilogy(
            E, np.abs(fission_remainder), "r:", label="|Remainder|", linewidth=2
        )
        plt.axhline(
            abs(fission_median),
            color="r",
            linestyle="-",
            alpha=0.7,
            label=f"Median |remainder| = {abs(fission_median):.4f} b",
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Cross section (b)")
        plt.title(f"{name} Fission Decomposition")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_out, f"{name}_fission_decomposition.png"), dpi=200
        )
        plt.close()

        print(f"Saved decomposition plots to {path_out}")

    return results


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

    # Add constant background as per Gavin's insight
    if background_constants:
        # These are rough estimates - in practice you'd determine these
        # from the nearly constant remainder after subtracting pole contributions
        elastic_xs += background_constants["elastic"]
        absorption_xs +=  background_constants["absorption"]
        if fissionable:
            fission_xs +=  background_constants["fission"]

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


def verify_residues(poles, residues, E_test, sigma_s, sigma_a, sigma_f=None):
    """
    Verify that poles and residues reconstruct the original cross sections
    """
    poles = poles[0]  # First piece
    residues = residues[0]  # First piece [elastic_res, absorption_res, ...]

    print(f"\nResidue Analysis:")
    print(f"Number of poles: {len(poles)}")
    print(f"Number of residue channels: {len(residues)}")

    # Analyze elastic residues
    elastic_res = residues[0]
    absorption_res = residues[1]

    print(f"\nElastic residues:")
    print(f"  Shape: {elastic_res.shape}")
    print(f"  Max magnitude: {np.max(np.abs(elastic_res)):.3e}")
    print(f"  Min magnitude: {np.min(np.abs(elastic_res)):.3e}")
    print(f"  Mean magnitude: {np.mean(np.abs(elastic_res)):.3e}")
    print(f"  Positive real: {np.sum(elastic_res.real > 0)}")
    print(f"  Negative real: {np.sum(elastic_res.real < 0)}")
    print(f"  First 5 residues: {elastic_res[:5]}")

    print(f"\nAbsorption residues:")
    print(f"  Shape: {absorption_res.shape}")
    print(f"  Max magnitude: {np.max(np.abs(absorption_res)):.3e}")
    print(f"  Min magnitude: {np.min(np.abs(absorption_res)):.3e}")
    print(f"  Mean magnitude: {np.mean(np.abs(absorption_res)):.3e}")
    print(f"  Positive real: {np.sum(absorption_res.real > 0)}")
    print(f"  Negative real: {np.sum(absorption_res.real < 0)}")
    print(f"  First 5 residues: {absorption_res[:5]}")

    # Test reconstruction at a few energy points
    print(f"\nTest reconstruction at sample energies:")
    test_energies = E_test[:: len(E_test) // 10][:5]  # Sample 5 points

    for E in test_energies:
        elastic_recon = 0
        absorption_recon = 0

        for j, pole in enumerate(poles):
            if abs(E - pole) > 1e-12:
                elastic_recon += (elastic_res[j] / (E - pole)).real
                absorption_recon += (absorption_res[j] / (E - pole)).real

        # Find closest point in original data
        idx = np.argmin(np.abs(E_test - E))

        print(f"\n  E = {E:.3e} eV:")
        print(
            f"    Elastic:    orig = {sigma_s[idx]:.3e}, recon = {elastic_recon:.3e}, ratio = {elastic_recon/sigma_s[idx]:.3f}"
        )
        print(
            f"    Absorption: orig = {sigma_a[idx]:.3e}, recon = {absorption_recon:.3e}, ratio = {absorption_recon/sigma_a[idx]:.3f}"
        )

    # Check for any huge residues that might cause instability
    elastic_outliers = np.abs(elastic_res) > 10 * np.median(np.abs(elastic_res))
    absorption_outliers = np.abs(absorption_res) > 10 * np.median(
        np.abs(absorption_res)
    )

    print(f"\nOutlier analysis:")
    print(f"  Elastic outliers: {np.sum(elastic_outliers)} / {len(elastic_res)}")
    print(
        f"  Absorption outliers: {np.sum(absorption_outliers)} / {len(absorption_res)}"
    )

    if np.sum(elastic_outliers) > 0:
        print(
            f"  Elastic outlier magnitudes: {np.abs(elastic_res[elastic_outliers])[:5]}"
        )
    if np.sum(absorption_outliers) > 0:
        print(
            f"  Absorption outlier magnitudes: {np.abs(absorption_res[absorption_outliers])[:5]}"
        )
