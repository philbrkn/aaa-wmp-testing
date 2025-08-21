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


def aaa_xs(
    E,
    sigma_s,
    sigma_a,
    sigma_f=None,
    rtol=1e-13,
    mmax=100,
    log=False,
    fit_mask=None,
    core_mask=None,
):
    """
    Cross-section AAA algorithm (Algorithm 1 from Ridley and Forget, 2025)

    Parameters
    ----------
    s : ndarray
        Scattering matrix.
    sigma_s : ndarray
        Scattering cross section.
    sigma_a : ndarray
        Absorption cross section.
    sigma_f : ndarray, optional
        Fission cross section, if applicable.
    tol : float, optional
        Tolerance for singular values, default is 1e-13.
    mmax : int, optional
        Maximum number of poles, default is 100.
    fit_mask : ndarray, optional
        Boolean mask indicating which energy points to include in the fit.
    core_mask : ndarray, optional
        Boolean mask indicating which energy points are in the core region.

    Returns
    -------
    tuple
        A tuple containing:
        - w : ndarray
            Final right singular vector.
        - z : ndarray
            Interpolation points.
        - fs : ndarray
            Scattering cross section values at interpolation points.
        - fa : ndarray
            Absorption cross section values at interpolation points.
        - ff : ndarray, optional
            Fission cross section values at interpolation points, if applicable.
        - R_s : ndarray
            Reconstructed scattering cross section.
        - R_a : ndarray
            Reconstructed absorption cross section.
        - R_f : ndarray, optional
            Reconstructed fission cross section, if applicable.
    """
    # Initialize the interpolation point indices
    n = E.shape[0]
    s = np.sqrt(E)

    # construct fit mask and core mask
    if fit_mask is None:
        fit_mask = np.ones(n, dtype=bool)
    if core_mask is None:
        core_mask = fit_mask.copy()
    # index sets
    J_fit = np.flatnonzero(fit_mask).astype(int)
    J_core = np.flatnonzero(core_mask).astype(int)

    # Initialize non support index set
    J = np.arange(n)  # J <- {0, 1, ..., |E|-1},
    z_list, fs_list, fa_list = [], [], []
    ff_list = [] if sigma_f is not None else None

    # Precompute once above the loop
    dE = np.empty_like(E)
    dE[1:-1] = 0.5 * (E[2:] - E[:-2])
    dE[0] = E[1] - E[0]
    dE[-1] = E[-1] - E[-2]

    # Initial constant guesses
    R_s = np.full_like(sigma_s, np.mean(sigma_s), dtype=float)
    R_a = np.full_like(sigma_a, np.mean(sigma_a), dtype=float)
    R_f = (
        np.full_like(sigma_f, np.mean(sigma_f), dtype=float)
        if sigma_f is not None
        else None
    )

    # eps = np.finfo(sigma_s.dtype).tiny
    eps = 1e-13
    tiny = np.finfo(float).tiny

    # Estimate local spacing around z_j from the s grid
    def local_E_spacing(zj, Egrid):
        Ej = zj**2
        k = np.searchsorted(Egrid, Ej)
        if k <= 0:
            left = Egrid[1] - Egrid[0]
        else:
            left = Ej - Egrid[k - 1]
        if k >= len(Egrid):
            right = Egrid[-1] - Egrid[-2]
        else:
            right = Egrid[k] - Ej
        return 0.5 * (left + right)

    Fmax = max(
        np.max(np.abs(sigma_s)),
        np.max(np.abs(sigma_a)),
        np.max(np.abs(sigma_f)) if sigma_f is not None else 0.0,
    )
    Fmax = max(Fmax, tiny)  # avoid zero

    # For loop m=0,1,...,mmax-1
    for m in range(mmax):
        # Compute residuals serr<-|sigmas-Rs|/sigmas, aerr
        s_err = np.abs(sigma_s - R_s) / np.maximum(sigma_s, eps)
        a_err = np.abs(sigma_a - R_a) / np.maximum(sigma_a, eps)
        if sigma_f is not None:
            f_err = np.abs(sigma_f - R_f) / np.maximum(sigma_f, eps)
            err = np.maximum(s_err, np.maximum(a_err, f_err))
        else:
            err = np.maximum(s_err, a_err)

        # Determine j* as the index with the maximum error among components
        candidate_errs = err[J_fit]  # pick out only the J‑entries
        jpos = np.argmax(candidate_errs)  # pos of worst error pt w/in candidate list J
        j_star = J_fit[
            jpos
        ]  # actual index into full arrays E, sigs of that worst error pt

        # Update zj, fs, fa with values at s[j*], sigma_s[j*], sigma_a[j*]
        z_list.append(s[j_star])
        fs_list.append(sigma_s[j_star])
        fa_list.append(sigma_a[j_star])
        if sigma_f is not None:
            ff_list.append(sigma_f[j_star])

        # remove j* from J
        # J_core = np.delete(J_core, jpos)
        # J_fit = np.delete(J_fit, jpos)
        J_core = J_core[J_core != j_star]  # remove by value, not by position
        J_fit = J_fit[J_fit != j_star]  # same here

        # Convert lists to arrays for matrix operations
        z = np.array(z_list)
        fs = np.array(fs_list)
        fa = np.array(fa_list)
        ff = np.array(ff_list) if sigma_f is not None else None

        # Compute Loewner matrices A_s, A_a, A_f
        delta = s[J_fit][:, None] - z[None, :]
        A_s = (sigma_s[J_fit][:, None] - fs[None, :]) / delta
        A_a = (sigma_a[J_fit][:, None] - fa[None, :]) / delta
        if sigma_f is not None:
            A_f = (sigma_f[J_fit][:, None] - ff[None, :]) / delta

        # approximate approximate continuous LS
        row_w = np.sqrt(np.maximum(dE[J_fit], np.finfo(float).tiny))[:, None]
        A_s *= row_w
        A_a *= row_w
        if sigma_f is not None:
            A_f *= row_w

        col_w = np.sqrt(np.array([local_E_spacing(zj, E) for zj in z]))
        A_s *= col_w[None, :]
        A_a *= col_w[None, :]
        if sigma_f is not None:
            A_f *= col_w[None, :]

        # Stack vertically to get a single shared w (length m)
        L = np.vstack((A_s, A_a)) if sigma_f is None else np.vstack((A_s, A_a, A_f))

        # Compute SVD([A_s, A_a, A_f])
        _, _, Vh = svd(L, full_matrices=False)

        # wj <- final right singular vector
        w = Vh[-1, :]
        w = w * col_w  # undo column weighting

        # Update R on full grid via barycentric evaluation (interp at supports)
        R_s = evaluate_aaa(E, w, z, fs)
        R_a = evaluate_aaa(E, w, z, fa)
        if sigma_f is not None:
            R_f = evaluate_aaa(E, w, z, ff)

        rel = np.maximum.reduce(
            [
                np.abs((sigma_s - R_s) / np.maximum(sigma_s, eps)),
                np.abs((sigma_a - R_a) / np.maximum(sigma_a, eps)),
                (
                    np.abs((sigma_f - R_f) / np.maximum(sigma_f, eps))
                    if sigma_f is not None
                    else 0.0
                ),
            ]
        )
        err_inf = np.max(rel[J_core])
        if log >= DETAILED_LOGGING:
            print(
                f"    m={m}, pick E={E[j_star]:.3e} eV, "
                f"AAA_err_inf={err_inf:.3e}, target={rtol:.3e}"
            )
        if err_inf <= rtol:
            if log >= DETAILED_LOGGING:
                print(f"Converged at m={m} (err_inf={err_inf:.3e} ≤ rtol={rtol:.1e})")
            break

    # Prepare outputs
    outputs = (w, z, fs, fa)
    if sigma_f is not None:
        outputs += (ff,)
    outputs += (R_s, R_a)
    if sigma_f is not None:
        outputs += (R_f,)
    return outputs


def evaluate_aaa(E, w, z, fz):
    """Barycentric evaluation with exact support-point handling."""
    s = np.sqrt(E)
    # Cauchy matrix only once
    # Avoid division by zero at support points
    with np.errstate(divide="ignore", invalid="ignore"):
        CC = 1.0 / (s[:, None] - z[None, :])
    num = CC @ (w * fz)
    den = CC @ w

    # Handle support points exactly
    tol = 1e-12
    for j, zj in enumerate(z):
        idx = np.argmin(np.abs(s - zj))
        if np.abs(s[idx] - zj) < tol:
            # Use exact value at support point
            num[idx] = w[j] * fz[j]
            den[idx] = w[j]

    return num / den


def extract_poles_and_residues(w, z, fvals, plane="s", log=False):
    """
    plane: "s" to return s-plane poles; "E" to return E-plane (E = s^2).
    """
    m = len(z)
    C = np.zeros((m + 1, m + 1), dtype=complex)
    C[0, 1:] = w
    C[1:, 0] = 1.0
    C[1:, 1:] = np.diag(z)
    C[0, 0] = 0.0

    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0

    lam, _ = la.eig(C, B)
    poles = lam[np.isfinite(lam)]  # finite eigenvalues

    if plane.lower() == "e":
        # a, b = poles.real, poles.imag
        # poles = (a*a - b*b) + 1j*(2*a*b)
        poles = poles**2

    # residues for each component fk at these poles (in current plane's variable)
    def residues_for(fk):
        fk = np.asarray(fk, dtype=complex)
        r = np.empty_like(poles, dtype=complex)
        for i, p in enumerate(poles):
            num = np.sum(w * fk / (p - z))
            dprime = -np.sum(w / (p - z) ** 2)
            r[i] = num / dprime
        return r

    residues = [residues_for(fk) for fk in fvals]

    # cosmetic: snap tiny Im parts to 0 for printing
    imag_thr = 100 * np.finfo(float).eps * np.maximum(1.0, np.abs(poles.real))
    poles = poles.real + 1j * np.where(np.abs(poles.imag) < imag_thr, 0.0, poles.imag)

    # sorted by real part for stable output
    idx = np.argsort(poles.real)
    poles = poles[idx]
    residues = [r[idx] for r in residues]

    if log:
        for p in poles:
            print(f"poles real {p.real:6.2f}   imag {p.imag:8.2e}")
    return poles, residues


def vectfit_nuclide(
    endf_file,
    njoy_error=5e-4,
    vf_pieces=None,
    log=False,
    path_out=None,
    mp_filename=None,
    njoy_input=None,
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
        base_dir = Path(path_out).parent  #TODO: this assumes path_out is a subdirectory
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
    E_min, E_max = energy[0], energy[-1]
    E_min = 0
    E_max = 200
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
        print(f"  Energy range: {E_min:.3e} to {E_max:.3e} eV ({n_points} points)")

    # ======================================================================
    # PERFORM VECTOR FITTING

    if vf_pieces is None:
        # divide into pieces for complex nuclides
        peaks, _ = find_peaks(total_xs)
        n_peaks = peaks.size
        if n_peaks > 200 or n_points > 30000 or n_peaks * n_points > 100 * 10000:
            vf_pieces = max(5, n_peaks // 50, n_points // 2000)
        else:
            vf_pieces = 1
    piece_width = (sqrt(E_max) - sqrt(E_min)) / vf_pieces

    alpha = nuc_ce.atomic_weight_ratio / (K_BOLTZMANN * TEMPERATURE_LIMIT)

    poles, residues = [], []
    # VF piece by piece
    for i_piece in range(vf_pieces):
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{vf_pieces}...")
        # start E of this piece
        e_bound = (sqrt(E_min) + piece_width * (i_piece - 0.5)) ** 2
        if i_piece == 0 or sqrt(alpha * e_bound) < 4.0:
        # if E_min == 0:
            e_start = E_min
            e_start_idx = 0
        else:
            e_start = max(E_min, (sqrt(alpha * e_bound) - 4.0) ** 2 / alpha)
            e_start_idx = np.searchsorted(energy, e_start, side="right") - 1
        # end E of this piece
        e_bound = (sqrt(E_min) + piece_width * (i_piece + 1)) ** 2
        e_end = min(E_max, (sqrt(alpha * e_bound) + 4.0) ** 2 / alpha)
        e_end_idx = np.searchsorted(energy, e_end, side="left") + 1
        e_idx = range(e_start_idx, min(e_end_idx + 1, n_points))
        print(
            f"  Piece {i_piece + 1}: E={energy[e_start_idx]:.3e} to {energy[e_end_idx - 1]:.3e} eV"
        )

        # no boundary mask
        E_piece = energy[e_idx]
        sig_s_piece = ce_xs[0, e_idx]
        sig_a_piece = ce_xs[1, e_idx]
        sig_f_piece = ce_xs[2, e_idx] if fissionable else None

        w, z, f_s_z, f_a_z, *rest = aaa_xs(
            E_piece,
            sig_s_piece,
            sig_a_piece,
            sigma_f=sig_f_piece,
            rtol=kwargs.get("rtol", 1e-13),
            mmax=kwargs.get("mmax", 100),
            log=log,
        )
        # s_piece = np.sqrt(E_piece)
        # F_s = sig_s_piece * E_piece
        # F_a = sig_a_piece * E_piece
        # F_f = sig_f_piece * E_piece if fissionable else None

        # w, z, F_s_z, F_a_z, *rest = aaa_xs(
        #     s_piece,
        #     F_s,
        #     F_a,
        #     sigma_f=F_f,
        #     rtol=kwargs.get("rtol", 1e-13),
        #     mmax=kwargs.get("mmax", 100),
        #     log=log,
        # )

        # F_f_z = rest[0] if fissionable else None

        # # Evaluate in s, then back to sigma by dividing by E
        # R_s_piece = evaluate_aaa(s_piece, w, z, F_s_z) / E_piece
        # R_a_piece = evaluate_aaa(s_piece, w, z, F_a_z) / E_piece
        # R_f_piece = (evaluate_aaa(s_piece, w, z, F_f_z) / E_piece) if fissionable else None
        # fvals_piece = [F_s_z, F_a_z] + ([F_f_z] if fissionable else [])
        # poles_s, residues_list = extract_poles_and_residues(
        #     w.astype(complex), z.astype(complex), fvals_piece, log=log
        # )

        # # boundary mask
        # g = 0.1 * (e_end - e_start)
        # mask_fit = (energy >= e_start-g) & (energy <= e_end + g)
        # mask_core = (energy >= e_start) & (energy <= e_end)
        # E_piece      = energy[mask_fit]
        # sig_s_piece  = elastic_xs[mask_fit]
        # sig_a_piece  = absorption_xs[mask_fit]
        # sig_f_piece  = fission_xs[mask_fit] if fissionable else None

        # w, z, f_s_z, f_a_z, *rest = aaa_xs(
        #     E_piece,
        #     sig_s_piece,
        #     sig_a_piece,
        #     sigma_f=sig_f_piece,
        #     rtol=kwargs.get("rtol", 1e-13),
        #     mmax=kwargs.get("mmax", 100),
        #     log=log,
        #     fit_mask=np.ones_like(E_piece, dtype=bool),
        #     core_mask=mask_core[mask_fit],
        # )

        # EXTRACT AND FROISSART
        f_f_z = rest[0] if fissionable else None
        fvals_piece = [f_s_z, f_a_z] + ([f_f_z] if fissionable else [])
        poles_s, residues_list = extract_poles_and_residues(
            w.astype(complex), z.astype(complex), fvals_piece, log=log
        )

        print("Cleaning up doublets...")
        w, z, f_s_z, f_a_z, f_f_z = cleanup_doublets(
            E_piece,
            sig_s_piece,
            sig_a_piece,
            z,
            f_s_z,
            f_a_z,
            w,
            sigma_f=sig_f_piece,
            ff=(f_f_z if fissionable else None),
            tol=1e-4,
            max_passes=3,
            log=True,
        )

        fvals_piece = [f_s_z, f_a_z] + ([f_f_z] if fissionable else [])
        poles_s, residues_list = extract_poles_and_residues(
            w.astype(complex), z.astype(complex), fvals_piece, log=log
        )

        poles.append(poles_s)
        residues.append(residues_list)
        R_s_piece = evaluate_aaa(E_piece, w, z, f_s_z)
        R_a_piece = evaluate_aaa(E_piece, w, z, f_a_z)
        R_f_piece = evaluate_aaa(E_piece, w, z, f_f_z) if fissionable else None
        plot_aaa_results(
            E_piece,
            sig_s_piece,
            sig_a_piece,
            R_s_piece,
            R_a_piece,
            sigma_f=sig_f_piece,
            R_f=R_f_piece,
            path_out="aaa_window",
        )

    # print number of poles
    n_poles = sum([p.size for p in poles])
    if log:
        print(f"Total number of poles: {n_poles}")

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

        R_s_piece = evaluate_aaa(E_piece, w, z, f_s_z)
        R_a_piece = evaluate_aaa(E_piece, w, z, f_a_z)
        R_f_piece = evaluate_aaa(E_piece, w, z, f_f_z) if fissionable else None
        # R_s_piece = evaluate_aaa(E_piece, w, z, F_s_z) / E_piece
        # R_a_piece = evaluate_aaa(E_piece, w, z, F_a_z) / E_piece
        # R_f_piece = (evaluate_aaa(E_piece, w, z, F_f_z) / E_piece if fissionable else None)
        plot_aaa_results(
            E_piece,
            sig_s_piece,
            sig_a_piece,
            R_s_piece,
            R_a_piece,
            sigma_f=sig_f_piece,
            R_f=R_f_piece,
        )

    return mp_data


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
        xs_fit = np.asarray(xs_fit, dtype=float)

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


def cleanup_doublets(
    E,
    sigma_s,
    sigma_a,
    z,
    fs,
    fa,
    w,
    sigma_f=None,
    ff=None,
    tol=1e-13,
    max_passes=3,
    log=True,
):
    sgrid = np.sqrt(E)

    def extract_primitives(curr_w, curr_z, curr_fs, curr_fa, curr_ff):
        # poles in s-plane and residues for each component
        fvals = [curr_fs, curr_fa] + ([curr_ff] if curr_ff is not None else [])
        poles_s, residues_list = extract_poles_and_residues(
            curr_w.astype(complex), curr_z.astype(complex), fvals, plane="s"
        )
        # define a scale similar to Chebfun’s geometric-mean |F|
        # use medians to avoid huge spikes near resonances
        Fscale = max(
            np.median(np.abs(sigma_s)),
            np.median(np.abs(sigma_a)),
            np.median(np.abs(sigma_f)) if sigma_f is not None else 0.0,
            np.finfo(float).tiny,
        )
        return poles_s, residues_list, Fscale

    def rebuild_w(curr_z, curr_fs, curr_fa, curr_ff):
        # re-solve the unweighted LS for w from stacked Loewner on non-support set
        s = sgrid
        J = np.arange(len(E))
        for zj in curr_z:
            jj = np.argmin(np.abs(s - zj))
            take = np.where(J == jj)[0]
            if take.size:
                J = np.delete(J, take[0])
        delta = s[J][:, None] - curr_z[None, :]
        A_s = (sigma_s[J][:, None] - curr_fs[None, :]) / delta
        A_a = (sigma_a[J][:, None] - curr_fa[None, :]) / delta
        blocks = [A_s, A_a]
        if sigma_f is not None:
            A_f = (sigma_f[J][:, None] - curr_ff[None, :]) / delta
            blocks.append(A_f)
        L = np.vstack(blocks)
        _, _, Vh = la.svd(L, full_matrices=False)
        return Vh[-1, :]

    zc, fsc, fac = z.copy(), fs.copy(), fa.copy()
    ffc = ff.copy() if (ff is not None) else None
    wc = w.copy()

    for it in range(max_passes):
        poles_s, residues_list, Fscale = extract_primitives(wc, zc, fsc, fac, ffc)
        # Define a spuriousness test like Chebfun: |res| / dist_to_grid < tol * Fscale
        sdist = np.array([np.min(np.abs(p - sgrid)) for p in poles_s])
        # magnitude across components (max)
        res_mag = np.max(np.vstack([np.abs(r) for r in residues_list]), axis=0)
        crit = res_mag / np.maximum(sdist, 1e-14)
        bad = np.where(crit < tol * Fscale)[0]
        if bad.size == 0:
            if log:
                print(f"  cleanup: no doublets at pass {it+1}")
            break

        if log:
            print(f"  cleanup: removing {bad.size} suspected doublets at pass {it+1}")

        # For each bad pole, remove the closest support point
        to_prune = set()
        for k in bad:
            pole = poles_s[k]
            jstar = int(np.argmin(np.abs(zc - pole)))
            to_prune.add(jstar)
        if not to_prune:
            break

        keep = np.array([j for j in range(zc.size) if j not in to_prune], dtype=int)
        zc, fsc, fac = zc[keep], fsc[keep], fac[keep]
        if ffc is not None:
            ffc = ffc[keep]
        if zc.size < 2:
            break
        wc = rebuild_w(zc, fsc, fac, ffc)

    return wc, zc, fsc, fac, ffc
