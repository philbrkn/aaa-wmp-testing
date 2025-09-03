from math import sqrt
from pathlib import Path

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import svd
import scipy.linalg as la
import warnings
from scipy.sparse import spdiags

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
    space="E",
    method="full_svd",
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
    if space == "sqrt_E":
        grid = np.sqrt(E)
    elif space == "E":
        grid = E
    else:
        raise ValueError(f"Unknown space: {space}")

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
    ds = np.zeros_like(grid)
    ds[:-1] += 0.5 * np.diff(grid)
    ds[1:] += 0.5 * np.diff(grid)

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
        z_list.append(grid[j_star])
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
        delta = grid[J_fit][:, None] - z[None, :]
        A_s = (sigma_s[J_fit][:, None] - fs[None, :]) / delta
        A_a = (sigma_a[J_fit][:, None] - fa[None, :]) / delta
        if sigma_f is not None:
            A_f = (sigma_f[J_fit][:, None] - ff[None, :]) / delta

        # approximate approximate continuous LS
        A_s = ds[J_fit, None] * A_s
        A_a = ds[J_fit, None] * A_a
        if sigma_f is not None:
            A_f = ds[J_fit, None] * A_f

        # relative error weighting for svd
        rel_w_s = 1.0 / np.maximum(np.abs(sigma_s[J_fit]), eps)
        rel_w_a = 1.0 / np.maximum(np.abs(sigma_a[J_fit]), eps)
        A_s *= rel_w_s[:, None]
        A_a *= rel_w_a[:, None]
        if sigma_f is not None:
            rel_w_f = 1.0 / np.maximum(np.abs(sigma_f[J_fit]), eps)
            A_f *= rel_w_f[:, None]

        # TODO:
        # divide every col by val of function
        # normalize error across channels

        # Stack vertically to get a single shared w (length m)
        L = np.vstack((A_s, A_a)) if sigma_f is None else np.vstack((A_s, A_a, A_f))

        # Compute SVD([A_s, A_a, A_f])
        # Q, R = np.linalg.qr(L, mode='reduced')
        if method == "full_svd":
            _, _, Vh = svd(L, full_matrices=False)
        elif method == "qr+svd":
            Q, R = np.linalg.qr(L, mode="reduced")
            _, _, Vh = svd(R, full_matrices=False)
        elif method == "randomized_svd":
            from sklearn.utils.extmath import randomized_svd

            U, s, Vh = randomized_svd(
                L, n_components=min(20, L.shape[1]), random_state=42, n_iter=7
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # wj <- final right singular vector
        w = Vh[-1, :]

        # Update R on full grid via barycentric evaluation (interp at supports)
        R_s = evaluate_aaa(E, w, z, fs, space)
        R_a = evaluate_aaa(E, w, z, fa, space)
        if sigma_f is not None:
            R_f = evaluate_aaa(E, w, z, ff, space)

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


def evaluate_aaa(E, w, z, fz, space="E"):
    """Barycentric evaluation with exact support-point handling."""
    if space == "sqrt_E":
        grid = np.sqrt(E)
    elif space == "E":
        grid = E
    else:
        raise ValueError(f"Unknown space: {space}")

    # Cauchy matrix only once
    # Avoid division by zero at support points
    with np.errstate(divide="ignore", invalid="ignore"):
        CC = 1.0 / (grid[:, None] - z[None, :])
    num = CC @ (w * fz)
    den = CC @ w

    # Handle support points exactly
    tol = 1e-12
    for j, zj in enumerate(z):
        idx = np.argmin(np.abs(grid - zj))
        if np.abs(grid[idx] - zj) < tol:
            # Use exact value at support point
            num[idx] = w[j] * fz[j]
            den[idx] = w[j]

    return num / den


def extract_poles_and_residues(w, z, fvals, space="E", log=False):
    """
    plane: "s" to return s-plane poles; "E" to return E-plane (E = s^2).
    """

    m = len(z)
    tol = 1e-2
    
    # Check if deflation is needed
    w_sum = np.sum(w)
    deflation_count = 0
    w_working = w.copy()
    deflated_indices = []
    print(f"w_working sum {abs(np.sum(w_working))}")
    
    while abs(np.sum(w_working)) < tol and deflation_count < m-1:
        # Choose a support point for deflation
        available = [i for i in range(m) if i not in deflated_indices]
        j = available[np.argmax([np.abs(z[i]) for i in available])]
        
        # Deflate: multiply by (z - z_j)
        w_working = np.array([w_working[i] * (z[i] - z[j]) if i != j else 0 
                              for i in range(m)])
        deflated_indices.append(j)
        deflation_count += 1
        
        if log:
            print(f"Deflation {deflation_count}: removed z[{j}] = {z[j]:.3e}")
    
    C = np.zeros((m + 1, m + 1), dtype=complex)
    C[0, 1:] = w
    C[1:, 0] = 1.0
    C[1:, 1:] = np.diag(z)
    C[0, 0] = 0.0

    B = np.eye(m + 1, dtype=complex)
    B[1:, 0] = 1.0
    B[0, 0] = 0.0
    lam, _ = la.eig(C, B)
    poles = lam[np.isfinite(lam)]  # finite eigenvalues

    if space == "sqrt_E":
        # poles = poles
        poles = poles**2
    elif space == "E":
        # poles = np.sqrt(poles)
        poles = poles
    else:
        raise ValueError(f"Unknown space: {space}")

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

    # sorted by real part for stable output
    idx = np.argsort(poles.real)
    poles = poles[idx]
    residues = [r[idx] for r in residues]

    if log:
        # cosmetic: snap tiny Im parts to 0 for printing
        imag_thr = 100 * np.finfo(float).eps * np.maximum(1.0, np.abs(poles.real))
        poles_print = poles.real + 1j * np.where(np.abs(poles.imag) < imag_thr, 0.0, poles.imag)

        for p in poles_print:
            print(f"poles real {p.real:6.2f}   imag {p.imag:8.2e}")

    # if log:
    #     print(f"Found {len(poles)} poles")
    #     # Test reconstruction away from support points
    #     test_E = (z[10] + z[11]) / 2  # Midpoint between first two support points

    #     for i, fk in enumerate(fvals):
    #         # Pole-residue reconstruction
    #         recon = np.sum(residues[i] / (test_E - poles))

    #         # Barycentric evaluation for comparison
    #         bary_num = np.sum(w * fk / (test_E - z))
    #         bary_den = np.sum(w / (test_E - z))
    #         bary_val = bary_num / bary_den

    #         print(
    #             f"  Channel {i} at E={test_E:.3e}: "
    #             f"bary={bary_val.real:.3e}, pole-res={recon.real:.3e}, "
    #             f"ratio={recon.real/bary_val.real:.3f}"
    #         )

    return poles, residues


# def extract_poles_and_residues(w, z, fvals, space="sqrt_E", log=False):
#     # Find poles as zeros of the common denominator
#     # D(x) = Σ w_j / (x - z_j)

#     # Build the companion matrix for the denominator
#     m = len(z)
#     C = np.zeros((m + 1, m + 1), dtype=complex)
#     C[0, 1:] = w
#     C[1:, 0] = 1.0
#     C[1:, 1:] = np.diag(z)
#     C[0, 0] = 0.0

#     B = np.eye(m + 1, dtype=complex)
#     B[0, 0] = 0.0
#     # B = np.zeros((m + 1, m + 1), dtype=complex)
#     B[1:, 0] = 1.0
#     # B[1:, 1:]

#     lam, _ = la.eig(C, B)
#     poles = lam[np.isfinite(lam)]  # finite eigenvalues

#     # Convert poles to E space if needed
#     if space == "sqrt_E":
#         poles = poles**2
#     poles = np.array(poles, dtype=complex)

#     # Compute residues for each channel
#     residues_list = []

#     for i, fk in enumerate(fvals):
#         residues = np.zeros(len(poles), dtype=complex)

#         # For each pole, compute residue using l'Hôpital's rule
#         for j, pole in enumerate(poles):
#             # Numerator at pole: N(pole) = Σ w_k * f_k / (pole - z_k)
#             # Denominator derivative at pole: D'(pole) = Σ -w_k / (pole - z_k)^2

#             # We need to be careful about numerical precision near poles
#             # Use a small offset to evaluate near the pole
#             eps = 1e-15
#             pole_offset = pole + eps

#             num = np.sum(w * fk / (pole_offset - z))
#             denom_deriv = np.sum(-w / (pole_offset - z) ** 2)

#             residues[j] = num / denom_deriv

#         residues_list.append(residues)

#     # Sort by real part of poles
#     idx = np.argsort(poles.real)
#     poles = poles[idx]
#     residues_list = [r[idx] for r in residues_list]

#     if log:
#         print(f"Found {len(poles)} poles")
#         # Test reconstruction away from support points
#         test_E = (z[10] + z[11]) / 2  # Midpoint between first two support points

#         for i, fk in enumerate(fvals):
#             # Pole-residue reconstruction
#             recon = np.sum(residues_list[i] / (test_E - poles))

#             # Barycentric evaluation for comparison
#             bary_num = np.sum(w * fk / (test_E - z))
#             bary_den = np.sum(w / (test_E - z))
#             bary_val = bary_num / bary_den

#             print(
#                 f"  Channel {i} at E={test_E:.3e}: "
#                 f"bary={bary_val.real:.3e}, pole-res={recon.real:.3e}, "
#                 f"ratio={recon.real/bary_val.real:.3f}"
#             )

#     return poles, residues_list


def cleanup(
    z,
    fs,
    fa,
    ff,
    w,
    Z,
    sigma_s,
    sigma_a,
    sigma_f=None,
    cleanup_tol=1e-13,
    space="E",
    log=False,
):
    """
    Simple Froissart doublet cleanup based on MATLAB's cleanup function.
    Removes spurious pole-zero pairs by identifying negligible residues.

    Parameters
    ----------
    z : array_like
        Support points (in transformed space)
    fs : array_like
        Scattering cross-section values at support points
    fa : array_like
        Absorption cross-section values at support points
    ff : array_like or None
        Fission cross-section values at support points
    w : array_like
        Weights
    Z : array_like
        Sample points (energy grid in original space)
    sigma_s : array_like
        Scattering cross-section values at sample points
    sigma_a : array_like
        Absorption cross-section values at sample points
    sigma_f : array_like or None
        Fission cross-section values at sample points
    cleanup_tol : float
        Tolerance for cleanup (default 1e-13)
    space : str
        Space for transformation ('sqrt_E' or 'E')
    log : bool
        Whether to print information

    Returns
    -------
    z, fs, fa, ff, w : arrays
        Cleaned support points, function values, and weights
    """
    eps = 1e-13
    has_fission = sigma_f is not None and ff is not None

    # Transform grid
    if space == "sqrt_E":
        grid = np.sqrt(Z)
    elif space == "E":
        grid = Z
    else:
        raise ValueError(f"Unknown space: {space}")

    # Compute poles and zeros for total function
    f_total = fs + fa
    if has_fission:
        f_total = f_total + ff

    pol, res, zer = prz(z, f_total, w)

    if log:
        print(f"  Before cleanup: {len(pol)} poles, {len(zer)} zeros")

    # Find pole-zero pairs that are very close
    poles_to_remove = []
    zeros_matched = set()

    for i, p in enumerate(pol):
        if len(zer) == 0:
            break
        # Find closest zero to this pole
        distances = np.abs(zer - p)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # If pole and zero are extremely close and zero hasn't been matched yet
        if min_dist < cleanup_tol and min_idx not in zeros_matched:
            poles_to_remove.append(i)
            zeros_matched.add(min_idx)
            if log > 1:
                print(
                    f"    Pole-zero pair found: pole={p:.6e}, zero={zer[min_idx]:.6e}, dist={min_dist:.3e}"
                )

    if len(poles_to_remove) == 0:
        if log:
            print("  No close pole-zero pairs found")
        return z, fs, fa, ff, w

    if log:
        print(f"  Found {len(poles_to_remove)} pole-zero pairs to remove")

    # Find support points closest to these poles
    indices_to_remove = []
    for pole_idx in poles_to_remove:
        p = pol[pole_idx]
        distances = np.abs(z - p)
        closest_idx = np.argmin(distances)
        indices_to_remove.append(closest_idx)

    # Remove duplicates and sort
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)

    if log:
        print(f"  Removing {len(indices_to_remove)} support points")

    # Remove support points
    for idx in indices_to_remove:
        z = np.delete(z, idx)
        fs = np.delete(fs, idx)
        fa = np.delete(fa, idx)
        if has_fission:
            ff = np.delete(ff, idx)

    m = len(z)
    if m == 0:
        warnings.warn("No support points left after cleanup", UserWarning)
        return z, fs, fa, ff, w

    # Rebuild approximation
    # Remove coincident points from grid
    delta = grid[:, None] - z[None, :]
    row_mask = ~np.any(np.isclose(delta, 0.0, atol=1e-14), axis=1)

    grid_clean = grid[row_mask]
    sigma_s_clean = sigma_s[row_mask]
    sigma_a_clean = sigma_a[row_mask]
    if has_fission:
        sigma_f_clean = sigma_f[row_mask] if sigma_f is not None else None

    # Recompute delta
    delta = grid_clean[:, None] - z[None, :]

    # Build Loewner matrices
    A_s = (sigma_s_clean[:, None] - fs[None, :]) / delta
    A_a = (sigma_a_clean[:, None] - fa[None, :]) / delta

    # Apply relative weighting
    rel_w_s = 1.0 / np.maximum(np.abs(sigma_s_clean), eps)
    rel_w_a = 1.0 / np.maximum(np.abs(sigma_a_clean), eps)
    A_s *= rel_w_s[:, None]
    A_a *= rel_w_a[:, None]

    if has_fission:
        A_f = (sigma_f_clean[:, None] - ff[None, :]) / delta
        rel_w_f = 1.0 / np.maximum(np.abs(sigma_f_clean), eps)
        A_f *= rel_w_f[:, None]
        L = np.vstack((A_s, A_a, A_f))
    else:
        L = np.vstack((A_s, A_a))

    # Solve for new weights
    try:
        _, _, Vh = svd(L, full_matrices=False)
        w = Vh[-1, :].conj() if np.iscomplexobj(Vh) else Vh[-1, :]
    except Exception as e:
        warnings.warn(f"SVD failed: {e}", UserWarning)
        return z, fs, fa, ff, w

    if log:
        # Verify result
        f_total_new = fs + fa
        if has_fission:
            f_total_new = f_total_new + ff
        pol_new, _, zer_new = prz(z, f_total_new, w)
        print(f"  After cleanup: {len(pol_new)} poles, {len(zer_new)} zeros")

    if has_fission:
        return z, fs, fa, ff, w
    else:
        return z, fs, fa, None, w


# Helper function to integrate with your existing code
def apply_cleanup2_to_aaa(
    E,
    sigma_s,
    sigma_a,
    z,
    fs,
    fa,
    w,
    sigma_f=None,
    ff=None,
    cleanup_tol=1e-3,
    space="E",
    log=False,
):
    """
    Apply cleanup2 to AAA approximation results.

    Returns cleaned z, fs, fa, ff (if applicable), and w
    """
    if log:
        print("Applying simple Froissart cleanup...")

    z_clean, fs_clean, fa_clean, ff_clean, w_clean = cleanup(
        z,
        fs,
        fa,
        ff,
        w,
        E,
        sigma_s,
        sigma_a,
        sigma_f,
        cleanup_tol=cleanup_tol,
        space=space,
        log=log,
    )

    if log:
        print(f"  Reduced from {len(z)} to {len(z_clean)} support points")

    return z_clean, fs_clean, fa_clean, ff_clean, w_clean


def prz(z, f, w):
    """
    Compute poles, residues, and zeros of rational function in barycentric form.
    This is an adapter that matches MATLAB's prz signature.

    Parameters
    ----------
    z : array_like
        Support points (already in transformed space)
    f : array_like
        Function values at support points
    w : array_like
        Weights

    Returns
    -------
    pol : array
        Poles of the rational approximation
    res : array
        Residues at the poles
    zer : array
        Zeros of the rational approximation
    """
    m = len(z)

    # Build matrices for generalized eigenvalue problem
    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0

    # Matrix E for poles
    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = w
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(z)

    # Compute poles via generalized eigenvalue problem
    lam, _ = la.eig(E, B)
    pol = lam[np.isfinite(lam)]  # finite eigenvalues only

    # Compute residues using formula for residue of quotient of analytic functions
    res = np.zeros_like(pol, dtype=complex)
    for i, p in enumerate(pol):
        num = np.sum(w * f / (p - z))
        dprime = -np.sum(w / (p - z) ** 2)
        res[i] = num / dprime

    # Matrix E for zeros (numerator)
    E_zer = np.zeros((m + 1, m + 1), dtype=complex)
    E_zer[0, 1:] = w * f  # Note: element-wise multiplication
    E_zer[1:, 0] = 1.0
    E_zer[1:, 1:] = np.diag(z)

    # Compute zeros via generalized eigenvalue problem
    lam_zer, _ = la.eig(E_zer, B)
    zer = lam_zer[np.isfinite(lam_zer)]  # finite eigenvalues only

    return pol, res, zer
