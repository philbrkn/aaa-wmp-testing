from math import sqrt
from pathlib import Path

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
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


def miaaa_xs(
    E,
    channels,
    space="E",
    method="full_svd",
    rtol=1e-13,
    mmax=100,
    log=False,
    fit_mask=None,
    core_mask=None,
    normalize=False,
    lawson_iter=0,
    greedy_metric="relative",  # "relative" or "absolute_sum"
):
    """
    Multi-Input AAA for cross-section fitting with common poles.

    Parameters
    ----------
    E : ndarray
        Energy grid points.
    channels : list of ndarray
        List of cross-section arrays [sigma_s, sigma_a, sigma_f, ...].
        Each array should have shape matching E.
    space : str
        "E" or "sqrt_E" for the interpolation space.
    method : str
        SVD method: "full_svd", "qr+svd", or "randomized_svd".
    rtol : float
        Relative tolerance for convergence.
    mmax : int
        Maximum number of support points.
    log : bool/int
        Verbosity level.
    fit_mask : ndarray, optional
        Boolean mask for points to include in fitting.
    core_mask : ndarray, optional
        Boolean mask for core region points.
    normalize : bool
        Whether to normalize channels before fitting.
    lawson_iter : int
        Number of Lawson iterations (0 to skip).
    greedy_metric : str
        "relative" for max relative error (like aaa_xs),
        "absolute_sum" for sum of absolute errors (like MIAAA).

    Returns
    -------
    tuple
        w : Barycentric weights (denominator)
        z : Support points
        fz : Function values at support points (k x m)
        R : Reconstructed functions (k x n)
        err_hist : Error history
    """
    # Setup
    n = E.shape[0]
    k = len(channels)  # Number of channels

    # Convert channels to matrix F (k x n)
    F = np.array(channels, dtype=np.complex128)

    # Grid transformation
    if space == "sqrt_E":
        grid = np.sqrt(E)
    elif space == "E":
        grid = E
    else:
        raise ValueError(f"Unknown space: {space}")

    # Masks
    if fit_mask is None:
        fit_mask = np.ones(n, dtype=bool)
    if core_mask is None:
        core_mask = fit_mask.copy()

    J_fit = np.flatnonzero(fit_mask).astype(int)
    J_core = np.flatnonzero(core_mask).astype(int)

    # Row weights for continuous LS (approximate integral)
    ds = np.zeros_like(grid)
    ds[:-1] += 0.5 * np.diff(grid)
    ds[1:] += 0.5 * np.diff(grid)

    # Normalize channels if requested
    norms = np.ones(k)
    if normalize:
        norms = np.linalg.norm(F, axis=1)
        norms[norms == 0] = 1.0
        F = F / norms[:, None]

    # Initialize
    z_list = []
    fz = np.empty((k, 0), dtype=np.complex128)
    J = np.arange(n, dtype=int)  # All indices initially
    eps = 1e-13

    # Initial approximation: channel-wise mean
    R = np.mean(F, axis=1, keepdims=True) * np.ones((k, n))
    err_hist = []

    # Greedy support selection
    for m in range(mmax):
        # Compute errors based on chosen metric
        if greedy_metric == "relative":
            # Relative error per channel
            errs = []
            for i in range(k):
                rel_err = np.abs(F[i] - R[i]) / np.maximum(np.abs(F[i]), eps)
                errs.append(rel_err)
            err = np.maximum.reduce(errs)  # Max across channels
        else:  # absolute_sum
            # Sum of absolute deviations across channels
            dev = np.abs(F - R)
            err = np.sum(dev, axis=0)

        # Record max error for convergence check
        max_err = np.max(err[J_core])
        err_hist.append(max_err)

        if log:
            print(f"  m={m}, max_err={max_err:.3e}, target={rtol:.3e}")

        # Check convergence
        if greedy_metric == "relative" and max_err <= rtol:
            if log:
                print(f"Converged at m={m} (err={max_err:.3e} ≤ rtol={rtol:.1e})")
            break
        elif greedy_metric == "absolute_sum" and np.max(np.abs(F - R)) < rtol:
            if log:
                print(f"Converged at m={m}")
            break

        # Select next support point from fit region
        candidate_errs = err[J_fit]
        if len(candidate_errs) == 0:
            break
        jpos = np.argmax(candidate_errs)
        j_star = J_fit[jpos]

        # Add to support
        z_list.append(grid[j_star])
        fz = np.hstack([fz, F[:, [j_star]]])

        # Remove from candidate sets
        J_fit = J_fit[J_fit != j_star]
        J_core = J_core[J_core != j_star]

        # Convert to arrays
        z = np.array(z_list)

        # Build stacked Loewner matrix for non-support points
        if len(J_fit) == 0:
            break

        # Cauchy matrix for non-support rows
        delta = grid[J_fit][:, None] - z[None, :]

        # Build per-channel Loewner blocks with weighting
        L_blocks = []
        for i in range(k):
            # Loewner matrix for channel i
            Li = (F[i, J_fit][:, None] - fz[i, :][None, :]) / delta

            # Row weighting (continuous LS)
            Li = ds[J_fit, None] * Li

            # # Column weighting (relative to function values at supports)
            # col_weights = 1.0 / np.maximum(np.abs(fz[i, :]), eps)
            # Li = Li * col_weights[None, :]

            # # Additional relative error row weighting
            row_weights = 1.0 / np.maximum(np.abs(F[i, J_fit]), eps)
            Li = row_weights[:, None] * Li

            L_blocks.append(Li)

        # Stack all channels
        L = np.vstack(L_blocks)

        # SVD to find common weights
        if method == "full_svd":
            _, _, Vh = svd(L, full_matrices=False)
        elif method == "qr+svd":
            Q, R_mat = np.linalg.qr(L, mode="reduced")
            _, _, Vh = svd(R_mat, full_matrices=False)
        elif method == "randomized_svd":
            from sklearn.utils.extmath import randomized_svd

            _, _, Vh = randomized_svd(
                L, n_components=min(20, L.shape[1]), random_state=42, n_iter=7
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract weights (last singular vector)
        w = Vh[-1, :]

        # Evaluate approximation on full grid
        R = evaluate_miaaa(grid, w, z, fz, space)

    # Lawson iteration (optional)
    if lawson_iter > 0 and len(z) > 0:
        w, R = lawson_iteration(grid, F, z, fz, w, J, lawson_iter, space, log)

    # Undo normalization
    if normalize:
        fz = fz * norms[:, None]
        R = R * norms[:, None]

    return w, z, fz, R, err_hist



def evaluate_miaaa(grid, w, z, fz, space="E"):
    """
    Evaluate multi-channel AAA approximation using barycentric formula.
    
    Parameters
    ----------
    grid : ndarray
        Evaluation points (in transformed space).
    w : ndarray
        Barycentric weights (length m).
    z : ndarray
        Support points (length m).
    fz : ndarray
        Function values at supports (k x m).
    space : str
        Interpolation space.
    
    Returns
    -------
    R : ndarray
        Evaluated functions (k x len(grid)).
    """
    k, m = fz.shape
    n = len(grid)
    R = np.zeros((k, n), dtype=np.complex128)
    
    # Build full Cauchy matrix
    C = 1.0 / (grid[:, None] - z[None, :])
    
    # Find support point indices (where grid matches z)
    support_indices = []
    for j, zj in enumerate(z):
        idx = np.where(np.abs(grid - zj) < 1e-14)[0]
        if len(idx) > 0:
            support_indices.extend([(idx[0], j)])
    
    # Create mask for non-support points
    is_support = np.zeros(n, dtype=bool)
    for idx, _ in support_indices:
        is_support[idx] = True
    non_support = ~is_support
    
    # Compute denominator for non-support points
    D = C @ w
    
    # Evaluate each channel
    for i in range(k):
        # Numerator
        N = C @ (w * fz[i, :])
        
        # Non-support points: use barycentric formula
        valid = non_support & (np.abs(D) > 1e-300)
        R[i, valid] = N[valid] / D[valid]
        
        # Support points: exact interpolation
        for idx, j in support_indices:
            R[i, idx] = fz[i, j]
    
    return R


def lawson_iteration(grid, F, z, fz, w_init, J_all, max_iter, space, log):
    """
    Full Lawson iteration following the MATLAB implementation.

    Parameters
    ----------
    grid : ndarray
        Full grid in transformed space.
    F : ndarray
        Function values (k x n).
    z : ndarray
        Support points.
    fz : ndarray
        Function values at supports (k x m).
    w_init : ndarray
        Initial weights.
    J_all : ndarray
        All indices (used for Jz calculation).
    max_iter : int
        Maximum Lawson iterations.
    space : str
        Interpolation space.
    log : bool
        Verbosity.

    Returns
    -------
    w : ndarray
        Optimized weights (2m for separated num/denom if Lawson succeeds).
    R : ndarray
        Optimized approximation.
    """
    k, n = F.shape
    m = len(z)

    if max_iter <= 0:
        # No Lawson, return initial
        return w_init, evaluate_miaaa(grid, w_init, z, fz, space)

    # Find support point indices in grid
    Jz = []  # Indices where grid points are support points
    J = []  # Indices where grid points are NOT support points

    for i in range(n):
        is_support = False
        for zj in z:
            if np.abs(grid[i] - zj) < 1e-14:
                is_support = True
                Jz.append(i)
                break
        if not is_support:
            J.append(i)

    J = np.array(J, dtype=int)
    Jz = np.array(Jz, dtype=int)

    # Build full Cauchy matrix
    C = 1.0 / (grid[:, None] - z[None, :])

    # Initialize Lawson variables
    gamma = 1.0
    lw = np.ones(n) / np.sqrt(n)  # Lawson weights, normalized

    # Initial approximation
    bcr = evaluate_miaaa(grid, w_init, z, fz, space)
    bestbcr = bcr.copy()
    bestw = np.concatenate([w_init, w_init])  # Store as [w_den, w_num]
    maxerror = np.max(np.abs(F - bestbcr))

    # Build non-interpolatory Loewner matrix
    # This separates numerator and denominator weights
    L_blocks = []

    # Support row replacement matrix
    Lsupp = np.hstack([np.eye(m), -np.eye(m)])

    for i in range(k):
        # Build diagonal matrix of function values
        # Li has shape (n, 2m): [diag(f_i)*C, -C*diag(fz_i)]
        Li_left = F[i, :, np.newaxis] * C  # (n, m)
        Li_right = -C * fz[i, :][np.newaxis, :]  # (n, m)
        Li = np.hstack([Li_left, Li_right])  # (n, 2m)

        # Replace support rows with interpolation conditions
        if len(Jz) > 0:
            Li[Jz, :] = Lsupp

        L_blocks.append(Li)

    L = np.vstack(L_blocks)  # (k*n, 2m)

    if log:
        print(f"  Starting Lawson iteration (max {max_iter} iterations)")

    # Lawson iterations
    for it in range(max_iter):
        # Build diagonal weight matrix
        lws = np.tile(lw, k)  # Repeat for each channel
        d = np.sqrt(lws)  # Square root for weighting

        # Weighted least squares
        Lw = d[:, np.newaxis] * L

        # SVD to find weights
        _, _, Vh = svd(Lw, full_matrices=False)
        w = Vh[-1, :]  # Last right singular vector

        # Split weights
        w_den = w[:m]
        w_num = w[m:]

        # Evaluate new approximation
        lbcr = np.zeros((k, n), dtype=np.complex128)

        if len(J) > 0:
            # Non-support points
            D = C[J, :] @ w_den
            valid = np.abs(D) > 1e-300
            J_valid = J[valid]

            for i in range(k):
                N = C[J, :] @ (w_num * fz[i, :])
                lbcr[i, J_valid] = N[valid] / D[valid]

        # Support points: use ratio of weights
        if len(Jz) > 0:
            for i in range(k):
                for idx in Jz:
                    # Find which support point this corresponds to
                    for j, zj in enumerate(z):
                        if np.abs(grid[idx] - zj) < 1e-14:
                            if np.abs(w_den[j]) > 1e-300:
                                lbcr[i, idx] = w_num[j] * F[i, idx] / w_den[j]
                            else:
                                lbcr[i, idx] = F[i, idx]
                            break

        # Compare with old approximation
        lmaxerror = np.max(np.abs(F - lbcr))

        if lmaxerror < maxerror:
            if log:
                print(
                    f"    Lawson iter {it+1}: improved from {maxerror:.3e} to {lmaxerror:.3e}"
                )
            maxerror = lmaxerror
            bestbcr = lbcr.copy()
            bestw = w.copy()

        # Update Lawson weights
        col_err = np.max(np.abs(F - lbcr), axis=0)  # Max error across channels
        testlw = lw * (col_err**gamma)

        # Handle infinities and NaNs
        mask_inf = np.isinf(testlw)
        mask_nan = np.isnan(testlw)

        if np.any(mask_inf):
            testlw[mask_inf] = np.max(testlw[~mask_inf]) if np.any(~mask_inf) else 1.0

        if np.any(mask_nan):
            testlw[mask_nan] = np.mean(testlw[~mask_nan]) if np.any(~mask_nan) else 1.0

        # Avoid zeros
        mask_zero = testlw == 0
        if np.any(mask_zero):
            testlw[mask_zero] = (
                np.mean(testlw[~mask_zero]) if np.any(~mask_zero) else 1e-10
            )

        # Normalize
        testlw = testlw / np.linalg.norm(testlw)

        # Check convergence
        if np.linalg.norm(testlw - lw, ord=np.inf) < 1e-8:
            if log:
                print(f"    Lawson converged at iteration {it+1}")
            break

        lw = testlw

    # Return best weights (as single vector for compatibility)
    # For simple compatibility, return just denominator weights
    # Full implementation would need to handle separated num/denom
    if max_iter > 0:
        return bestw[:m], bestbcr  # Return denominator weights and best approximation
    else:
        return w_init, bcr


def extract_poles_residues(w, z, fz):
    """
    Extract poles and residues from AAA barycentric representation.

    Parameters
    ----------
    w : ndarray
        Barycentric weights (length m).
    z : ndarray
        Support points (length m).
    fz : ndarray
        Function values at support points.
        Shape (m,) for single function or (k, m) for multiple functions.

    Returns
    -------
    poles : ndarray
        Poles of the rational approximation.
    residues : ndarray or list of ndarray
        Residues for each function.
        Single array for scalar function, list of arrays for multiple functions.
    """
    m = len(z)

    # Handle both single and multi-function cases
    if fz.ndim == 1:
        fz = fz.reshape(1, -1)
        single_function = True
    else:
        single_function = False

    k = fz.shape[0]  # number of functions

    # Build companion matrix for denominator (Froissart doublet removal)
    # This finds poles as eigenvalues of generalized eigenvalue problem
    B = np.eye(m + 1, dtype=np.complex128)
    # B[1:, 0] = 1
    B[0, 0] = 0

    # Build matrix E for generalized eigenvalue problem
    E = np.zeros((m + 1, m + 1), dtype=np.complex128)
    E[0, 1:] = w
    E[1:, 0] = 1
    E[1:, 1:] = np.diag(z)

    # Solve generalized eigenvalue problem
    eigenvalues = la.eigvals(E, B)

    # Remove poles at infinity
    finite_mask = ~np.isinf(eigenvalues)
    poles = eigenvalues[finite_mask]

    # Calculate residues for each function
    all_residues = []

    for i in range(k):
        # For each pole, calculate residue using formula:
        # res_j = lim_{s->p_j} (s-p_j) * N(s)/D(s)
        # where N(s) = sum_i w_i*f_i/(s-z_i) and D(s) = sum_i w_i/(s-z_i)

        residues = np.zeros_like(poles, dtype=np.complex128)

        for j, pole in enumerate(poles):
            # Calculate derivative of denominator at pole
            # D'(p) = -sum_i w_i/(p-z_i)^2
            D_prime = -np.sum(w / (pole - z) ** 2)

            # Calculate numerator at pole using L'Hopital
            # N(p) = sum_i w_i*f_i/(p-z_i)
            N_val = np.sum(w * fz[i, :] / (pole - z))

            # Residue = N(p) / D'(p)
            if np.abs(D_prime) > 1e-14:
                residues[j] = N_val / D_prime

        all_residues.append(residues)

    if single_function:
        return poles, all_residues[0]
    else:
        return poles, all_residues


def przd_for_poles(z, w, tol=1e-10):
    """
    Python implementation of the MATLAB przd function for finding poles.
    Uses generalized eigenvalue problem with deflation.

    Parameters
    ----------
    z : ndarray
        Support points (poles of barycentric form).
    w : ndarray
        Weights (residues of barycentric form).
    tol : float
        Tolerance for sum of residues to trigger deflation.

    Returns
    -------
    poles : ndarray
        Poles of the rational function.
    """
    pv = z.copy()
    rv = w.copy()
    count = 0

    # Deflation loop - remove pole-zero pairs when sum of residues is small
    while np.abs(np.sum(rv)) < tol and len(rv) > 0:
        count += 1

        # Remove first residue and pole
        rv = rv[1:]
        first_pv = pv[0]
        pv = pv[1:]

        if len(pv) == 0:
            break

        # Recalculate residues over new set of poles
        fr_pv = first_pv - pv
        rv = rv * fr_pv

        # Normalize
        norm_rv = np.linalg.norm(rv)
        if norm_rv > 0:
            rv = rv / norm_rv

    if len(pv) == 0:
        return np.array([])

    # Build and solve generalized eigenvalue problem
    m = len(pv)
    B = np.eye(m + 1, dtype=np.complex128)
    B[0, 0] = 0

    E = np.zeros((m + 1, m + 1), dtype=np.complex128)
    E[0, 1:] = rv
    E[1:, 0] = 1
    E[1:, 1:] = np.diag(pv)

    # Solve for poles
    poles = la.eigvals(E, B)

    # Remove poles at infinity
    poles = poles[~np.isinf(poles)]

    if count > 0:
        print(f"{count} deflations performed")

    return poles


def proper_rational(z, wnum, wden, fz, bcf, Z, maxpolydegree=0):
    """
    Convert barycentric rational approximation to proper rational form.

    This function extracts poles and residues from a barycentric representation
    and optionally fits a polynomial to the remainder.

    Parameters
    ----------
    z : ndarray
        Support points from AAA (length m).
    wnum : ndarray
        Numerator weights. For single function, same as wden.
        For multiple functions after Lawson, shape (m,) or (k, m).
    wden : ndarray
        Denominator weights (length m).
    fz : ndarray
        Function values at support points.
        Shape (m,) for single function or (k, m) for multiple.
    bcf : ndarray
        Barycentric function evaluations on full grid Z.
        Shape (len(Z),) for single or (k, len(Z)) for multiple.
    Z : ndarray
        Full evaluation grid.
    maxpolydegree : int
        Maximum polynomial degree to fit remainder (0 = no polynomial).

    Returns
    -------
    poles : ndarray
        Extracted poles.
    residues : ndarray
        Residues for each function (shape (n_poles,) or (n_poles, k)).
    bestpra : ndarray
        Best proper rational approximation on grid Z.
    pr_handles : list
        List of callable functions for evaluation.
    bestpoly : list
        Polynomial coefficients for each function (if maxpolydegree > 0).
    """
    # Handle input dimensions
    if fz.ndim == 1:
        fz = fz.reshape(1, -1)
        bcf = bcf.reshape(1, -1)
        single_function = True
        k = 1
    else:
        single_function = False
        k = fz.shape[0]

    # Handle wnum dimensions
    if wnum.ndim == 1:
        # Same weights for numerator and denominator (no Lawson)
        wnum = np.tile(wnum, (k, 1))
    elif wnum.shape[0] != k:
        # Broadcast if needed
        wnum = np.tile(wnum.reshape(1, -1), (k, 1))

    # Extract poles using the przd eigenvalue method
    poles = przd_for_poles(z, wden)

    # Compute residues via Cauchy matrices
    # Cnum: (n_poles, m) matrix with entries 1/(pole_i - z_j)
    Cnum = 1.0 / (poles[:, np.newaxis] - z[np.newaxis, :])

    # Cden: derivative of denominator at poles
    # -d/dp sum_j w_j/(p - z_j) = sum_j w_j/(p - z_j)^2
    Cden = -1.0 * Cnum**2

    # Calculate residues for each function
    res = np.zeros((len(poles), k), dtype=np.complex128)
    for i in range(k):
        # Residue = numerator(pole) / denominator'(pole)
        num_at_poles = Cnum @ (wnum[i, :] * fz[i, :])
        denom_deriv = Cden @ wden

        # Avoid division by zero
        mask = np.abs(denom_deriv) > 1e-14
        res[mask, i] = num_at_poles[mask] / denom_deriv[mask]

    # Evaluate partial fraction part on full grid Z
    # CC: (len(Z), n_poles) matrix with entries 1/(Z_i - pole_j)
    CC = 1.0 / (Z[:, np.newaxis] - poles[np.newaxis, :])

    # pra: (k, len(Z)) partial fraction approximation
    pra = res.T @ CC.T

    # Calculate remainder
    remainder = bcf - pra
    bestpra = pra.copy()
    bestpoly = [None] * k

    # Instead of polynomial fitting, add pseudopoles
    if maxpolydegree > 0:  # Reuse this flag to mean "add pseudopoles"
        # Choose pseudopole locations far outside domain
        Z_min, Z_max = np.min(Z), np.max(Z)
        Z_range = Z_max - Z_min
        
        # Place pseudopoles symmetrically outside domain
        n_pseudo = 2  # Start with 2 pseudopoles for a constant
        pseudo_locs = np.array([
            Z_min - 10*Z_range,  # Far below
            Z_max + 10*Z_range   # Far above
        ])

        # For each channel, fit pseudopole residues to match the remainder
        pseudo_residues = np.zeros((len(pseudo_locs), k), dtype=complex)

        for i in range(k):
            # Build matrix for pseudopole contribution
            C_pseudo = 1.0 / (Z[:, None] - pseudo_locs[None, :])

            # Least squares fit: find residues that approximate remainder
            pseudo_residues[:, i], _, _, _ = np.linalg.lstsq(
                C_pseudo, remainder[i, :], rcond=None
            )

        # Append pseudopoles to physical poles
        poles = np.concatenate([poles, pseudo_locs])
        res = np.vstack([res, pseudo_residues])

    # Create function handles for evaluation
    pr_handles = []
    for i in range(k):
        pr_handles.append(create_pr_function(poles, res[:, i], bestpoly[i]))

    # Return in appropriate format
    if single_function:
        return poles, res[:, 0], bestpra[0, :], pr_handles[0], bestpoly[0]
    else:
        return poles, res, bestpra, pr_handles, bestpoly


def create_pr_function(poles, residues, polycoeffs=None):
    """
    Create a callable function for proper rational evaluation.

    Parameters
    ----------
    poles : ndarray
        Poles of the rational function.
    residues : ndarray
        Residues corresponding to poles.
    polycoeffs : ndarray or None
        Polynomial coefficients (highest degree first).

    Returns
    -------
    callable
        Function that evaluates the proper rational at given points.
    """

    def pr_eval(w):
        """Evaluate proper rational function at points w."""
        w = np.asarray(w)
        result = np.zeros_like(w, dtype=np.complex128)

        # Partial fraction part
        for pole, res in zip(poles, residues):
            result += res / (w - pole)

        # Polynomial part
        if polycoeffs is not None:
            result += np.polyval(polycoeffs, w)

        return result

    return pr_eval


def pfeval(w, poles, residues, polycoeffs=None):
    """
    Direct evaluation of proper rational (compatibility function).

    Parameters
    ----------
    w : array_like
        Points at which to evaluate.
    poles : ndarray
        Poles of the rational function.
    residues : ndarray
        Residues corresponding to poles.
    polycoeffs : ndarray or None
        Polynomial coefficients.

    Returns
    -------
    ndarray
        Function values at w.
    """
    w = np.asarray(w)
    result = np.zeros_like(w, dtype=np.complex128)

    # Partial fraction part
    for pole, res in zip(poles, residues):
        result += res / (w - pole)

    # Polynomial part
    if polycoeffs is not None:
        result += np.polyval(polycoeffs, w)

    return result


def validate_pole_residue_reconstruction(E, w, z, fz, poles, residues, space="E"):
    """
    Validate the pole-residue extraction by reconstructing the function.

    Parameters
    ----------
    E : ndarray
        Energy grid for evaluation.
    w, z, fz : arrays
        Barycentric representation.
    poles, residues : arrays
        Extracted poles and residues.
    space : str
        Interpolation space.

    Returns
    -------
    dict
        Dictionary with original, reconstructed values and errors.
    """
    # Transform grid if needed
    if space == "sqrt_E":
        grid = np.sqrt(E)
    else:
        grid = E

    # Original evaluation using barycentric
    if fz.ndim == 1:
        fz = fz.reshape(1, -1)

    k = fz.shape[0]
    n = len(grid)

    # Barycentric evaluation
    C = 1.0 / (grid[:, None] - z[None, :])
    D = C @ w

    original = np.zeros((k, n), dtype=np.complex128)
    for i in range(k):
        N = C @ (w * fz[i, :])
        mask = np.abs(D) > 1e-14
        original[i, mask] = N[mask] / D[mask]

    # Partial fraction evaluation
    reconstructed = np.zeros((k, n), dtype=np.complex128)

    if not isinstance(residues, list):
        residues = [residues]

    for i in range(k):
        for j, pole in enumerate(poles):
            if space == "sqrt_E":
                # In sqrt(E) space: res/(s - pole) where s = sqrt(E)
                reconstructed[i, :] += residues[i][j] / (grid - pole)
            else:
                # In E space
                reconstructed[i, :] += residues[i][j] / (E - pole)

    # Calculate errors
    abs_error = np.abs(original - reconstructed)
    rel_error = abs_error / (np.abs(original) + 1e-14)

    return {
        "original": original,
        "reconstructed": reconstructed,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "max_abs_error": np.max(abs_error),
        "max_rel_error": np.max(rel_error),
        "mean_abs_error": np.mean(abs_error),
        "mean_rel_error": np.mean(rel_error),
    }


# Example usage function
def example_pole_residue_extraction():
    """
    Example showing how to use the pole/residue extraction.
    """
    # Generate sample data
    E = np.logspace(-2, 4, 1000)

    # Create a rational function with known poles
    true_poles = np.array([10.0, 100.0, 1000.0])
    true_residues = np.array([1.0, 2.0, 3.0])

    # Generate function values
    f = np.zeros_like(E, dtype=complex)
    for p, r in zip(true_poles, true_residues):
        f += r / (E - p)

    # Run AAA to get barycentric representation
    # (This would come from your miaaa_xs function)
    # w, z, fz, _, _ = miaaa_xs(E, [f], ...)

    # For this example, let's assume we have w, z, fz from AAA
    # ... (you would get these from miaaa_xs)

    # Extract poles and residues
    # poles, residues = extract_poles_residues(w, z, fz)

    # Validate reconstruction
    # validation = validate_pole_residue_reconstruction(
    #     E, w, z, fz, poles, residues, space="E"
    # )

    print("Example setup complete")
    print(f"True poles: {true_poles}")
    print(f"True residues: {true_residues}")

    return E, f, true_poles, true_residues


def example_cross_section_pole_extraction():
    """
    Example of extracting poles and residues from cross-section data.
    """
    import numpy as np

    # Example: after running miaaa_xs
    print("=" * 60)
    print("Cross-Section Pole/Residue Extraction Example")
    print("=" * 60)

    # Suppose we have run miaaa_xs and obtained:
    # w_den, w_num, z, fz, bcf, _, _ = miaaa_xs(
    #     E, [sigma_s, sigma_a, sigma_f], space="sqrt_E", ...
    # )

    # Mock data for demonstration
    E = np.logspace(-2, 4, 1000)
    grid = np.sqrt(E)  # If using sqrt_E space

    # Mock barycentric data (would come from miaaa_xs)
    m = 10  # number of support points
    z = np.sort(np.random.choice(grid, m, replace=False))
    w_den = np.random.randn(m) + 1j * np.random.randn(m) * 0.01
    w_num = w_den.copy()  # Same if no Lawson iteration

    # Mock function values at supports for 3 channels
    k = 3  # sigma_s, sigma_a, sigma_f
    fz = np.random.randn(k, m) + 1j * np.random.randn(k, m) * 0.01

    # Mock barycentric evaluations on full grid
    bcf = np.random.randn(k, len(E))

    print(f"Number of channels: {k}")
    print(f"Number of support points: {m}")
    print(f"Grid size: {len(E)}")

    # Extract poles and residues using proper_rational
    poles, residues, pra, pr_handles, polycoeffs = proper_rational(
        z, w_num, w_den, fz, bcf, grid, maxpolydegree=0
    )

    print(f"\nExtracted {len(poles)} poles")
    print(f"Residue matrix shape: {residues.shape}")

    # The poles are common to all channels
    print("\nCommon poles (in sqrt(E) space):")
    for i, pole in enumerate(poles[:5]):  # Show first 5
        print(f"  Pole {i+1}: {pole:.3f}")

    # Each channel has different residues
    print("\nResidue strengths by channel:")
    for i in range(k):
        channel_name = ["sigma_s", "sigma_a", "sigma_f"][i]
        avg_residue = np.mean(np.abs(residues[:, i]))
        print(f"  {channel_name}: avg |residue| = {avg_residue:.3e}")

    # Function handles can be used for evaluation
    print("\nFunction handles created for each channel")

    # Convert back to E-space if needed
    poles_E = poles**2  # Since poles are in sqrt(E) space
    print(f"\nPoles in E-space: {poles_E[:3]}")

    # Reconstruction in partial fraction form:
    # sigma_i(E) = sum_j residues[j,i] / (sqrt(E) - poles[j]) + poly_i(sqrt(E))

    return poles, residues, pr_handles
