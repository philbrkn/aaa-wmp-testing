# conversion.py
"""
Enhanced proper_rational function with built-in remainder analysis.
"""

import numpy as np
import scipy.linalg as la
from .fitting import miaaa_xs


def proper_rational(z, wnum, wden, fz, bcf, Z,
                    pole_extraction=None,
                    max_poly_degree=0, n_pseudo_poles=2):
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
    physical_poles = przd_for_poles(z, wden)

    # Compute residues via Cauchy matrices
    # Cnum: (n_poles, m) matrix with entries 1/(pole_i - z_j)
    Cnum = 1.0 / (physical_poles[:, np.newaxis] - z[np.newaxis, :])

    # Cden: derivative of denominator at poles
    # -d/dp sum_j w_j/(p - z_j) = sum_j w_j/(p - z_j)^2
    Cden = -1.0 * Cnum**2

    # Calculate residues for each function
    physical_res = np.zeros((len(physical_poles), k), dtype=np.complex128)
    for i in range(k):
        # Residue = numerator(pole) / denominator'(pole)
        num_at_poles = Cnum @ (wnum[i, :] * fz[i, :])
        denom_deriv = Cden @ wden

        # Avoid division by zero
        mask = np.abs(denom_deriv) > 1e-14
        physical_res[mask, i] = num_at_poles[mask] / denom_deriv[mask]

    # Evaluate partial fraction part on full grid Z
    # CC: (len(Z), n_poles) matrix with entries 1/(Z_i - pole_j)
    CC = 1.0 / (Z[:, np.newaxis] - physical_poles[np.newaxis, :])

    # pra: (k, len(Z)) partial fraction approximation
    pra = physical_res.T @ CC.T

    # Calculate remainder
    remainder = bcf - pra
    # Initialize output
    poles = physical_poles.copy()
    res = physical_res.copy()
    bestpra = pra.copy()
    info = {"method": pole_extraction}
    
    if pole_extraction == "polynomial" and max_poly_degree > 0:
        # Fit polynomial to remainder
        poly_coeffs = []
        for i in range(k):
            if np.max(np.abs(remainder[i, :])) > 1e-12:
                # Fit polynomial
                p = np.polyfit(Z.real if np.allclose(Z.imag, 0) else Z, 
                              remainder[i, :], max_poly_degree)
                poly_part = np.polyval(p, Z)
                bestpra[i, :] += poly_part
                poly_coeffs.append(p)
            else:
                poly_coeffs.append(None)
        info["poly_coeffs"] = poly_coeffs
        
    elif pole_extraction == "pseudo_pole" and n_pseudo_poles > 0:
        # Z_real = Z.real if np.iscomplexobj(Z) else Z
        # Z_min, Z_max = np.min(Z_real), np.max(Z_real)
        # Z_range = Z_max - Z_min
        
        # # g = 3 * Z_range
        # # mask_supports = (Z_real >= Z_min - g) & (Z_real <= Z_max + g)
        # # mask_fit = (Z_real >= Z_min) & (Z_real <= Z_max)
        # # Z_real_fit = Z_real[mask_fit]

        # w_p, z_p, fz_p, R_p, err_hist_p = miaaa_xs(
        #     Z,
        #     list(remainder),
        #     space="E",
        #     rtol=1e-4,
        #     mmax=20,
        #     log=2,
        #     # fit_mask=mask_fit,
        #     # core_mask=mask_supports,
        #     normalize=True,
        #     greedy_metric="relative",  # "relative" or "absolute_sum"
        # )
        # pseudo_poles, pseudo_residues = extract_poles_residues(w_p, z_p, fz_p)

        # # Append pseudo-poles to physical poles
        # poles = np.concatenate([poles, pseudo_poles])
        # pseudo_residues = np.array(pseudo_residues).T
        # res = np.vstack([res, pseudo_residues])

        # SIMPLIFIED PSEUDO-POLE APPROACH
        # Place pseudo-poles geometrically outside the domain
        
        Z_real = Z.real if np.iscomplexobj(Z) else Z
        Z_min, Z_max = np.min(Z_real), np.max(Z_real)
        Z_range = Z_max - Z_min
        
        # Strategy 1: Place pseudo-poles logarithmically outside domain
        # This gives better coverage for wide energy ranges
        if Z_min > 0:  # For positive energy grids
            # Place poles logarithmically spaced below and above
            left_poles = Z_min * np.logspace(0.5, -2, n_pseudo_poles//2, base=10)[::-1]
            right_poles = Z_max * np.logspace(0.5, 1, (n_pseudo_poles+1)//2, base=10)
            pseudo_poles = np.concatenate([left_poles, right_poles])
        else:
            # For grids including zero, use linear spacing
            left_poles = Z_min - Z_range * np.linspace(0.5, 2.0, n_pseudo_poles//2)
            right_poles = Z_max + Z_range * np.linspace(0.5, 2.0, (n_pseudo_poles+1)//2)
            pseudo_poles = np.concatenate([left_poles, right_poles])
        
        # Ensure pseudo-poles are real for real problems
        if not np.iscomplexobj(Z) and not np.iscomplexobj(bcf):
            pseudo_poles = pseudo_poles.real
        
        # Fit residues using regularized least squares
        pseudo_residues = np.zeros((len(pseudo_poles), k), dtype=np.complex128)
        
        for i in range(k):
            remainder_i = remainder[i, :]
            max_remainder = np.max(np.abs(remainder_i))
            
            if max_remainder > 1e-12:
                # Build Cauchy matrix for pseudo-poles
                C_pseudo = 1.0 / (Z[:, np.newaxis] - pseudo_poles[np.newaxis, :])
                
                # Use Tikhonov regularization for stability
                # The regularization parameter scales with the remainder magnitude
                lambda_reg = 1e-10 * max_remainder
                
                # Solve normal equations with regularization
                # (C^T C + lambda*I) * residues = C^T * remainder
                CTC = C_pseudo.T @ C_pseudo
                CTr = C_pseudo.T @ remainder_i
                
                # Add regularization to diagonal
                CTC_reg = CTC + lambda_reg * np.eye(len(pseudo_poles))
                
                try:
                    # Solve using Cholesky decomposition for better numerical stability
                    pseudo_residues[:, i] = np.linalg.solve(CTC_reg, CTr)
                except np.linalg.LinAlgError:
                    # Fallback to least squares if Cholesky fails
                    pseudo_residues[:, i], _, _, _ = np.linalg.lstsq(
                        C_pseudo, remainder_i, rcond=1e-10
                    )
                
                # Update approximation
                bestpra[i, :] += C_pseudo @ pseudo_residues[:, i]
        
        # Check quality of pseudo-pole fit
        final_remainder = bcf - bestpra
        improvement = np.max(np.abs(remainder)) - np.max(np.abs(final_remainder))
        
        #DEBUG
        print(f"Z range: [{Z_min:.3e}, {Z_max:.3e}]")
        print(f"Pseudo-poles: {pseudo_poles}")
        print(f"Min distance to grid: {np.min(np.abs(Z[:, None] - pseudo_poles[None, :])):.3e}")
        
        print(f"Pseudo-poles: Added {len(pseudo_poles)} poles")
        print(f"  Remainder before: {np.max(np.abs(remainder)):.3e}")
        print(f"  Remainder after:  {np.max(np.abs(final_remainder)):.3e}")
        print(f"  Improvement:      {improvement:.3e}")
        
        # Append pseudo-poles to physical poles
        if len(poles) > 0:
            poles = np.concatenate([poles, pseudo_poles])
            res = np.vstack([res, pseudo_residues])
        else:
            poles = pseudo_poles
            res = pseudo_residues
        
        info["pseudo_poles"] = pseudo_poles
        info["pseudo_residues"] = pseudo_residues
        info["n_pseudo_poles"] = n_pseudo_poles
    else:
        info["poly_coeffs"] = [None] * k
    
    # Create function handles for evaluation
    pr_handles = []
    for i in range(k):
        if pole_extraction == "polynomial":
            pr_handles.append(create_pr_function(
                poles, res[:, i], 
                info.get("poly_coeffs", [None]*k)[i]
            ))
        else:
            pr_handles.append(create_pr_function(poles, res[:, i], None))
    
    # Return in appropriate format
    if single_function:
        return poles, res[:, 0], bestpra[0, :], pr_handles[0], info
    else:
        return poles, res, bestpra, pr_handles, info


def przd_for_poles(z, w, deflation_tol=1e-10):
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
    while np.abs(np.sum(rv)) < deflation_tol and len(rv) > 0:
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
    print(f"Deflated {count} poles at tolerance {deflation_tol}")

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


#TODO: i dont believe this is necessary
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
