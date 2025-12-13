"""
Enhanced proper_rational function with built-in remainder analysis.
"""

from math import pi, sqrt

import numpy as np
import scipy.linalg as la
from openmc.data.multipole_old import WindowedMultipole
from scipy.special import wofz

K_BOLTZMANN = 8.617333262145e-5  # eV/K


def evaluate_simple(Z, poles, residues, poly_coeffs=None, fit_space="sqrt_E"):
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

    Z_array = np.atleast_1d(Z)
    if fit_space == "sqrt_E":
        s = np.sqrt(Z_array)  # Poles are in sqrt_E space
    else:
        s = Z_array  # Poles are in E space

    fissionable = residues.shape[0] == 3

    # Initialize cross sections
    elastic_xs = np.zeros_like(Z, dtype=float)
    absorption_xs = np.zeros_like(Z, dtype=float)
    fission_xs = np.zeros_like(Z, dtype=float) if fissionable else None

    # Add pole contributions
    for i, s_val in enumerate(s):
        # Using a vectorized operation is much faster than a second for-loop
        denominators = s_val - poles

        # Elastic (column 1)
        elastic_xs[i] = np.sum((residues[0] / denominators).real)

        # Absorption (column 2)
        absorption_xs[i] = np.sum((residues[1] / denominators).real)

        # Fission (column 3, if present)
        if fissionable:
            fission_xs[i] = np.sum((residues[2] / denominators).real)

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

    return np.asarray([elastic_xs, absorption_xs, fission_xs])


def evaluate_openmc_T(
    E, T, poles, residues, poly_coeffs=None, sqrtAWR=1.0, broaden_poly=False
):
    """
    OpenMC-compatible multipole evaluation with temperature dependence.

    Parameters
    ----------
    E : ndarray
        Energies (eV)
    T : float
        Temperature (K)
    poles : ndarray (M,)
        Poles in sqrt(E) space (WMP format, upper half-plane only)
    residues : ndarray (k, M)
        Residues divided by 1j (OpenMC storage convention)
    poly_coeffs : ndarray (k, n_poly), optional
        Polynomial coefficients for F(s)=E*sigma(E)
    sqrtAWR : float
        sqrt(AWR)
    broaden_poly : bool
        Whether to Doppler-broaden the polynomial (OpenMC-style)

    Returns
    -------
    sig_s, sig_a, sig_f : ndarray
    """

    E = np.asarray(E)
    poles = np.asarray(poles)
    residues = np.asarray(residues)

    sqrtE = np.sqrt(E)
    invE = 1.0 / E

    k = residues.shape[0]
    sig = np.zeros((k, len(E)))

    sqrtkT = sqrt(K_BOLTZMANN * T)

    # ------------------------------------------------------------------
    # Polynomial contribution (background)
    # ------------------------------------------------------------------
    if poly_coeffs is not None:
        if sqrtkT != 0.0 and broaden_poly:
            raise NotImplementedError(
                "Polynomial Doppler broadening requires "
                "_broaden_wmp_polynomials (OpenMC internal)."
            )
        else:
            temp = invE.copy()
            for q in range(poly_coeffs.shape[1]):
                sig += poly_coeffs[:, q, None] * temp
                temp *= sqrtE

    # ------------------------------------------------------------------
    # Pole contribution
    # ------------------------------------------------------------------
    if sqrtkT == 0.0:
        # -------- 0 K (asymptotic form) --------
        for j, p in enumerate(poles):
            psi = -1j / (p - sqrtE)
            contrib = psi * invE
            sig += (residues[:, j, None] * contrib).real

    else:
        # -------- Finite temperature (Faddeeva) --------
        dopp = sqrtAWR / sqrtkT
        for j, p in enumerate(poles):
            Z = (sqrtE - p) * dopp
            wval = _faddeeva(Z) * dopp * invE * sqrt(pi)
            sig += (residues[:, j, None] * wval).real

    # unpack
    if k == 3:
        return sig[0], sig[1], sig[2]
    elif k == 2:
        return sig[0], sig[1], None
    else:
        return tuple(sig)


def _faddeeva(z):
    z = np.asarray(z, dtype=np.complex128)

    out = np.empty_like(z, dtype=np.complex128)

    mask = np.angle(z) > 0
    out[mask] = wofz(z[mask])
    out[~mask] = -np.conj(wofz(np.conj(z[~mask])))

    return out
    # # OpenMC branch convention
    # if np.angle(z) > 0:
    #     return wofz(z)
    # else:
    #     return -np.conj(wofz(z.conjugate()))


# def evaluate_openmc(E, poles, residues, poly_coeffs=None):
#     E = np.asarray(E)
#     poles = np.asarray(poles)
#     residues = np.asarray(residues)
#
#     sig_s = np.zeros_like(E, dtype=float)
#     sig_a = np.zeros_like(E, dtype=float)
#     sig_f = np.zeros_like(E, dtype=float)
#
#     for i, e_pt in enumerate(E):
#         sqrtE = np.sqrt(e_pt)
#         invE = 1.0 / e_pt
#
#         # polynomial contribution to sigma: poly(s)/E
#         if poly_coeffs is not None and poly_coeffs.size > 0:
#             temp = invE
#             for q in range(poly_coeffs.shape[1]):
#                 sig_s[i] += poly_coeffs[0, q] * temp
#                 sig_a[i] += poly_coeffs[1, q] * temp
#                 if residues.shape[0] == 3:
#                     sig_f[i] += poly_coeffs[2, q] * temp
#                 temp *= sqrtE
#
#         # pole contribution
#         for j in range(len(poles)):
#             psi_chi = -1j / (poles[j] - sqrtE)
#             c_temp = psi_chi * invE
#             sig_s[i] += (residues[0, j] * c_temp).real
#             sig_a[i] += (residues[1, j] * c_temp).real
#             if residues.shape[0] == 3:
#                 sig_f[i] += (residues[2, j] * c_temp).real
#
#     return sig_s, sig_a, sig_f


def proper_rational(z, wnum, wden, fz, bcf, Z, pole_extraction=None, max_poly_degree=0):
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
    k = fz.shape[0]

    # Handle wnum dimensions
    if wnum.ndim == 1:
        # Same weights for numerator and denominator (no Lawson)
        wnum = np.tile(wnum, (k, 1))
    elif wnum.shape[0] != k:
        # Broadcast if needed
        wnum = np.tile(wnum.reshape(1, -1), (k, 1))

    # Extract poles using the przd eigenvalue method
    physical_poles = przd_for_poles(z, wden, deflation_tol=1e-10)

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
        mask_div = np.abs(denom_deriv) > 1e-14
        physical_res[mask_div, i] = num_at_poles[mask_div] / denom_deriv[mask_div]

    # Evaluate partial fraction part on full grid Z
    # # CC: (len(Z), n_poles) matrix with entries 1/(Z_i - pole_j)
    # CC = 1.0 / (Z[:, np.newaxis] - physical_poles[np.newaxis, :])

    # # pra: (k, len(Z)) partial fraction approximation
    # pra = physical_res.T @ CC.T

    physical_res = physical_res.T

    # Separate real and complex poles
    # res_transposed = physical_res.T
    # # Separate real and complex poles
    # real_idx = np.where(np.abs(physical_poles.imag) < 1e-10)[0]
    # complex_idx = np.where(np.abs(physical_poles.imag) >= 1e-10)[0]
    # # For complex poles, keep only those with positive imaginary part
    # # (the conjugates are implied)
    # conj_idx = complex_idx[physical_poles[complex_idx].imag > 0]
    # # Build WMP-compatible poles and residues
    # physical_poles = np.concatenate(
    #     [physical_poles[real_idx], physical_poles[conj_idx]]
    # )
    # physical_res = np.concatenate(
    #     [
    #         res_transposed[:, real_idx],  # Real pole residues as-is
    #         res_transposed[:, conj_idx]
    #         * 2,  # Complex residues doubled (for conjugate pair)
    #     ],
    #     axis=1,
    # )
    #
    # Calculate remainder
    CC = 1.0 / (Z[:, np.newaxis] - physical_poles[np.newaxis, :])
    pra = physical_res @ CC.T
    remainder = bcf - pra
    remainder = remainder.real

    # Initialize output
    poles = physical_poles.copy()
    res = physical_res.copy()

    if pole_extraction == "polynomial" and max_poly_degree > 0:
        info = {"method": pole_extraction}
        # Fit polynomial to remainder
        poly_coeffs = []
        for i in range(k):
            if np.max(np.abs(remainder[i, :])) > 1e-12:
                # Fit polynomial
                p = np.polyfit(
                    Z.real if np.allclose(Z.imag, 0) else Z,
                    remainder[i, :],
                    max_poly_degree,
                )
                poly_coeffs.append(p)
            else:
                poly_coeffs.append(None)
        info["poly_coeffs"] = poly_coeffs
    elif pole_extraction == "pseudo_pole":
        # info = fit_pseudopoles(Z, remainder, n_pseudo_poles, bcf, bestpra)
        info = {"method": "pseudo_pole", "poly_coeffs": None}
        p_poles, p_residues = fit_pseudopoles_adaptive(
            Z,
            remainder,
            bcf,
            max_poles=6,
            rtol=1e-6,
            verbose=False,
        )
        print(poles.shape, p_poles.shape, res.shape, p_residues.shape)
        # Append pseudo-poles to physical poles
        if len(p_poles) > 0:
            poles = np.concatenate([poles, p_poles])
            res = np.hstack([res, p_residues])

    else:
        info = {"method": None}
        info["poly_coeffs"] = [None] * k

    # Return in appropriate format
    # res = res / 1j
    return poles, res, remainder, info


def to_wmp_form(poles, residues, tol=1e-12):
    poles = np.asarray(poles)
    residues = np.asarray(residues)  # (k, m)

    keep_poles = []
    keep_res = []

    used = np.zeros(len(poles), dtype=bool)

    for i, p in enumerate(poles):
        if used[i]:
            continue

        if abs(p.imag) < tol:
            # real pole
            keep_poles.append(p)
            keep_res.append(residues[:, i])
            used[i] = True
        else:
            # complex pole: must have a conjugate
            j = np.where(np.abs(poles - np.conj(p)) < tol)[0]
            if len(j) == 0:
                raise RuntimeError("Unpaired complex pole")

            j = j[0]
            used[i] = used[j] = True

            if p.imag > 0:
                keep_poles.append(p)
                keep_res.append(2 * residues[:, i])
            else:
                keep_poles.append(poles[j])
                keep_res.append(2 * residues[:, j])

    return np.array(keep_poles), np.column_stack(keep_res)


def build_wmp_poles(poles_full, tol=1e-10, eps_rel=1e-2):
    poles_full = np.asarray(poles_full)
    mp = []
    for p in poles_full:
        if p.imag > tol:
            mp.append(p)
        elif abs(p.imag) <= tol:
            eps = eps_rel * max(1.0, abs(p))
            mp.append(p + 1j * eps)  # keep only the upper representative
        # drop imag < -tol
    # optional: deduplicate near-equal poles
    return np.array(mp)


def refit_residues_openmc(s, F, poles, weights=None):
    """
    Fit residues r so that:
        F(u) = E*sigma(E) ≈ Re( sum_j (-1j*r_j)/(u - p_j) )

    s: (n,) sqrt(E)
    F: (k,n) real, where F = E*sigma(E)
    poles: (m,)
    returns r: (k,m) complex, DIRECTLY usable by OpenMC (no /1j later)
    """
    s = np.asarray(s)
    poles = np.asarray(poles)
    F = np.asarray(F)

    if F.ndim == 1:
        F = F[None, :]

    # Cauchy matrix
    A = 1.0 / (s[:, None] - poles[None, :])  # (n,m) complex
    Ar = A.real
    Ai = A.imag

    # Design matrix:
    # Re( (-i r)/(u-p) ) = y*Ar + x*Ai
    M = np.hstack([Ai, Ar])  # (n, 2m)

    r_out = np.zeros((F.shape[0], poles.size), dtype=np.complex128)

    for c in range(F.shape[0]):
        b = -F[c]

        # Relative-error weighting (VF-style)
        eps = 1e-30
        w = 1.0 / np.maximum(np.abs(b), eps)

        Mw = M * w[:, None]
        bw = b * w

        xy, *_ = np.linalg.lstsq(Mw, bw, rcond=None)

        x = xy[: poles.size]
        y = xy[poles.size :]

        r_out[c] = x + 1j * y

    return r_out


def refit_residues_realpart(s, f, poles, weights=None):
    """
    Fit r (complex) so that Re(A r) ≈ f, where A_ij = 1/(s_i - p_j).
    s: (n,)
    f: (k,n) real (this should be E*sigma(E), in sqrt(E) space)
    poles: (m,)
    returns r_vf: (k,m) complex
    """
    s = np.asarray(s)
    poles = np.asarray(poles)
    f = np.asarray(f)
    if f.ndim == 1:
        f = f.reshape(1, -1)

    A = 1.0 / (s[:, None] - poles[None, :])  # (n,m) complex
    Ar = A.real
    Ai = A.imag
    M = np.hstack([Ar, -Ai])  # (n,2m)

    # if weights is not None:
    #     w = np.asarray(weights).reshape(-1, 1)  # (n,1)
    #     M_w = M * w
    # else:
    #     M_w = M

    r = np.zeros((f.shape[0], poles.size), dtype=np.complex128)
    for c in range(f.shape[0]):
        b = f[c]
        # if weights is not None:
        #     b = b * weights
        # VF-like inverse weighting to target relative error in f
        eps = 1e-30
        w = 1.0 / np.maximum(np.abs(b), eps)
        Mw = M * w[:, None]
        bw = b * w
        xy, *_ = np.linalg.lstsq(Mw, bw, rcond=None)  # (2m,)
        x = xy[: poles.size]
        y = xy[poles.size :]
        r[c] = x + 1j * y
    return r


def create_single_window_wmp(poles, residues, E_min, E_max, sqrtAWR, name="test"):
    """
    Create a WindowedMultipole object with a single window from AAA poles/residues.

    Parameters
    ----------
    poles : ndarray
        Complex poles from AAA (in sqrt_E space)
    residues : ndarray
        Residues from AAA, shape (n_channels, n_poles)
    E_min, E_max : float
        Energy bounds in eV
    sqrtAWR : float
        sqrt(atomic weight ratio)
    name : str
        Nuclide name

    Returns
    -------
    WindowedMultipole
        WMP object ready for _evaluate_aaa
    """
    n_poles = len(poles)
    n_channels = residues.shape[0]
    fissionable = n_channels == 3

    # Sort poles by real part
    sort_idx = np.argsort(poles.real)
    poles = poles[sort_idx]
    residues = residues[:, sort_idx]

    # Build data array: [pole, res_s, res_a, (res_f)]
    if fissionable:
        data = np.zeros((n_poles, 4), dtype=complex)
    else:
        data = np.zeros((n_poles, 3), dtype=complex)

    data[:, 0] = poles
    data[:, 1] = residues[0]
    data[:, 2] = residues[1]
    if fissionable:
        data[:, 3] = residues[2]

    # Create WMP object
    wmp = WindowedMultipole(name)
    wmp.data = data
    wmp.E_min = E_min
    wmp.E_max = E_max
    wmp.sqrtAWR = sqrtAWR
    wmp.spacing = np.sqrt(E_max) - np.sqrt(E_min)  # single window spans entire range
    wmp.windows = np.array([[1, n_poles]])  # 1-indexed, all poles in one window

    # No pseudopoles for simple case
    wmp.pseudo_poles = [np.array([])]
    wmp.pseudo_residues = [np.zeros((n_channels, 0))]

    # Dummy curvefit for compatibility
    wmp.curvefit = np.zeros((1, 1, n_channels))
    wmp.broaden_poly = np.array([False])

    return wmp


def fit_pseudopoles(Z, remainder, n_pseudo_poles, bcf, bestpra):
    # Place pseudo-poles geometrically outside the domain
    info = {"method": "pseudo_pole"}
    k = remainder.shape[0]

    Z_real = Z.real if np.iscomplexobj(Z) else Z
    Z_min, Z_max = np.min(Z_real), np.max(Z_real)
    Z_range = Z_max - Z_min

    # Strategy 1: Place pseudo-poles logarithmically outside domain
    # This gives better coverage for wide energy ranges
    if Z_min > 0:  # For positive energy grids
        # Place poles logarithmically spaced below and above
        n_left = n_pseudo_poles // 2
        n_right = (n_pseudo_poles + 1) // 2
        left_factors = np.logspace(4, 5, n_left)  # [10, 100, ...1000]
        pseudo_poles_left = Z_min / left_factors
        # Right poles: above Z_max
        right_factors = np.logspace(4, 5, n_right)  # [10, 100, ...1000]
        pseudo_poles_right = Z_max * right_factors

        pseudo_poles = np.concatenate([pseudo_poles_left, pseudo_poles_right])
    else:
        # For grids including zero, use linear spacing
        left_poles = Z_min - Z_range * np.linspace(0.5, 2.0, n_pseudo_poles // 2)
        right_poles = Z_max + Z_range * np.linspace(0.5, 2.0, (n_pseudo_poles + 1) // 2)
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
            lambda_reg = 1e-20 * max_remainder

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

    # DEBUG
    print(f"Z range: [{Z_min:.3e}, {Z_max:.3e}]")
    print(f"Pseudo-poles: {pseudo_poles}")

    print(f"Pseudo-poles: Added {len(pseudo_poles)} poles")
    print(f"  Remainder before: {np.max(np.abs(remainder)):.3e}")
    print(f"  Remainder after:  {np.max(np.abs(final_remainder)):.3e}")
    print(f"  Improvement:      {improvement:.3e}")

    info["pseudo_poles"] = pseudo_poles
    info["pseudo_residues"] = pseudo_residues
    info["n_pseudo_poles"] = n_pseudo_poles

    return info


def fit_pseudopoles_adaptive(Z, remainder, bcf, max_poles=6, rtol=1e-6, verbose=True):
    """
    Simple loop to find best pseudo-pole configuration.

    Try different numbers of poles and different distances.
    Pick the best one.
    """
    k = remainder.shape[0]
    Z_real = Z.real if np.iscomplexobj(Z) else Z
    Z_min, Z_max = np.min(Z_real), np.max(Z_real)
    Z_range = Z_max - Z_min

    # Define distance multipliers to try
    if Z_min > 0:
        # For positive domains, use multiplicative factors
        # distance_factors = [2, 5, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]
        distance_factors = [1e4, 1e5, 1e6, 1e7, 1e8]
    else:
        # For domains with negative values, use range multiples
        distance_factors = [0.5, 1, 2, 5, 10, 50, 100]

    best_error = np.inf
    best_config = None

    # Simple loop: try different pole counts and distances
    for n_poles in range(1, max_poles + 1):
        for dist_factor in distance_factors:
            # Place poles symmetrically
            poles = []
            if Z_min > 0:
                # Multiplicative placement
                n_left = n_poles // 2
                n_right = n_poles - n_left  # Handles odd numbers

                # Left poles
                for i in range(n_left):
                    pole = Z_min / dist_factor
                    if pole > 0:  # Don't go below zero
                        poles.append(pole)

                # Right poles
                for i in range(n_right):
                    poles.append(Z_max * dist_factor)
            else:
                # Additive placement
                n_left = n_poles // 2
                n_right = n_poles - n_left

                # Left poles
                for i in range(n_left):
                    poles.append(Z_min - Z_range * dist_factor)

                # Right poles
                for i in range(n_right):
                    poles.append(Z_max + Z_range * dist_factor)

            poles = np.array(poles)

            # Skip if we couldn't place all poles
            if len(poles) != n_poles:
                continue

            # Fit residues using least squares
            C = 1.0 / (Z[:, np.newaxis] - poles[np.newaxis, :])
            residues = np.zeros((n_poles, k), dtype=np.complex128)

            for i in range(k):
                if np.max(np.abs(remainder[i, :])) > 1e-12:
                    residues[:, i], _, _, _ = np.linalg.lstsq(
                        C, remainder[i, :], rcond=1e-12
                    )

            # Calculate approximation
            approx = residues.T @ C.T
            error_array = remainder - approx

            # Calculate max relative error
            max_rel_error = 0
            for i in range(k):
                nonzero = np.abs(bcf[i, :]) > 1e-15
                if np.any(nonzero):
                    rel_err = np.max(np.abs(error_array[i, nonzero] / bcf[i, nonzero]))
                    max_rel_error = max(max_rel_error, rel_err)

            # Check if this is better
            if max_rel_error < best_error:
                best_error = max_rel_error
                best_config = {
                    "n_poles": n_poles,
                    "poles": poles,
                    "residues": residues,
                    "dist_factor": dist_factor,
                    "error": max_rel_error,
                }

                if verbose:
                    print(
                        f"n={n_poles}, dist={dist_factor:6.1f}x, error={max_rel_error:.3e} ← best"
                    )

                # Stop if good enough
                if max_rel_error < rtol:
                    break
            elif verbose:  # Only print first few
                print(
                    f"n={n_poles}, dist={dist_factor:6.1f}x, error={max_rel_error:.3e}"
                )

        # Stop if we found good solution
        if best_error < rtol:
            break

    # Apply best configuration
    if best_config:
        # C_best = 1.0 / (Z[:, np.newaxis] - best_config['poles'][np.newaxis, :])
        # bestpra += best_config['residues'].T @ C_best.T

        if verbose:
            print(
                f"\nBest: {best_config['n_poles']} poles at {best_config['dist_factor']}x distance"
            )
            print(f"Poles: {best_config['poles']}")
            print(f"Final relative error: {best_config['error']:.3e}")

        # return {
        #     "method": "pseudo_pole",
        #     "pseudo_poles": best_config['poles'],
        #     "pseudo_residues": best_config['residues'],
        #     "n_pseudo_poles": best_config['n_poles'],
        #     "distance_factor": best_config['dist_factor']
        # }
        return best_config["poles"], best_config["residues"].T

    return {"method": "pseudo_pole", "error": "failed"}


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
    # while np.abs(np.sum(rv)) < deflation_tol and len(rv) > 0:
    #     count += 1

    #     # Remove first residue and pole
    #     rv = rv[1:]
    #     first_pv = pv[0]
    #     pv = pv[1:]

    #     if len(pv) == 0:
    #         break

    #     # Recalculate residues over new set of poles
    #     fr_pv = first_pv - pv
    #     rv = rv * fr_pv

    #     # Normalize
    #     norm_rv = np.linalg.norm(rv)
    #     if norm_rv > 0:
    #         rv = rv / norm_rv
    # print(f"Deflated {count} poles at tolerance {deflation_tol}")

    if len(pv) == 0:
        return np.array([])

    # Build and solve generalized eigenvalue problem
    m = len(pv)
    B = np.eye(m + 1, dtype=np.complex128)
    B[0, 0] = 0

    E = np.zeros((m + 1, m + 1), dtype=np.complex128)
    E[1:, 0] = 1
    E[1:, 1:] = np.diag(pv)
    E[0, 1:] = rv

    # Solve for poles
    poles = la.eigvals(E, B)

    # Remove poles at infinity
    poles = poles[~np.isinf(poles)]

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
