"""
Enhanced proper_rational function with built-in remainder analysis.
"""

from math import pi, sqrt

import numpy as np
import scipy.linalg as la
from scipy.special import wofz

K_BOLTZMANN = 8.617333262145e-5  # eV/K


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
        n_poly = poly_coeffs.shape[1]
        if sqrtkT != 0.0 and broaden_poly:
            # Doppler-broaden the polynomial
            dopp = sqrtAWR / sqrtkT
            broadened_factors = _broaden_wmp_polynomials(E, dopp, n_poly)

            # Sum contributions: sig[channel] += sum over poly terms
            for q in range(n_poly):
                sig += poly_coeffs[:, q, None] * broadened_factors[q, :]
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


def _broaden_wmp_polynomials(E, dopp, n):
    """
    Evaluate Doppler-broadened windowed multipole curvefit.

    The curvefit is a polynomial of the form:
    a/E + b/sqrt(E) + c + d*sqrt(E) + ...

    Parameters
    ----------
    E : ndarray
        Energy to evaluate at (eV)
    dopp : float
        sqrt(atomic weight ratio / kT) in units of 1/sqrt(eV)
    n : int
        Number of components to the polynomial

    Returns
    -------
    ndarray of shape (n, len(E))
        The value of each Doppler-broadened curvefit polynomial term
    """
    from scipy.special import erf

    E = np.asarray(E)
    sqrtE = np.sqrt(E)
    beta = sqrtE * dopp
    half_inv_dopp2 = 0.5 / dopp**2
    quarter_inv_dopp4 = half_inv_dopp2**2

    # Vectorized erf and exp
    erf_beta = np.where(beta > 6.0, 1.0, erf(beta))
    exp_m_beta2 = np.where(beta > 6.0, 0.0, np.exp(-(beta**2)))

    factors = np.zeros((n, len(E)))

    # factors[0] corresponds to 1/E term
    if n >= 1:
        factors[0] = erf_beta / E

    # factors[1] corresponds to 1/sqrt(E) term
    if n >= 2:
        factors[1] = 1.0 / sqrtE

    # factors[2] corresponds to constant term
    if n >= 3:
        factors[2] = factors[0] * (half_inv_dopp2 + E) + exp_m_beta2 / (
            beta * np.sqrt(np.pi)
        )

    # Recursive broadening of higher order components
    for i in range(1, n - 2):
        if i != 1:
            factors[i + 2] = -factors[i - 2] * (
                i - 1.0
            ) * i * quarter_inv_dopp4 + factors[i] * (
                E + (1.0 + 2.0 * i) * half_inv_dopp2
            )
        else:
            factors[i + 2] = factors[i] * (E + (1.0 + 2.0 * i) * half_inv_dopp2)

    return factors


def _faddeeva(z):
    z = np.asarray(z, dtype=np.complex128)

    out = np.empty_like(z, dtype=np.complex128)

    mask = np.angle(z) > 0
    out[mask] = wofz(z[mask])
    out[~mask] = -np.conj(wofz(np.conj(z[~mask])))

    return out


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

    # DEB
    # Filter far poles
    # z_center = np.mean(z.real)
    # z_span = np.max(z.real) - np.min(z.real)
    # max_dist = 5.0 * max(z_span, 0.1)  # At least 0.5 units
    #
    # keep_mask = np.abs(physical_poles.real - z_center) < max_dist
    # n_removed = np.sum(~keep_mask)
    # if n_removed > 0:
    #     print(f"Filtered {n_removed} far poles")
    # physical_poles = physical_poles[keep_mask]
    # DEB

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

    physical_res = physical_res.T

    # Calculate remainder
    CC = 1.0 / (Z[:, np.newaxis] - physical_poles[np.newaxis, :])
    pra = physical_res @ CC.T
    remainder = bcf - pra
    remainder = remainder.real

    # DEBUG
    # Compare analytical c0 vs mean remainder
    # c0_analytical = np.array([np.sum(wden * fz[i, :]) / np.sum(wden) for i in range(k)])
    # remainder_mean = np.mean(remainder, axis=1)
    #
    # print("Channel | c0 (analytical) | remainder mean | remainder std")
    # for i in range(k):
    #     print(
    #         f"{i:7} | {c0_analytical[i].real:15.6e} | {remainder_mean[i]:14.6e} | {np.std(remainder[i]):13.6e}"
    #     )
    # DEBUGEND

    # Initialize output
    poles = physical_poles.copy()
    res = physical_res.copy()

    if pole_extraction == "polynomial" and max_poly_degree >= 0:
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
        # analytical:
        c0 = np.array([np.sum(wden * fz[i, :]) / np.sum(wden) for i in range(k)])
        # Store in info
        info["c0"] = c0.real
    elif pole_extraction == "pseudo_pole":
        info = {"method": "pseudo_pole", "poly_coeffs": None}
        p_poles, p_residues = fit_pseudopoles_adaptive_0K(
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

    return poles, res, remainder, info


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


def convert_to_wmp_format(mp_data, tol=1e-9, log=False):
    """
    Convert multipole data from AAA export format to WMP format.

    This function:
    1. Applies to_wmp_form to merge conjugate poles
    2. Converts residues to WMP convention (divide by 1j)
    3. Returns data compatible with WindowedMultipole.from_multipole()

    Parameters
    ----------
    mp_data : dict or str
        Either a multipole data dictionary or path to pickle file with keys:
        - 'poles': list of pole arrays (one per piece)
        - 'residues': list of residue arrays (one per piece, shape (k, m))
        - 'name': nuclide name
        - 'AWR': atomic weight ratio
        - 'E_min': minimum energy (eV)
        - 'E_max': maximum energy (eV)
        - (optional) other metadata
    tol : float, optional
        Tolerance for identifying conjugate poles (default: 1e-9)
    log : bool, optional
        Whether to print conversion statistics

    Returns
    -------
    dict
        Multipole data in WMP format, ready for WindowedMultipole.from_multipole()

    Examples
    --------
    >>> # From dictionary
    >>> wmp_data = convert_to_wmp_format(mp_data)
    >>> wmp = WindowedMultipole.from_multipole(wmp_data)

    >>> # From file
    >>> wmp_data = convert_to_wmp_format("U238_mp.pickle")
    >>> wmp = WindowedMultipole.from_multipole(wmp_data)

    >>> # Save converted format
    >>> wmp_data = convert_to_wmp_format(mp_data)
    >>> with open("U238_mp_wmp.pickle", "wb") as f:
    ...     pickle.dump(wmp_data, f)
    """
    import pickle

    # Load from file if path provided
    if isinstance(mp_data, str):
        with open(mp_data, "rb") as f:
            mp_data = pickle.load(f)

    # Validate input
    required_keys = ["poles", "residues", "name", "AWR", "E_min", "E_max"]
    for key in required_keys:
        if key not in mp_data:
            raise ValueError(f"mp_data missing required key: '{key}'")

    poles_list = mp_data["poles"]
    residues_list = mp_data["residues"]
    n_pieces = len(poles_list)

    if len(residues_list) != n_pieces:
        raise ValueError(
            f"Inconsistent data: {n_pieces} pole arrays but "
            f"{len(residues_list)} residue arrays"
        )

    wmp_poles = []
    wmp_residues = []
    pr_constant = []

    total_input_poles = 0
    total_output_poles = 0

    for i_piece in range(n_pieces):
        poles = poles_list[i_piece]
        residues = residues_list[i_piece]

        total_input_poles += len(poles)

        # Apply to_wmp_form to:
        # - Keep only positive imaginary parts of complex conjugate pairs
        # - Double the residues for complex pairs
        mp_poles, mp_residues = to_wmp_form(poles, residues, tol=tol)

        # Convert residues to WMP convention
        # (WMP expects residues divided by 1j relative to AAA convention)
        mp_residues = mp_residues / 1j

        wmp_poles.append(mp_poles)
        wmp_residues.append(mp_residues)

        # Extract c0 from poly_info_list if available
        if "poly_info_list" in mp_data and i_piece < len(mp_data["poly_info_list"]):
            c0 = mp_data["poly_info_list"][i_piece].get("c0", None)
            if c0 is not None:
                pr_constant.append(np.asarray(c0))
            else:
                # Default to zeros if c0 not found
                k = residues.shape[0]  # number of channels
                pr_constant.append(np.zeros(k))
                if log:
                    print(
                        f"  Warning: No c0 found for piece {i_piece + 1}, using zeros"
                    )
        else:
            # Default to zeros if poly_info_list not present
            k = residues.shape[0]
            pr_constant.append(np.zeros(k))
            if log and i_piece == 0:
                print("  Warning: No poly_info_list found, using zero constants")

        total_output_poles += len(mp_poles)

        if log:
            n_real = np.sum(np.abs(mp_poles.imag) < tol)
            n_complex = len(mp_poles) - n_real
            print(
                f"Piece {i_piece + 1}/{n_pieces}: "
                f"{len(poles)} poles → {len(mp_poles)} WMP poles "
                f"({n_real} real, {n_complex} complex)"
            )

    # Create output dictionary with WMP-formatted data
    wmp_data = {
        "name": mp_data["name"],
        "AWR": mp_data["AWR"],
        "E_min": mp_data["E_min"],
        "E_max": mp_data["E_max"],
        "poles": wmp_poles,
        "residues": wmp_residues,
        "pr_constant": pr_constant,
    }

    # Preserve any additional metadata
    optional_keys = [
        "poly_info_list",
        "remainder_list",
        "energy_indices_list",
        "bcf_list",
        "err_hist_list",
        "vf_pieces",
        "space",
    ]
    for key in optional_keys:
        if key in mp_data:
            wmp_data[key] = mp_data[key]

    if log:
        print("\nConversion summary:")
        print(f"  Total input poles: {total_input_poles}")
        print(f"  Total WMP poles: {total_output_poles}")
        print(f"  Compression ratio: {total_input_poles / total_output_poles:.2f}x")

    return wmp_data


def fit_pseudopoles_adaptive_0K(
    Z,
    remainder,
    xs_0K_recon,
    max_poles=6,
    rtol=1e-3,  # Changed default to match VF (0.1%)
    atol=1e-5,  # NEW: absolute tolerance (barns)
    verbose=True,
    target_satisfaction=0.99,  # NEW: aim for 99% of points acceptable
):
    """
    Adaptive pseudo-pole fit using VF/WMP-style hybrid error metric.

    A point is considered accurate if EITHER:
      - Absolute error < atol, OR
      - Relative error < rtol

    Parameters
    ----------
    Z : (n,) array
        Fit grid (sqrt(E) or E)
    remainder : (k,n) array
        Target remainder per channel
    xs_0K_recon : (k,n) array
        0K reconstruction (denominator for relative error)
    max_poles : int
        Maximum pseudo poles to try
    rtol : float
        Relative tolerance (default 1e-3 = 0.1%)
    atol : float
        Absolute tolerance (default 1e-5 barns)
    target_satisfaction : float
        Stop if this fraction of points meet tolerance
    """
    remainder = np.asarray(remainder)
    xs_0K_recon = np.asarray(xs_0K_recon)

    if remainder.ndim != 2:
        raise ValueError(f"remainder must be 2D. Got {remainder.shape}")
    if xs_0K_recon.shape != remainder.shape:
        raise ValueError("xs_0K_recon must match remainder shape")

    k, n = remainder.shape
    Z = np.asarray(Z)
    if Z.shape[0] != n:
        raise ValueError("Z must match remainder columns")

    Z_real = Z.real if np.iscomplexobj(Z) else Z
    Z_min, Z_max = np.min(Z_real), np.max(Z_real)
    Z_range = Z_max - Z_min

    # Distance factors for pole placement
    if Z_min > 0:
        distance_factors = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    else:
        distance_factors = [0.5, 1, 2, 5, 10, 50, 100]

    best_satisfaction = -np.inf
    best_config = None

    for n_poles in range(1, max_poles + 1):
        for dist_factor in distance_factors:
            # Place poles symmetrically
            poles = []
            n_left = n_poles // 2
            n_right = n_poles - n_left

            if Z_min > 0:
                for _ in range(n_left):
                    pole = Z_min / dist_factor
                    if pole > 0:
                        poles.append(pole)
                for _ in range(n_right):
                    poles.append(Z_max * dist_factor)
            else:
                for _ in range(n_left):
                    poles.append(Z_min - Z_range * dist_factor)
                for _ in range(n_right):
                    poles.append(Z_max + Z_range * dist_factor)

            poles = np.asarray(poles, dtype=np.complex128)
            if poles.size != n_poles:
                continue

            # Fit residues
            C = 1.0 / (Z[:, None] - poles[None, :])
            residues = np.zeros((n_poles, k), dtype=np.complex128)

            for i in range(k):
                if np.max(np.abs(remainder[i, :])) > 1e-12:
                    residues[:, i], _, _, _ = np.linalg.lstsq(
                        C, remainder[i, :], rcond=1e-12
                    )

            approx = residues.T @ C.T  # (k,n)
            error_array = remainder - approx

            # ============================================================
            # NEW: VF-style hybrid error assessment
            # ============================================================
            total_points = 0
            satisfied_points = 0
            max_rel_error = 0.0

            for i in range(k):
                # Absolute error
                abs_err = np.abs(error_array[i, :])

                # Relative error (normalized by full xs, not remainder)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel_err = abs_err / np.abs(xs_0K_recon[i, :])
                    rel_err[np.isnan(rel_err) | np.isinf(rel_err)] = 0

                # Point satisfies if EITHER abs < atol OR rel < rtol
                satisfied = (abs_err < atol) | (rel_err < rtol)

                satisfied_points += np.sum(satisfied)
                total_points += n

                # Max relative error (only where abs error matters)
                significant = abs_err > atol
                if np.any(significant):
                    max_rel = np.max(rel_err[significant])
                    max_rel_error = max(max_rel_error, max_rel)

            satisfaction_ratio = satisfied_points / total_points
            # ============================================================

            # Choose based on satisfaction ratio (like VF does)
            if satisfaction_ratio > best_satisfaction:
                best_satisfaction = satisfaction_ratio
                best_config = {
                    "n_poles": n_poles,
                    "poles": poles,
                    "residues": residues,
                    "dist_factor": dist_factor,
                    "satisfaction": satisfaction_ratio,
                    "max_rel_error": max_rel_error,
                }

                if verbose:
                    print(
                        f"n={n_poles}, dist={dist_factor:6.1f}x, "
                        f"satisfy={satisfaction_ratio * 100:.1f}%, "
                        f"max_rel={max_rel_error * 100:.2f}% ← best"
                    )

                # Early exit if target met
                if satisfaction_ratio >= target_satisfaction:
                    break
            elif verbose:
                print(
                    f"n={n_poles}, dist={dist_factor:6.1f}x, "
                    f"satisfy={satisfaction_ratio * 100:.1f}%, "
                    f"max_rel={max_rel_error * 100:.2f}%"
                )

        if best_satisfaction >= target_satisfaction:
            break

    if best_config is not None:
        if verbose:
            print(
                f"\n✓ Best: {best_config['n_poles']} poles "
                f"at {best_config['dist_factor']}x distance"
            )
            print(f"  Satisfaction: {best_config['satisfaction'] * 100:.1f}%")
            print(f"  Max rel error: {best_config['max_rel_error'] * 100:.2f}%")

        return best_config["poles"], best_config["residues"].T

    # Fallback: no poles
    return np.array([], dtype=np.complex128), np.zeros((k, 0), dtype=np.complex128)


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
