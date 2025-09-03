"""
Enhanced proper_rational function with built-in remainder analysis.
"""

import numpy as np
import scipy.linalg as la


def proper_rational(
    z,
    wnum,
    wden,
    fz,
    evaluation_grid,
    remainder_method="pseudopoles",
    remainder_tolerance=1e-6,
    max_pseudopoles=10,
    polynomial_degree=2,
    analyze_remainder=True,
    pseudopole_factor=10.0,
    adaptive_max_iter=20,
):
    """
    Convert barycentric rational approximation to proper rational form with
    sophisticated remainder handling and analysis.

    Parameters
    ----------
    z : ndarray
        Support points from AAA (length m)
    wnum : ndarray
        Numerator weights (length m or (k, m) for multi-function)
    wden : ndarray
        Denominator weights (length m)
    fz : ndarray
        Function values at support points (m,) or (k, m)
    evaluation_grid : ndarray
        Points where to evaluate and analyze the approximation
    remainder_method : str
        Method for handling remainder:
        - "none": Extract poles/residues only, leave remainder
        - "pseudopoles": Add fixed number of pseudopoles
        - "adaptive": Adaptively add pseudopoles to tolerance
        - "polynomial": Fit polynomial to remainder
    remainder_tolerance : float
        Target tolerance for adaptive method
    max_pseudopoles : int
        Maximum number of pseudopoles to add
    polynomial_degree : int
        Degree of polynomial for polynomial method
    analyze_remainder : bool
        Whether to compute and return detailed remainder analysis
    pseudopole_factor : float
        How far outside domain to place pseudopoles (factor * range)
    adaptive_max_iter : int
        Maximum iterations for adaptive method

    Returns
    -------
    dict
        Results dictionary containing:
        - 'poles': Complex poles
        - 'residues': Residues (array or list of arrays for multi-function)
        - 'approximation': Evaluated approximation on evaluation_grid
        - 'remainder_analysis': Detailed remainder information (if analyze_remainder=True)
    """
    # Handle input dimensions
    if fz.ndim == 1:
        fz = fz.reshape(1, -1)
        single_function = True
        k = 1
    else:
        single_function = False
        k = fz.shape[0]

    # Handle wnum dimensions
    if wnum.ndim == 1:
        wnum = np.tile(wnum, (k, 1))
    elif wnum.shape[0] != k:
        wnum = np.tile(wnum.reshape(1, -1), (k, 1))

    # Extract physical poles using eigenvalue method
    physical_poles = _extract_physical_poles(z, wden)

    # Compute residues for physical poles
    physical_residues = _compute_residues(physical_poles, z, wnum, wden, fz)

    # Evaluate original barycentric approximation
    original_approx = _evaluate_barycentric(evaluation_grid, z, wden, wnum, fz)

    # Evaluate physical pole contribution only
    physical_contribution = _evaluate_pole_contribution(
        evaluation_grid, physical_poles, physical_residues
    )

    # Compute original remainder
    original_remainder = original_approx - physical_contribution

    # Initialize results
    final_poles = physical_poles.copy()
    final_residues = [res.copy() for res in physical_residues]
    final_approximation = physical_contribution.copy()

    remainder_analysis = {
        "original_remainder": original_remainder,
        "final_remainder": original_remainder.copy(),
        "method_used": remainder_method,
        "remainder_stats": _compute_remainder_stats(original_remainder),
    }

    # Handle remainder according to specified method
    if remainder_method == "none":
        # Do nothing - keep original remainder
        pass

    elif remainder_method == "pseudopoles":
        pseudopoles, pseudo_residues = _add_fixed_pseudopoles(
            evaluation_grid, original_remainder, max_pseudopoles, pseudopole_factor
        )
        final_poles = np.concatenate([final_poles, pseudopoles])
        for i in range(k):
            final_residues[i] = np.concatenate(
                [final_residues[i], pseudo_residues[:, i]]
            )

        # Recompute approximation
        final_approximation = _evaluate_pole_contribution(
            evaluation_grid, final_poles, final_residues
        )
        remainder_analysis["final_remainder"] = original_approx - final_approximation

    elif remainder_method == "adaptive":
        pseudopoles, pseudo_residues, convergence_info = _add_adaptive_pseudopoles(
            evaluation_grid,
            original_remainder,
            remainder_tolerance,
            max_pseudopoles,
            pseudopole_factor,
            adaptive_max_iter,
        )

        if len(pseudopoles) > 0:
            final_poles = np.concatenate([final_poles, pseudopoles])
            for i in range(k):
                final_residues[i] = np.concatenate(
                    [final_residues[i], pseudo_residues[:, i]]
                )

            final_approximation = _evaluate_pole_contribution(
                evaluation_grid, final_poles, final_residues
            )
            remainder_analysis["final_remainder"] = (
                original_approx - final_approximation
            )

        remainder_analysis["convergence_info"] = convergence_info

    elif remainder_method == "polynomial":
        poly_coeffs, poly_approximation = _fit_polynomial_remainder(
            evaluation_grid, original_remainder, polynomial_degree
        )

        # Add polynomial contribution
        final_approximation = physical_contribution + poly_approximation
        remainder_analysis["final_remainder"] = original_approx - final_approximation
        remainder_analysis["polynomial_coefficients"] = poly_coeffs

    else:
        raise ValueError(f"Unknown remainder_method: {remainder_method}")

    # Update final remainder statistics
    remainder_analysis["final_remainder_stats"] = _compute_remainder_stats(
        remainder_analysis["final_remainder"]
    )

    # Prepare output
    results = {
        "poles": final_poles,
        "residues": final_residues[0] if single_function else final_residues,
        "approximation": (
            final_approximation[0] if single_function else final_approximation
        ),
    }

    if analyze_remainder:
        results["remainder_analysis"] = remainder_analysis

    return results


def _extract_physical_poles(z, w, deflation_tol=1e-10):
    """Extract poles using generalized eigenvalue method with deflation."""
    m = len(z)
    w_working = w.copy()
    deflated_indices = []

    # Deflation loop to handle cases where sum of weights is small
    deflation_count = 0
    while abs(np.sum(w_working)) < deflation_tol and deflation_count < m - 1:
        available = [i for i in range(m) if i not in deflated_indices]
        if not available:
            break

        # Choose support point with largest weight magnitude for deflation
        j = available[np.argmax([np.abs(w_working[i]) for i in available])]

        # Deflate: multiply by (z - z_j)
        for i in range(m):
            if i != j:
                w_working[i] *= z[i] - z[j]
            else:
                w_working[i] = 0

        deflated_indices.append(j)
        deflation_count += 1

    # Build generalized eigenvalue problem
    C = np.zeros((m + 1, m + 1), dtype=complex)
    C[0, 1:] = w_working
    C[1:, 0] = 1.0
    C[1:, 1:] = np.diag(z)
    C[0, 0] = 0.0

    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0

    # Solve for poles
    eigenvals, _ = la.eig(C, B)
    poles = eigenvals[np.isfinite(eigenvals)]

    return poles


def _compute_residues(poles, z, wnum, wden, fz):
    """Compute residues for each function channel."""
    k = wnum.shape[0]
    residues = []

    for i in range(k):
        channel_residues = np.zeros(len(poles), dtype=complex)

        for j, pole in enumerate(poles):
            # Numerator at pole
            num = np.sum(wnum[i] * fz[i] / (pole - z))
            # Denominator derivative at pole
            denom_deriv = -np.sum(wden / (pole - z) ** 2)

            if abs(denom_deriv) > 1e-15:
                channel_residues[j] = num / denom_deriv

        residues.append(channel_residues)

    return residues


def _evaluate_barycentric(grid, z, wden, wnum, fz):
    """Evaluate original barycentric approximation."""
    k = fz.shape[0]
    n = len(grid)
    result = np.zeros((k, n), dtype=complex)

    # Cauchy matrix
    C = 1.0 / (grid[:, None] - z[None, :])
    D = C @ wden

    for i in range(k):
        N = C @ (wnum[i] * fz[i])
        valid = np.abs(D) > 1e-15
        result[i, valid] = N[valid] / D[valid]

        # Handle support points exactly
        for j, zj in enumerate(z):
            idx = np.argmin(np.abs(grid - zj))
            if np.abs(grid[idx] - zj) < 1e-12:
                result[i, idx] = fz[i, j]

    return result


def _evaluate_pole_contribution(grid, poles, residues):
    """Evaluate pole-residue contribution."""
    k = len(residues)
    result = np.zeros((k, len(grid)), dtype=complex)

    for i in range(k):
        for j, pole in enumerate(poles):
            result[i] += residues[i][j] / (grid - pole)

    return result


def _compute_remainder_stats(remainder):
    """Compute statistical properties of remainder."""
    remainder = np.asarray(remainder)
    return {
        "max_abs_error": np.max(np.abs(remainder)),
        "rms_error": np.sqrt(np.mean(np.abs(remainder) ** 2)),
        "mean": np.mean(remainder),
        "std": np.std(remainder),
        "median": np.median(remainder),
    }


def _add_fixed_pseudopoles(grid, remainder, n_pseudo, factor):
    """Add fixed number of pseudopoles to approximate remainder."""
    grid_min, grid_max = np.min(grid), np.max(grid)
    grid_range = grid_max - grid_min

    # Place pseudopoles symmetrically outside domain
    pseudo_locs = np.linspace(
        grid_min - factor * grid_range, grid_max + factor * grid_range, n_pseudo
    )

    k = remainder.shape[0] if remainder.ndim > 1 else 1
    if remainder.ndim == 1:
        remainder = remainder.reshape(1, -1)

    # Fit residues for each channel
    pseudo_residues = np.zeros((len(pseudo_locs), k), dtype=complex)

    for i in range(k):
        # Build Cauchy matrix for pseudopoles
        C_pseudo = 1.0 / (grid[:, None] - pseudo_locs[None, :])

        # Least squares fit
        try:
            pseudo_residues[:, i], *_ = np.linalg.lstsq(
                C_pseudo, remainder[i], rcond=None
            )
        except np.linalg.LinAlgError:
            # Fallback: distribute remainder equally
            pseudo_residues[:, i] = np.mean(remainder[i]) / len(pseudo_locs)

    return pseudo_locs, pseudo_residues


def _add_adaptive_pseudopoles(grid, remainder, tolerance, max_pseudo, factor, max_iter):
    """Adaptively add pseudopoles until tolerance is met."""
    if remainder.ndim == 1:
        remainder = remainder.reshape(1, -1)
    k = remainder.shape[0]

    current_remainder = remainder.copy()
    pseudo_locs = []
    pseudo_residues = np.zeros((0, k), dtype=complex)

    grid_min, grid_max = np.min(grid), np.max(grid)
    grid_range = grid_max - grid_min

    convergence_info = {
        "error_history": [],
        "pseudopole_locations": [],
        "n_iterations": 0,
        "converged": False,
    }

    for iteration in range(max_iter):
        # Check convergence
        max_error = np.max(np.abs(current_remainder))
        convergence_info["error_history"].append(max_error)

        if max_error < tolerance or len(pseudo_locs) >= max_pseudo:
            convergence_info["converged"] = max_error < tolerance
            break

        # Add new pseudopole at location of maximum error
        error_idx = np.unravel_index(
            np.argmax(np.abs(current_remainder)), current_remainder.shape
        )
        error_location = grid[error_idx[1]]  # Grid location of max error

        # Place pseudopole outside domain, biased toward error location
        if error_location < (grid_min + grid_max) / 2:
            new_pseudo = grid_min - factor * grid_range * (1 + 0.1 * iteration)
        else:
            new_pseudo = grid_max + factor * grid_range * (1 + 0.1 * iteration)

        pseudo_locs.append(new_pseudo)
        convergence_info["pseudopole_locations"].append(new_pseudo)

        # Fit residue for new pseudopole
        new_residues = np.zeros(k, dtype=complex)
        for i in range(k):
            # Simple fit: residue = remainder * (grid - pseudopole)
            weights = 1.0 / np.abs(grid - new_pseudo)
            weights /= np.sum(weights)
            new_residues[i] = np.sum(weights * current_remainder[i])

        # Update remainder by subtracting new pseudopole contribution
        for i in range(k):
            current_remainder[i] -= new_residues[i] / (grid - new_pseudo)

        # Store new residues
        if len(pseudo_residues) == 0:
            pseudo_residues = new_residues.reshape(1, -1)
        else:
            pseudo_residues = np.vstack([pseudo_residues, new_residues])

        convergence_info["n_iterations"] += 1

    pseudo_locs = np.array(pseudo_locs) if pseudo_locs else np.array([])

    return pseudo_locs, pseudo_residues, convergence_info


def _fit_polynomial_remainder(grid, remainder, degree):
    """Fit polynomial to remainder."""
    if remainder.ndim == 1:
        remainder = remainder.reshape(1, -1)
    k = remainder.shape[0]

    poly_coeffs = []
    poly_approx = np.zeros_like(remainder)

    for i in range(k):
        # Fit polynomial
        coeffs = np.polyfit(grid, remainder[i].real, degree)
        poly_coeffs.append(coeffs)

        # Evaluate polynomial
        poly_approx[i] = np.polyval(coeffs, grid)

    return poly_coeffs, poly_approx
