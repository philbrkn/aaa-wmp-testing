# conversion.py
"""
Enhanced proper_rational function with built-in remainder analysis.
"""

import numpy as np
import scipy.linalg as la


def proper_rational(z, wnum, wden, fz, bcf, Z,
                    pole_extraction="polynomial",
                    max_poly_degree=0,
                    n_pseudo_poles=2,
                    pseudo_pole_strategy="geometric",
                    pseudo_pole_scale=10.0):
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
        # Add pseudo-poles to approximate remainder
        Z_min, Z_max = np.min(Z.real), np.max(Z.real)
        Z_center = (Z_min + Z_max) / 2
        Z_range = Z_max - Z_min
        
        # Generate pseudo-pole locations based on strategy
        pseudo_locs = generate_pseudo_pole_locations(
            Z_min, Z_max, Z_range, Z_center, n_pseudo_poles,
            pseudo_pole_strategy, pseudo_pole_scale
        )
        
        # Option to optimize pseudo-pole locations
        if pseudo_pole_strategy == "optimize":
            print("optimizing pseudo pole location")
            pseudo_locs = optimize_pseudo_poles(
                Z, remainder, pseudo_locs, n_iter=100
            )
        
        # Fit pseudo-pole residues for each channel
        pseudo_residues = np.zeros((len(pseudo_locs), k), dtype=complex)
        for i in range(k):
            if np.max(np.abs(remainder[i, :])) > 1e-12:
                # Build matrix for pseudo-pole contribution
                C_pseudo = 1.0 / (Z[:, None] - pseudo_locs[None, :])
                
                # Use regularized least squares for stability
                # Add small regularization to improve conditioning
                lambda_reg = 1e-10
                A = C_pseudo.T @ C_pseudo + lambda_reg * np.eye(len(pseudo_locs))
                b = C_pseudo.T @ remainder[i, :]
                pseudo_residues[:, i] = np.linalg.solve(A, b)
                
                # Update best approximation
                bestpra[i, :] += C_pseudo @ pseudo_residues[:, i]
        
        # Append pseudo-poles to physical poles
        poles = np.concatenate([poles, pseudo_locs])
        res = np.vstack([res, pseudo_residues])
        
        info["pseudo_poles"] = pseudo_locs
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


def generate_pseudo_pole_locations(Z_min, Z_max, Z_range, Z_center, 
                                  n_pseudo_poles, strategy, scale):
    """
    Generate pseudo-pole locations based on different strategies.
    
    Parameters
    ----------
    Z_min, Z_max : float
        Domain boundaries.
    Z_range : float
        Domain range (Z_max - Z_min).
    Z_center : float
        Domain center.
    n_pseudo_poles : int
        Number of pseudo-poles to generate.
    strategy : str
        Placement strategy.
    scale : float
        Scale factor for distance from domain.
    
    Returns
    -------
    pseudo_locs : ndarray
        Array of pseudo-pole locations.
    """
    if strategy == "geometric":
        # Place poles geometrically outside domain
        if n_pseudo_poles == 1:
            # Single pole far away
            pseudo_locs = np.array([Z_center + scale * Z_range])
        elif n_pseudo_poles == 2:
            # Two poles symmetrically placed
            pseudo_locs = np.array([
                Z_min - scale * Z_range,
                Z_max + scale * Z_range
            ])
        else:
            # Multiple poles: split between above and below
            n_below = n_pseudo_poles // 2
            n_above = n_pseudo_poles - n_below
            
            # Geometric spacing
            ratios_below = np.geomspace(scale, scale * 3, n_below)
            ratios_above = np.geomspace(scale, scale * 3, n_above)
            
            locs_below = Z_min - ratios_below * Z_range
            locs_above = Z_max + ratios_above * Z_range
            
            pseudo_locs = np.concatenate([locs_below, locs_above])
    
    elif strategy == "exponential":
        # Exponentially spaced poles
        if n_pseudo_poles <= 2:
            return generate_pseudo_pole_locations(
                Z_min, Z_max, Z_range, Z_center, 
                n_pseudo_poles, "geometric", scale
            )
        
        # Place half below, half above
        n_below = n_pseudo_poles // 2
        n_above = n_pseudo_poles - n_below
        
        # Exponential spacing starting from scale*Z_range
        exp_factors = np.exp(np.linspace(0, 2, max(n_below, n_above)))
        
        locs_below = Z_min - scale * Z_range * exp_factors[:n_below]
        locs_above = Z_max + scale * Z_range * exp_factors[:n_above]
        
        pseudo_locs = np.concatenate([locs_below, locs_above])
    
    elif strategy == "chebyshev":
        # Based on scaled Chebyshev nodes outside domain
        # Map Chebyshev nodes from [-1, 1] to regions outside domain
        cheb_nodes = np.cos(np.pi * np.arange(n_pseudo_poles) / (n_pseudo_poles - 1))
        
        # Split nodes between regions below and above domain
        n_below = n_pseudo_poles // 2
        n_above = n_pseudo_poles - n_below
        
        # Map to regions outside
        below_region = [Z_min - scale * Z_range * 2, Z_min - scale * Z_range * 0.5]
        above_region = [Z_max + scale * Z_range * 0.5, Z_max + scale * Z_range * 2]
        
        locs_below = below_region[0] + (below_region[1] - below_region[0]) * (cheb_nodes[:n_below] + 1) / 2
        locs_above = above_region[0] + (above_region[1] - above_region[0]) * (cheb_nodes[:n_above] + 1) / 2
        
        pseudo_locs = np.concatenate([locs_below, locs_above])
    
    else:  # Default to geometric
        return generate_pseudo_pole_locations(
            Z_min, Z_max, Z_range, Z_center, 
            n_pseudo_poles, "geometric", scale
        )
    
    return pseudo_locs


def optimize_pseudo_poles(Z, remainder, initial_locs, n_iter=100):
    """
    Optimize pseudo-pole locations to better fit the remainder.
    
    Parameters
    ----------
    Z : ndarray
        Evaluation grid.
    remainder : ndarray
        Remainder to be fitted (shape (k, len(Z))).
    initial_locs : ndarray
        Initial pseudo-pole locations.
    n_iter : int
        Number of optimization iterations.
    
    Returns
    -------
    optimized_locs : ndarray
        Optimized pseudo-pole locations.
    """
    k = remainder.shape[0]
    current_locs = initial_locs.copy()
    
    def residual(locs, Z, remainder):
        """Compute residual for given pole locations."""
        C = 1.0 / (Z[:, None] - locs[None, :])
        total_residual = 0
        
        for i in range(k):
            # Least squares solution for this set of locations
            res, _, _, _ = np.linalg.lstsq(C, remainder[i, :], rcond=None)
            fit = C @ res
            total_residual += np.sum(np.abs(remainder[i, :] - fit)**2)
        
        return total_residual
    
    # Simple optimization: perturb locations and keep improvements
    best_locs = current_locs.copy()
    best_residual = residual(best_locs, Z, remainder)
    
    for _ in range(n_iter):
        # Random perturbation
        perturbation = np.random.randn(len(current_locs)) * np.abs(current_locs) * 0.1
        new_locs = current_locs + perturbation
        
        new_residual = residual(new_locs, Z, remainder)
        
        if new_residual < best_residual:
            best_locs = new_locs.copy()
            best_residual = new_residual
            current_locs = new_locs
    
    return best_locs


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
