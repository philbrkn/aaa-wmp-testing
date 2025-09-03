import numpy as np
from scipy.linalg import svd
import scipy.linalg as la


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
