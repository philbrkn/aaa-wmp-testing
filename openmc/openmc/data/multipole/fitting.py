



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
