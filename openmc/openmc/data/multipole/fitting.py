import numpy as np
from scipy.linalg import svd
import scipy.linalg as la
from scipy.sparse import spdiags


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
    # J = np.arange(n, dtype=int)  # All indices initially
    Jz = []
    eps = 1e-13

    # Initial approximation: channel-wise mean
    R = np.mean(F, axis=1, keepdims=True) * np.ones((k, n))
    err_hist = []

    # FIND PEAKS:
    all_peak_energies = []
    from scipy.signal import find_peaks
    for i, channel in enumerate(channels):
        # Find peaks in this channel
        peaks, _ = find_peaks(np.real(channel))
        peak_energies = E[peaks]
        all_peak_energies.extend(peak_energies)
    resonances = np.unique(np.array(all_peak_energies))
    resonances_energies = np.sort(resonances)
    # print(resonances_energies)

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

        # (Greedily) select next support point from fit region
        force_physical = False
        if force_physical:
            candidate_indices = np.argsort(err[J_fit])[::-1]
            selected = False
            fallback_strategy = "skip"
            for idx in candidate_indices:
                j_star = J_fit[idx]
                candidate_energy = E[j_star]

                # Check if physical
                min_distance = np.min(np.abs(resonances_energies - candidate_energy))
                relative_distance = min_distance / candidate_energy
                # if log:
                #     print(f"  min distance: {min_distance:.3f}, relative distance: {relative_distance:.3f}")
                tolerance = 5e-3
                # Check if this candidate is near a physical resonance
                if relative_distance <= tolerance:
                    # Great! Use this point
                    selected = True
                    break
                elif fallback_strategy == "snap":
                    # Snap to nearest resonance and use that
                    current_energy = E[j_star]
                    nearest_resonance = resonances_energies[np.argmin(np.abs(resonances_energies - current_energy))]
                    # Find grid point closest to this resonance
                    j_star = np.argmin(np.abs(E - nearest_resonance))
                    selected = True
                    if log:
                        print(f"  current energy {current_energy:.3f}, nearest resonance {nearest_resonance:.3f}, switched to {j_star}")
                    break
                elif fallback_strategy == "skip":
                    # Try next candidate
                    continue
                # Could add other strategies here

            if not selected:
                if log:
                    print(f"No physical resonance found at iteration {m}, stopping")
                break
        else:
            candidate_errs = err[J_fit]
            if len(candidate_errs) == 0:  # no more points left in piece
                break
            jpos = np.argmax(candidate_errs)  # index of maximum error
            j_star = J_fit[jpos]  # maximum error point

        Jz.append(j_star)  # for lawson

        # Remove from candidate sets via mask
        J_fit = J_fit[J_fit != j_star]
        J_core = J_core[J_core != j_star]

        # Check if we still have points to fit
        if len(J_fit) == 0:
            print("Exiting MIAAA. Can't build Loewner matrix with no points.")
            # Don't add this support point since we can't compute weights
            break

        # Add to support
        z_list.append(grid[j_star])
        fz = np.hstack([fz, F[:, [j_star]]])
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

            # Additional relative error row weighting
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
        w_lawson, R = lawson_iteration(grid, F, z, fz, w, R, Jz, lawson_iter, space, log)
        if len(w_lawson) == 2 * len(z):
            if log > 2:
                print("Lawson successful. Removing zero weight supports")
            # Remove zero weight supports before using
            z, fz, w_lawson = remove_zero_weight_supports(z, fz, w_lawson)
            w = w_lawson  # Keep the full 2m weights for downstream use
        else:
            w = w_lawson  # Use as-is

    # Undo normalization
    if normalize:
        fz = fz * norms[:, None]
        R = R * norms[:, None]

    return w, z, fz, R, err_hist


def evaluate_miaaa(grid, w, z, fz, space="E", w_den=None, w_num=None):
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

    # Check for coincident points
    min_distance = np.min(np.abs(grid[:, np.newaxis] - z[np.newaxis, :]))
    if min_distance < 1e-14:
        # print(f"  Warning: Grid point coincides with support point (min dist: {min_distance:.3e})")
        # Add small perturbation to avoid singularity
        grid = grid + 1e-14

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
    if w_den is not None:
        D = C @ w_den
    else:
        D = C @ w

    # Evaluate each channel
    for i in range(k):
        # Numerator
        if w_num is not None:
            N = C @ (w_num * fz[i, :])
        else:
            N = C @ (w * fz[i, :])

        # Non-support points: use barycentric formula
        valid = non_support & (np.abs(D) > 1e-300)
        R[i, valid] = N[valid] / D[valid]

        # Support points: exact interpolation
        for idx, j in support_indices:
            R[i, idx] = fz[i, j]

    return R


def lawson_iteration(grid, F, z, fz, w_init, R, Jz, max_iter, space, log=True):
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
    J_non_support : ndarray
        Indices of non-support points
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
    m = len(z)
    k, M = F.shape
    gama = 1
    lw = np.ones(M)
    lw_norm = np.linalg.norm(lw)
    lw = lw / lw_norm
    lbcr = []
    best_w = None
    best_R = R
    max_error = np.max(np.abs(F-best_R))
    best_error = np.max(np.abs(F-best_R))

    # cauchy matrix
    # Build Cauchy matrix - handle coincident points properly
    C = np.zeros((M, m), dtype=np.complex128)
    for i in range(m):
        diff = grid - z[i]
        mask = np.abs(diff) > 1e-14
        C[mask, i] = 1.0 / diff[mask]
        # For coincident points, set to 0 (will be handled by interpolation condition)
        C[~mask, i] = 0.0

    sm = [] # Equivalent to a MATLAB cell array
    for i in range(k):
        # Select the i-th row of f. Python uses 0-based indexing.
        # f[i, :] is the direct equivalent of f(i,:)
        # No transpose is needed because scipy.sparse.diags takes a 1-D array
        diagonal_values = F[i, :]
        # Create the sparse diagonal matrix using scipy.sparse.diags
        # The arguments are: (diagonals, offsets, shape)
        # The '0' for the offset means it's the main diagonal
        matrix = spdiags(diagonal_values, 0, (M, M))
        sm.append(matrix)

    L_list = []
    # replacement for the interpolation condition r(ti)~f(ti)
    L_supp = np.hstack([np.identity(m), -np.identity(m)])
    for i in range(k):
        # Li=[sm{i}*C -C*diag(fz(i,:))];
        Li = np.hstack([sm[i].dot(C), -C.dot(np.diag(fz[i, :]))])
        # Li(Jz,:)=Lsupp;
        Li[Jz, :] = L_supp
        L_list.append(Li)

    # Stack all Li matrices vertically to create L
    L = np.vstack(L_list)

    lerr = []
    eps = 1e-13
    for l in range(max_iter):
        lws = np.tile(lw, k)
        d = spdiags(np.sqrt(lws), 0, (M*k, M*k))
        _, _, Vh = svd(d.dot(L), full_matrices=False)
        w = Vh[-1, :]
        w_den = w[:m]     # First m elements: denominator weights
        w_num = w[m:2*m]  # Should be m weights, not all remaining

        R_new = evaluate_miaaa(grid, w, z, fz, space, w_den=w_den, w_num=w_num)
        # lmaxerror = np.max(np.abs(F-R_new))
        errs = []
        for i in range(k):
            rel_err = np.abs(F[i] - R_new[i]) / np.maximum(np.abs(F[i]), eps)
            errs.append(rel_err)
        rel_errors = np.maximum.reduce(errs)  # Max across channels
        lmaxerror = np.max(rel_errors)
        lerr.append(lmaxerror)
        if lmaxerror < best_error:
            best_error = lmaxerror
            best_w = w.copy()
            best_R = R_new.copy()
            if log:
                print(f"  Lawson iter {l}: optimized from {max_error:.3e} to {lmaxerror:.3e}")
        else:
            if log:
                print(f"  Lawson iter {l}: error {lmaxerror:.3e} hasn't beat {best_error:.3e}")
        # Update the Lawson wieghts(extended to multiple functions with a summation)
        # testlw=lw.*((max(abs(f-lbcr),[],1)).^gama);
        # absollute:
        # error_per_point = np.max(np.abs(F - R_new), axis=0)  # Shape: (M,)
        # relative:
        rel_errors = []
        for i in range(k):
            rel_err = np.abs(F[i] - R_new[i]) / np.maximum(np.abs(F[i]), 1e-30)
            rel_errors.append(rel_err)
        error_per_point = np.maximum.reduce(rel_errors)  # Max relative error across channels

        testlw = lw * (error_per_point ** gama)
        # testlw(find(testlw==Inf))=max(testlw(testlw~=Inf));
        if np.any(np.isinf(testlw)):
            max_non_inf = np.max(testlw[np.isfinite(testlw)])
            testlw[np.isinf(testlw)] = max_non_inf
        # MATLAB: testlw(isnan(testlw))=mean(testlw(~isnan(testlw)));
        if np.any(np.isnan(testlw)):
            mean_non_nan = np.mean(testlw[~np.isnan(testlw)])
            testlw[np.isnan(testlw)] = mean_non_nan

        if np.any(np.isnan(testlw)) or np.any(np.isinf(testlw)):
            print('Lawson terminated, Weights could not be fixed')
            break

        # testlw(find(testlw==0))=mean(testlw(testlw~=0));   %avoid any 0's from perfect interpolation
        if np.any(testlw == 0):
            mean_non_zero = np.mean(testlw[testlw != 0])
            testlw[testlw == 0] = mean_non_zero
        testlw = testlw / np.linalg.norm(testlw)
        # if(norm(testlw-lw,inf)<1e-8) %This Tolerance should be tested
        inf_norm_diff = np.linalg.norm(testlw - lw, np.inf)
        if inf_norm_diff < 1e-20:  # This Tolerance should be tested
            print(f"Lawson converged at iteration {l}, inf_norm_diff: {inf_norm_diff:.2e}")
            break
        lw = testlw.copy()
    if best_w is not None:
        # If Lawson succeeded, we have 2m weights
        return best_w, best_R
    else:
        # If Lawson failed, return original
        return w_init, R


def remove_zero_weight_supports(z, fz, w):
    m = len(z)
    # MATLAB: io=find(w(1:m)==0);
    # Python: Find indices of zero-weights in the first half of w
    io = np.where(w[:m] == 0)[0]

    # MATLAB: io2=find(w(m+1:end)==0);
    # Python: Find indices of zero-weights in the second half of w (relative to the start of that half)
    io2 = np.where(w[m:] == 0)[0]

    # MATLAB: io=intersect(io,io2);
    # Python: Find the common indices. This identifies support points where both weights are zero.
    io_common = np.intersect1d(io, io2)
    if len(io_common) > 0:
        z = np.delete(z, io_common)
        fz = np.delete(fz, io_common, axis=1)
        indices_to_delete = np.concatenate([io_common, io_common + m])
        w = np.delete(w, indices_to_delete)
        print(f"zero weight supports removed at {indices_to_delete}")
    else:
        print('No zero-weight support points found.')
    # return either way
    return z, fz, w
