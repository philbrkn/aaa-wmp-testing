import numpy as np

from .conversion import przd_for_poles
from .fitting import evaluate_miaaa
from scipy.sparse import diags


def spurious_cleanup(
    pol,
    res,
    z,
    fz,
    w,
    Z,
    F,
    cleanup_tol=1e-6,
    max_error_ratio=2.0,
    space="E",
    log=False,
):
    """Spurious poles / froissart doublet cleanup with error monitoring"""

    # find negligible residues
    ii = np.where(np.max(np.abs(res) / np.max(np.abs(F), axis=0), axis=1) < cleanup_tol)[0]
    ni = len(ii)
    if ni == 0:
        print("  No small residues found, cleanup over.")
    else:
        print(f" {ni} froissart doublets found.")

    indices_to_remove = []

    # For each spurious pole find and remove closest support point:
    for j in range(ni):
        azp = np.abs(z - pol[ii[j]])
        jj = np.argmin(azp)  # index of closest support point
        indices_to_remove.append(jj)
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)
        # # Remove the deleted support point from Z and F
        # deleted_point = z[jj]  # Save the deleted point value
        # z = np.delete(z, jj)
        # fz = np.delete(fz, jj, axis=1)  # Note: axis=1 for removing column
        # # Remove corresponding points from Z and F
        # mask = (Z != deleted_point)
        # Z = Z[mask]
        # F = F[mask, :]  # Assuming F is 2D
    # Now remove from z and fz
    for jj in indices_to_remove:
        z = np.delete(z, jj)
        fz = np.delete(fz, jj, axis=1)

    # Find indices in Z that are NOT in z
    mask = np.ones(len(Z), dtype=bool)
    for zi in z:
        mask &= np.abs(Z - zi) > 1e-14  # Exclude support points
    Z = Z[mask]
    F = F[mask, :]

    m = len(z)
    k = fz.shape[0]  # Infer k from F dimensions
    delta = Z[:, None] - z[None, :]   # M x m
    L_blocks = []
    for i in range(k):
        Li = (F[:, i][:, None] - fz[i, :][None, :]) / delta
        L_blocks.append(Li)

    L = np.vstack(L_blocks)
    _, _, Vh = np.linalg.svd(L, full_matrices=False)
    V = Vh.T
    w = V[:, -1]   # column m, zero-based

    # pol = przd_for_poles(z, w, deflation_tol=1e-14)
    # return pol
    return z, fz, w

    # Get initial approximation error
    # if ff is not None and sigma_f is not None:
    #     fz = np.array([fs, fa, ff])
    #     target_channels = [sigma_s, sigma_a, sigma_f]
    # else:
    #     fz = np.array([fs, fa])
    #     target_channels = [sigma_s, sigma_a]
    initial_error = evaluate_approximation_error(z, fz, w, Z, target_channels, space)
    if log:
        print(f"Initial approximation error: {initial_error:.3e}")


    # Find candidates for removal
    pole_candidates = find_close_pole_zero_pairs(pol, z, cleanup_tol)
    # Convert pole indices to support point indices
    # This is approximate - find support points closest to the problematic poles
    support_candidates = []
    for pole_idx in pole_candidates:
        pole = pol[pole_idx]
        distances = np.abs(z - pole)
        closest_support_idx = np.argmin(distances)
        support_candidates.append(closest_support_idx)

    # Remove duplicates
    support_candidates = list(set(support_candidates))

    points_removed = 0
    current_z, current_fs, current_fa, current_ff, current_w = z, fs, fa, ff, w

    # Test each candidate removal
    # Remove from end to preserve indices
    for candidate in sorted(support_candidates, reverse=True):
        # if candidate >= len(current_z):  # Skip if already removed
        #     continue
        # Try removing this support point
        test_z, test_fs, test_fa, test_ff, test_w = remove_support_point(
            current_z, current_fs, current_fa, current_ff, current_w, candidate
        )

        # Check if error stays acceptable
        if ff is not None and sigma_f is not None:
            test_fz = np.array([test_fs, test_fa, test_ff])
        else:
            test_fz = np.array([test_fs, test_fa])

        test_error = evaluate_approximation_error(test_z, test_fz, test_w, Z, target_channels)

        error_ratio = test_error / initial_error if initial_error > 0 else float("inf")

        if error_ratio < max_error_ratio:
            # Safe to remove
            current_z, current_fs, current_fa, current_ff, current_w = (
                test_z,
                test_fs,
                test_fa,
                test_ff,
                test_w,
            )
            points_removed += 1
            if log:
                print(f"Removed support point {candidate}: error ratio {error_ratio:.2f}")
        else:
            if log:
                print(f"Keeping support point {candidate}: removal would increase error to {error_ratio:.2f}x")

    if log:
        # Check if error stays acceptable
        if ff is not None and sigma_f is not None:
            current_fz = np.array([current_fs, current_fa, current_ff])
        else:
            current_fz = np.array([current_fs, current_fa])

        final_error = evaluate_approximation_error(current_z, current_fz, current_w, Z, target_channels, space)
        print(f"Cleanup complete: removed {points_removed} points, final error: {final_error:.3e}")

    return current_z, current_fs, current_fa, current_ff, current_w


def remove_support_point(z, fs, fa, ff, w, support_index):
    """
    Remove a support point and return modified arrays.

    Parameters
    ----------
    z, fs, fa, ff, w : arrays
        Current support points and data
    support_index : int
        Index of support point to remove

    Returns
    -------
    Modified arrays with support point removed
    """
    mask = np.ones(len(z), dtype=bool)
    mask[support_index] = False

    new_z = z[mask]
    new_fs = fs[mask]
    new_fa = fa[mask]
    new_ff = ff[mask] if ff is not None else None
    new_w = w[mask]

    return new_z, new_fs, new_fa, new_ff, new_w


def find_close_pole_zero_pairs(poles, support_points, cleanup_tol):
    """
    Find pole-zero pairs that are closer than cleanup_tol.

    Returns list of pole indices that should be removed.
    """
    if len(poles) == 0 or len(support_points) == 0:
        return []
    
    candidates = []
    
    for i, pole in enumerate(poles):
        # Find closest support point to this pole
        distances = np.abs(support_points - pole)
        min_dist = np.min(distances)
        
        if min_dist < cleanup_tol:
            candidates.append(i)
    
    return candidates

def evaluate_approximation_error(
    z, fz, w, Z, target_channels, space="E", error_type="relative"
):
    """
    Evaluate approximation error using your existing evaluate_miaaa function.

    Parameters
    ----------
    z : ndarray
        Support points
    fz : ndarray
        Function values at supports (k x m)
    w : ndarray
        Barycentric weights
    Z : ndarray
        Evaluation grid (original energy space)
    target_channels : list
        Target data [sigma_s, sigma_a, sigma_f] (same format as miaaa_xs input)
    space : str
        "E" or "sqrt_E"
    error_type : str
        "relative" or "absolute"

    Returns
    -------
    float
        Maximum error across all channels and points
    """
    # Transform grid for evaluation
    if space == "sqrt_E":
        grid = np.sqrt(Z)
    else:
        grid = Z

    # Use your existing evaluation function
    R = evaluate_miaaa(grid, w, z, fz, space)

    # Compute errors
    F_target = np.array(target_channels)
    eps = 1e-13

    if error_type == "relative":
        errors = []
        for i in range(len(target_channels)):
            if (
                target_channels[i] is not None
            ):  # Handle case where fission might be None
                rel_err = np.abs(F_target[i] - R[i]) / np.maximum(
                    np.abs(F_target[i]), eps
                )
                errors.append(np.max(rel_err))
        return max(errors) if errors else 0.0
    else:  # absolute
        total_error = 0.0
        for i in range(len(target_channels)):
            if target_channels[i] is not None:
                total_error += np.max(np.abs(F_target[i] - R[i]))
        return total_error
