import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa_wmp.core.conversion import to_wmp_form
from aaa_wmp.io.njoy_interface import generate_temperature_references
from aaa_wmp.processing.piece_fitting import calculate_piece_boundaries

K_BOLTZMANN = 8.617333262e-5  # eV/K
TEMPERATURE_LIMIT = 3000  # K

mp_file = "data/output/U238/mp_data/U238_mp.pickle"
temp = 0

with open(mp_file, "rb") as f:
    mp_data = pickle.load(f)

poles_list = mp_data["poles"]
residues_list = mp_data["residues"]
E_min, E_max = mp_data["E_min"], mp_data["E_max"]
vf_pieces = len(poles_list)

# Load reference data
ref_data = generate_temperature_references(
    endf_file="data/input/ENDF/ENDF-VIII-data/n-092_U_238.endf",
    name="U238",
    temperatures=[0],
    cache_dir="data/input/NJOY_pickles",
    njoy_error=5e-4,
    log=0,
)

ref_0K = ref_data[0]
energy_full = ref_0K["energy"]

channels = ["elastic", "absorption"]
fissionable = ref_0K["fissionable"]
if fissionable:
    channels.append("fission")
k = len(channels)

# Build sigE and xs on FULL grid
sigE_njoy_full = np.vstack(
    [
        ref_0K["elastic_xs"] * energy_full,
        ref_0K["absorption_xs"] * energy_full,
        ref_0K["fission_xs"] * energy_full
        if fissionable
        else np.zeros(len(energy_full)),
    ]
)

xs_njoy_full = np.vstack(
    [
        ref_0K["elastic_xs"],
        ref_0K["absorption_xs"],
        ref_0K["fission_xs"] if fissionable else np.zeros(len(energy_full)),
    ]
)

sqrt_awr = np.sqrt(mp_data["AWR"])
alpha = mp_data["AWR"] / (K_BOLTZMANN * TEMPERATURE_LIMIT)
piece_width = (np.sqrt(E_max) - np.sqrt(E_min)) / vf_pieces
Z_full = np.sqrt(energy_full)

# Bounded grid for final output
mask_bounded = (energy_full >= E_min) & (energy_full <= E_max)
energy = energy_full[mask_bounded]
original_0K = xs_njoy_full[:, mask_bounded]
Z = np.sqrt(energy)

# Accumulators for reconstruction (on bounded grid)
xs_recon_c0_total = np.zeros((k, len(energy)))
xs_recon_poly_total = np.zeros((k, len(energy)))  # polyfit version
xs_weight = np.zeros(len(energy))

# Store polyfit coeffs per piece per channel for plotting
polyfit_coeffs_all = []

window_bounds = []
total_wmp_poles = 0

import openmc.data.vectfit as vf

print("=" * 60)
print("PIECE-BY-PIECE ANALYSIS")
print("=" * 60)

n_poly_global = 3  # Polynomial degree for comparison

for i_piece, poles in enumerate(poles_list):
    sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
    sqrt_E_right = min(np.sqrt(E_max), sqrt_E_left + piece_width)
    window_bounds.append((sqrt_E_left**2, sqrt_E_right**2))

    residues = residues_list[i_piece]
    c0 = mp_data["poly_info_list"][i_piece]["c0"]
    bcf_i = mp_data["bcf_list"][i_piece]

    mp_poles, mp_residues = to_wmp_form(poles, residues, tol=1e-9)
    total_wmp_poles += len(mp_poles)

    # Get exact same indices as fitting
    e_start_idx, e_end_idx = calculate_piece_boundaries(
        i_piece,
        piece_width,
        {"energy": energy_full, "E_min": E_min, "E_max": E_max},
        alpha,
        "sqrt_E",
    )

    energy_i = energy_full[e_start_idx:e_end_idx]
    Z_i = Z_full[e_start_idx:e_end_idx]
    sigE_njoy_i = sigE_njoy_full[:, e_start_idx:e_end_idx]

    # Evaluate poles only
    sigE_poles = np.real(
        vf.evaluate(Z_i, mp_poles, mp_residues, poly_coefficients=None)
    )
    sigE_recon_c0 = sigE_poles + c0[:, None]

    print(
        f"\nPiece {i_piece}: [{energy_i[0]:.2f}, {energy_i[-1]:.2f}] eV, {len(energy_i)} pts, {len(mp_poles)} poles"
    )

    # Store polyfit coeffs for this piece
    piece_polyfit_coeffs = []

    for ch_idx, ch_name in enumerate(channels):
        bcf_ch = np.real(bcf_i[ch_idx])
        njoy_ch = sigE_njoy_i[ch_idx]
        recon_c0_ch = sigE_recon_c0[ch_idx]
        poles_ch = sigE_poles[ch_idx]

        eps = 1e-30

        # Existing comparisons
        err_bcf_njoy = np.max(np.abs(bcf_ch - njoy_ch) / (np.abs(njoy_ch) + eps))
        err_recon_c0_bcf = np.max(np.abs(recon_c0_ch - bcf_ch) / (np.abs(bcf_ch) + eps))
        err_recon_c0_njoy = np.max(
            np.abs(recon_c0_ch - njoy_ch) / (np.abs(njoy_ch) + eps)
        )

        # Fit polynomial to NJOY residual
        residual_njoy = njoy_ch - poles_ch

        print(f"  {ch_name}:")
        print(f"    bcf vs NJOY:           {err_bcf_njoy:.2e}")
        print(f"    poles+c0 vs bcf:       {err_recon_c0_bcf:.2e}")
        print(f"    poles+c0 vs NJOY:      {err_recon_c0_njoy:.2e}")

        for n_poly in [1, 2, 3, 5]:
            poly_basis = np.vstack([Z_i**j for j in range(n_poly)]).T
            coeffs, _, _, _ = np.linalg.lstsq(poly_basis, residual_njoy, rcond=None)
            poly_fit = poly_basis @ coeffs
            recon_polyfit = poles_ch + poly_fit
            err_recon_polyfit_njoy = np.max(
                np.abs(recon_polyfit - njoy_ch) / (np.abs(njoy_ch) + eps)
            )

            if n_poly == 1:
                print(
                    f"    poles+poly1 vs NJOY:   {err_recon_polyfit_njoy:.2e}  (c0_fit={coeffs[0]:.4e}, c0_ana={c0[ch_idx]:.4e})"
                )
            else:
                print(f"    poles+poly{n_poly} vs NJOY:   {err_recon_polyfit_njoy:.2e}")

            # Store coeffs for global poly degree
            if n_poly == n_poly_global:
                piece_polyfit_coeffs.append(coeffs.copy())

    polyfit_coeffs_all.append(piece_polyfit_coeffs)

    # ==========================================================================
    # ACCUMULATE FOR FINAL PLOT (using Doppler extension mask on bounded grid)
    # ==========================================================================
    E_left = sqrt_E_left**2
    if i_piece == 0 or np.sqrt(alpha) * sqrt_E_left < 4.0:
        e_start = E_left
    else:
        e_start = max(E_min, (np.sqrt(alpha) * sqrt_E_left - 4.0) ** 2 / alpha)
    e_end = min(E_max, (np.sqrt(alpha) * sqrt_E_right + 4.0) ** 2 / alpha)

    piece_mask = (energy >= e_start) & (energy <= e_end)
    if not np.any(piece_mask):
        continue

    energy_mask = energy[piece_mask]
    Z_mask = np.sqrt(energy_mask)

    # Reconstruct with c0
    poly_coeffs_c0 = c0[:, np.newaxis]
    F_c0 = vf.evaluate(Z_mask, mp_poles, mp_residues, poly_coefficients=poly_coeffs_c0)
    xs_recon_c0 = np.real(F_c0) / energy_mask[None, :]

    # Reconstruct with polyfit
    F_poles = vf.evaluate(Z_mask, mp_poles, mp_residues, poly_coefficients=None)
    sigE_poles_mask = np.real(F_poles)

    xs_recon_poly = np.zeros((k, len(energy_mask)))
    for ch_idx in range(k):
        poly_basis_mask = np.vstack([Z_mask**j for j in range(n_poly_global)]).T
        poly_fit_mask = poly_basis_mask @ polyfit_coeffs_all[i_piece][ch_idx]
        xs_recon_poly[ch_idx] = (sigE_poles_mask[ch_idx] + poly_fit_mask) / energy_mask

    # Accumulate
    xs_recon_c0_total[:, piece_mask] += xs_recon_c0
    xs_recon_poly_total[:, piece_mask] += xs_recon_poly
    xs_weight[piece_mask] += 1.0

print(f"\nTotal WMP poles: {total_wmp_poles}")

# Normalize by overlap weight
nonzero = xs_weight > 0
xs_recon_c0_total[:, nonzero] /= xs_weight[nonzero]
xs_recon_poly_total[:, nonzero] /= xs_weight[nonzero]


# =============================================================================
# ASSESSMENT
# =============================================================================
def assess_reconstruction(xs_recon, xs_ref, rtol=1e-3, atol=1e-5):
    abserr = np.abs(xs_recon - xs_ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        relerr = abserr / xs_ref
    if np.any(np.isnan(abserr)):
        return {"maxre": np.inf, "ratio": 0.0, "ratio2": 0.0, "status": "FAILED - NaN"}
    if np.all(abserr <= atol):
        return {"maxre": 0.0, "ratio": 1.0, "ratio2": 1.0, "status": "PERFECT"}
    maxre = np.max(relerr[abserr > atol])
    ratio = np.sum((relerr < rtol) | (abserr < atol)) / relerr.size
    ratio2 = np.sum((relerr < 10 * rtol) | (abserr < atol)) / relerr.size
    if ratio >= 0.99:
        status = "EXCELLENT"
    elif ratio >= 0.95:
        status = "GOOD"
    elif ratio >= 0.90:
        status = "ACCEPTABLE"
    else:
        status = "POOR"
    return {"maxre": maxre, "ratio": ratio, "ratio2": ratio2, "status": status}


print("\n" + "=" * 60)
print(f"Assessment at {temp}K:")
print("=" * 60)

for i, channel in enumerate(channels):
    print(f"\n{channel.upper()}:")

    # c0 reconstruction
    result_c0 = assess_reconstruction(
        xs_recon_c0_total[i], original_0K[i], rtol=1e-3, atol=1e-5
    )
    print(
        f"  [c0] Max rel err: {result_c0['maxre'] * 100:.3f}%, within tol: {result_c0['ratio'] * 100:.1f}%, status: {result_c0['status']}"
    )

    # polyfit reconstruction
    result_poly = assess_reconstruction(
        xs_recon_poly_total[i], original_0K[i], rtol=1e-3, atol=1e-5
    )
    print(
        f"  [poly{n_poly_global}] Max rel err: {result_poly['maxre'] * 100:.3f}%, within tol: {result_poly['ratio'] * 100:.1f}%, status: {result_poly['status']}"
    )


# =============================================================================
# PLOTTING
# =============================================================================
def _get_contiguous_regions(mask):
    regions = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions


def plot_wmp_validation_compare(
    E,
    original,
    recon_c0,
    recon_poly,
    c0,
    poly_coeffs,
    channel_name,
    symbol,
    name,
    path_out=None,
    rtol=1e-3,
    atol=1e-5,
    window_bounds=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Errors for c0
    abs_err_c0 = np.abs(recon_c0 - original)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_c0 = abs_err_c0 / np.abs(original)
        rel_err_c0[~np.isfinite(rel_err_c0)] = 0
    satisfied_c0 = (rel_err_c0 < rtol) | (abs_err_c0 < atol)
    pct_c0 = 100 * np.sum(satisfied_c0) / len(satisfied_c0)

    # Errors for poly
    abs_err_poly = np.abs(recon_poly - original)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_poly = abs_err_poly / np.abs(original)
        rel_err_poly[~np.isfinite(rel_err_poly)] = 0
    satisfied_poly = (rel_err_poly < rtol) | (abs_err_poly < atol)
    pct_poly = 100 * np.sum(satisfied_poly) / len(satisfied_poly)

    # Top Left: Original vs Reconstructions
    ax = axes[0, 0]
    ax.loglog(E, original, "b-", label=f"Reference {symbol}", linewidth=1.5)
    ax.loglog(E, recon_c0, "r--", label="c0 recon", linewidth=1.0, alpha=0.8)
    ax.loglog(
        E,
        recon_poly,
        "g:",
        label=f"poly{len(poly_coeffs)} recon",
        linewidth=1.0,
        alpha=0.8,
    )
    if window_bounds:
        for i, (E_l, E_r) in enumerate(window_bounds):
            ax.axvline(E_l, color="black", linestyle=":", alpha=0.5, linewidth=1)
            if i == len(window_bounds) - 1:
                ax.axvline(E_r, color="black", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(f"{name} {channel_name}: Reference vs Reconstructions")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # Top Right: Remainder comparison
    ax = axes[0, 1]
    c0_contribution = c0 / E
    pole_only = recon_c0 - c0_contribution
    remainder = original - pole_only

    # Build poly contribution (need to sum across pieces - approximate with first piece)
    poly_contribution = np.zeros_like(E)
    Z_plot = np.sqrt(E)
    for j, coeff in enumerate(poly_coeffs):
        poly_contribution += coeff * Z_plot**j
    poly_contribution /= E  # Convert from sigE to sigma

    ax.semilogx(E, remainder, "b-", label="Remainder (ref - poles)", linewidth=1.0)
    ax.semilogx(E, c0_contribution, "r--", label=f"c0/E (c0={c0:.3e})", linewidth=1.5)
    ax.semilogx(E, poly_contribution, "g:", label="poly/E", linewidth=1.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.set_title(f"{channel_name}: Remainder vs Background")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # Bottom Left: Relative Error comparison
    ax = axes[1, 0]
    ax.semilogy(
        E,
        rel_err_c0 * 100,
        "r-",
        linewidth=0.8,
        label=f"c0 ({pct_c0:.1f}% ok)",
        alpha=0.7,
    )
    ax.semilogy(
        E,
        rel_err_poly * 100,
        "g-",
        linewidth=0.8,
        label=f"poly ({pct_poly:.1f}% ok)",
        alpha=0.7,
    )
    ax.axhline(
        rtol * 100,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=f"rtol={rtol * 100:.1f}%",
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"{channel_name}: Relative Error Comparison")
    ax.set_xscale("log")
    ax.set_ylim(1e-6, 1e2)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # Bottom Right: Absolute Error comparison
    ax = axes[1, 1]
    ax.semilogy(E, abs_err_c0, "r-", linewidth=0.8, label="c0", alpha=0.7)
    ax.semilogy(E, abs_err_poly, "g-", linewidth=0.8, label="poly", alpha=0.7)
    ax.axhline(atol, color="k", linestyle="--", linewidth=1.5, label=f"atol={atol:.0e}")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absolute error (barns)")
    ax.set_title(f"{channel_name}: Absolute Error Comparison")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"{name} {channel_name} WMP Validation (c0 vs polyfit)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if path_out:
        os.makedirs(path_out, exist_ok=True)
        filename = f"{name}_{channel_name}_wmp_validation_compare.png"
        filepath = os.path.join(path_out, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()
    else:
        plt.show()


print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

output_dir = "data/output/U238/validation/"
symbols = {"elastic": "σ_el", "absorption": "σ_abs", "fission": "σ_f"}

for i, channel in enumerate(channels):
    # Use first piece's polyfit coeffs for plotting (approximate)
    plot_wmp_validation_compare(
        E=energy,
        original=original_0K[i],
        recon_c0=xs_recon_c0_total[i],
        recon_poly=xs_recon_poly_total[i],
        c0=mp_data["poly_info_list"][0]["c0"][i],
        poly_coeffs=polyfit_coeffs_all[0][i],
        channel_name=channel,
        symbol=symbols[channel],
        name="U238",
        path_out=output_dir,
        rtol=1e-3,
        atol=1e-5,
        window_bounds=window_bounds,
    )

# Add this after the piece-by-piece analysis loop

print("\n" + "=" * 60)
print("C0 ANALYSIS ACROSS PIECES")
print("=" * 60)

# Collect c0 values and piece centers
piece_centers_sqrt = []
piece_centers_E = []
c0_per_piece = {ch: [] for ch in channels}

for i_piece in range(vf_pieces):
    sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
    sqrt_E_right = sqrt_E_left + piece_width
    sqrt_E_center = (sqrt_E_left + sqrt_E_right) / 2
    E_center = sqrt_E_center**2

    piece_centers_sqrt.append(sqrt_E_center)
    piece_centers_E.append(E_center)

    c0 = mp_data["poly_info_list"][i_piece]["c0"]
    for ch_idx, ch_name in enumerate(channels):
        c0_per_piece[ch_name].append(c0[ch_idx])

piece_centers_sqrt = np.array(piece_centers_sqrt)
piece_centers_E = np.array(piece_centers_E)

# Print c0 values
for ch_name in channels:
    c0_vals = c0_per_piece[ch_name]
    print(f"\n{ch_name}:")
    for i, (E_c, c0_val) in enumerate(zip(piece_centers_E, c0_vals)):
        print(f"  Piece {i}: E_center={E_c:.2f} eV, c0={c0_val:.6e}")

# Plot c0 vs energy (not c0/E)
fig, axes = plt.subplots(1, k, figsize=(5 * k, 4))
if k == 1:
    axes = [axes]

for ch_idx, ch_name in enumerate(channels):
    ax = axes[ch_idx]
    c0_vals = np.array(c0_per_piece[ch_name])

    # Plot c0 at piece centers
    ax.plot(
        piece_centers_E,
        c0_vals,
        "bo-",
        markersize=10,
        linewidth=2,
        label="c0 per piece",
    )

    # Mark piece boundaries
    for i_piece in range(vf_pieces):
        sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
        sqrt_E_right = sqrt_E_left + piece_width
        ax.axvline(sqrt_E_left**2, color="gray", linestyle="--", alpha=0.5)
        if i_piece == vf_pieces - 1:
            ax.axvline(sqrt_E_right**2, color="gray", linestyle="--", alpha=0.5)

    # Try fitting a polynomial to c0 vs sqrt(E)
    if vf_pieces >= 2:
        for n_poly in [1, 2, 3]:
            if n_poly <= vf_pieces:
                poly_basis = np.vstack([piece_centers_sqrt**j for j in range(n_poly)]).T
                coeffs, _, _, _ = np.linalg.lstsq(poly_basis, c0_vals, rcond=None)

                # Evaluate on fine grid
                E_fine = np.linspace(E_min, E_max, 200)
                sqrt_E_fine = np.sqrt(E_fine)
                poly_basis_fine = np.vstack([sqrt_E_fine**j for j in range(n_poly)]).T
                c0_poly = poly_basis_fine @ coeffs

                ax.plot(E_fine, c0_poly, "--", alpha=0.7, label=f"poly{n_poly} fit")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("c0 (in E*sigma space)")
    ax.set_title(f"{ch_name}: c0 across pieces")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
if output_dir:
    filepath = os.path.join(output_dir, "U238_c0_across_pieces.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {filepath}")
    plt.close()
# Enhanced c0 visualization showing both discrete c0 values and c0/E curves per piece
# Add this after the existing c0 analysis section

fig, axes = plt.subplots(2, k, figsize=(5 * k, 8))
if k == 1:
    axes = axes.reshape(-1, 1)

for ch_idx, ch_name in enumerate(channels):
    # Top plot: c0 values (in E*sigma space)
    ax_top = axes[0, ch_idx]
    c0_vals = np.array(c0_per_piece[ch_name])

    ax_top.plot(
        piece_centers_E,
        c0_vals,
        "bo-",
        markersize=10,
        linewidth=2,
        label="c0 per piece",
    )

    # Mark piece boundaries
    for i_piece in range(vf_pieces):
        sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
        sqrt_E_right = sqrt_E_left + piece_width
        ax_top.axvline(sqrt_E_left**2, color="gray", linestyle="--", alpha=0.5)
        if i_piece == vf_pieces - 1:
            ax_top.axvline(sqrt_E_right**2, color="gray", linestyle="--", alpha=0.5)

    ax_top.set_xscale("log")
    ax_top.set_xlabel("Energy (eV)")
    ax_top.set_ylabel("c0 (E*sigma space)")
    ax_top.set_title(f"{ch_name}: c0 values across pieces")
    ax_top.legend()
    ax_top.grid(True, which="both", alpha=0.3)

    # Bottom plot: c0/E curves showing the actual background contribution
    ax_bot = axes[1, ch_idx]

    # Plot c0/E as continuous curves within each piece
    for i_piece in range(vf_pieces):
        sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
        sqrt_E_right = sqrt_E_left + piece_width
        E_left = sqrt_E_left**2
        E_right = sqrt_E_right**2

        # Create fine energy grid for this piece
        E_piece = np.logspace(np.log10(E_left), np.log10(E_right), 100)
        c0_val = c0_per_piece[ch_name][i_piece]
        c0_over_E = c0_val / E_piece

        # Plot the c0/E curve for this piece
        ax_bot.semilogx(
            E_piece,
            c0_over_E,
            linewidth=2,
            alpha=0.7,
            label=f"Piece {i_piece}" if vf_pieces <= 5 else None,
        )

        # Mark piece boundaries
        ax_bot.axvline(E_left, color="gray", linestyle="--", alpha=0.5)
        if i_piece == vf_pieces - 1:
            ax_bot.axvline(E_right, color="gray", linestyle="--", alpha=0.5)

    # Also plot the discrete c0/E values at piece centers
    c0_sigma_vals = c0_vals / piece_centers_E
    ax_bot.semilogx(
        piece_centers_E,
        c0_sigma_vals,
        "ko",
        markersize=8,
        label="c0/E at centers",
        zorder=10,
    )

    ax_bot.set_xlabel("Energy (eV)")
    ax_bot.set_ylabel("c0/E (barns)")
    ax_bot.set_title(f"{ch_name}: c0/E background contribution")
    if vf_pieces <= 5:
        ax_bot.legend()
    ax_bot.grid(True, which="both", alpha=0.3)

plt.tight_layout()
if output_dir:
    filepath = os.path.join(output_dir, "U238_c0_detailed_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {filepath}")
    plt.close()
