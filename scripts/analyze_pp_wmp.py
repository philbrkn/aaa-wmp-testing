import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa_wmp.core.conversion import (
    evaluate_openmc_T,
    to_wmp_form,
)
from aaa_wmp.io.njoy_interface import generate_temperature_references
from aaa_wmp.visualization.plotting import (
    plot_wmp_temperature_validation,
    plot_wmp_validation,
)

K_BOLTZMANN = 8.617333262e-5  # eV/K
TEMPERATURE_LIMIT = 3000  # K

# mp_file = "data/output/U238/mp_data/U238_mp.pickle"
mp_file = "data/output/U238/mp_data/U238_mp_100p_1e-3_EsigE3.pickle"
# temp = 0
temp = 600

with open(mp_file, "rb") as f:
    mp_data = pickle.load(f)

# Handle multi-piece format (concatenate if needed)
poles_list = mp_data["poles"]
residues_list = mp_data["residues"]

E_min, E_max = mp_data["E_min"], mp_data["E_max"]
vf_pieces = len(poles_list)

ref_data = generate_temperature_references(
    endf_file="data/input/ENDF/ENDF-VIII-data/n-092_U_238.endf",
    name="U238",
    temperatures=[294, 600, 900, 1200, 1500],
    cache_dir="data/input/NJOY_pickles",
    njoy_error=5e-4,
    log=1,
)
# Create common energy grid within bounds
ref_0K = ref_data[0]
energy_0K = ref_0K["energy"]
mask_0K = (energy_0K >= E_min) & (energy_0K <= E_max)
energy = energy_0K[mask_0K]  # Use 0K grid as common grid
channels = ["elastic", "absorption"]
if ref_0K["fissionable"]:
    channels.append("fission")
k = len(channels)


def interp_xs(energy_src, xs_src, energy_dst):
    return np.interp(energy_dst, energy_src, xs_src)


# Interpolate 0K reference onto common grid
original_0K = np.vstack(
    [
        interp_xs(ref_0K["energy"], ref_0K["elastic_xs"], energy),
        interp_xs(ref_0K["energy"], ref_0K["absorption_xs"], energy),
        interp_xs(ref_0K["energy"], ref_0K["fission_xs"], energy)
        if ref_0K["fissionable"]
        else np.zeros(len(energy)),
    ]
)
# Interpolate T reference onto common grid
if temp > 0:
    ref_T = ref_data[temp]
    original_T = np.vstack(
        [
            interp_xs(ref_T["energy"], ref_T["elastic_xs"], energy),
            interp_xs(ref_T["energy"], ref_T["absorption_xs"], energy),
            interp_xs(ref_T["energy"], ref_T["fission_xs"], energy)
            if ref_T["fissionable"]
            else np.zeros(len(energy)),
        ]
    )
else:
    original_T = original_0K.copy()

sqrt_awr = np.sqrt(mp_data["AWR"])
alpha = mp_data["AWR"] / (K_BOLTZMANN * TEMPERATURE_LIMIT)
piece_width = (np.sqrt(mp_data["E_max"]) - np.sqrt(mp_data["E_min"])) / vf_pieces

# Global reconstruction accumulator
xs_recon_0K_total = np.zeros((k, energy.size))
xs_recon_T_total = np.zeros((k, energy.size))
xs_poles_only_total = np.zeros((k, energy.size))
xs_weight = np.zeros(energy.size)
# each window is probably going to have a different set of pseudo poles, hence why we do this
window_bounds = []
total_wmp_poles = 0

Z = np.sqrt(energy)

for i_piece, poles in enumerate(poles_list):
    sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
    sqrt_E_right = min(np.sqrt(E_max), sqrt_E_left + piece_width)
    window_bounds.append((sqrt_E_left**2, sqrt_E_right**2))
    E_left = sqrt_E_left**2
    # E_right = sqrt_E_right**2
    # Doppler broadening extension
    if i_piece == 0 or np.sqrt(alpha) * sqrt_E_left < 4.0:
        e_start = E_left
    else:
        e_start = max(E_min, (np.sqrt(alpha) * sqrt_E_left - 4.0) ** 2 / alpha)
    e_end = min(E_max, (np.sqrt(alpha) * sqrt_E_right + 4.0) ** 2 / alpha)
    piece_mask = (energy >= e_start) & (energy <= e_end)
    if not np.any(piece_mask):
        continue

    energy_i = energy[piece_mask]
    residues = residues_list[i_piece]
    c0 = mp_data["poly_info_list"][i_piece]["c0"]  # Get c0 for this piece
    poly_coeffs = c0[:, np.newaxis]  # Shape (k, 1)
    mp_poles, mp_residues = to_wmp_form(poles, residues, tol=1e-9)
    total_wmp_poles += len(mp_poles)

    import openmc.data.vectfit as vf

    Z_i = Z[piece_mask]
    F = vf.evaluate(Z_i, mp_poles, mp_residues, poly_coefficients=poly_coeffs)
    xs_recon_0K = np.real(F) / energy_i[None, :]
    # Also compute poles-only (no c0) for plotting
    F_poles_only = vf.evaluate(Z_i, mp_poles, mp_residues, poly_coefficients=None)
    xs_poles_only = np.real(F_poles_only) / energy_i[None, :]

    # Reconstruction at any temperature + same background
    xs_recon_T = evaluate_openmc_T(
        energy_i,
        temp,
        mp_poles,
        mp_residues / 1j,
        sqrtAWR=sqrt_awr,
        poly_coeffs=poly_coeffs,
        broaden_poly=False,
    )

    # Accumulate both
    xs_recon_0K_total[:, piece_mask] += np.asarray(xs_recon_0K).real
    xs_recon_T_total[:, piece_mask] += np.asarray(xs_recon_T).real
    xs_poles_only_total[:, piece_mask] += np.asarray(xs_poles_only).real
    xs_weight[piece_mask] += 1.0

print(f"Total WMP poles: {total_wmp_poles}")

# Normalize by overlap weight
nonzero = xs_weight > 0
xs_recon_0K_total[:, nonzero] /= xs_weight[nonzero]
xs_recon_T_total[:, nonzero] /= xs_weight[nonzero]
xs_poles_only_total[:, nonzero] /= xs_weight[nonzero]

# Channel symbols for plotting
symbols = {"elastic": "σ_el", "absorption": "σ_abs", "fission": "σ_f"}

# Plot each channel
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

output_dir = "/home/philip/Documents/aaa-wmp-testing/data/output/U238/validation/"
for i, channel in enumerate(channels):
    if temp == 0:
        c0 = mp_data["poly_info_list"][0]["c0"][i]  # Get c0 for this channel
        stats = plot_wmp_validation(
            E=energy,
            original=original_0K[i],
            reconstructed=xs_recon_0K_total[i],
            poles_only=xs_poles_only_total[i],
            channel_name=channel,
            symbol=symbols[channel],
            name="U238",
            path_out=output_dir,
            rtol=1e-3,
            atol=1e-5,
            window_bounds=window_bounds,
        )
    else:
        stats = plot_wmp_temperature_validation(
            E=energy,
            ref_0K=original_0K[i],
            ref_T=original_T[i],
            recon_0K=xs_recon_0K_total[i],
            recon_T=xs_recon_T_total[i],
            temperature=temp,
            channel_name=channel,
            symbol=symbols[channel],
            name="U238",
            path_out=output_dir,
            rtol=1e-3,
            atol=1e-5,
            window_bounds=window_bounds,
        )

    # print(f"{channel}: {stats['satisfaction_pct']:.1f}% within tolerance, "
    #       f"max rel err = {stats['max_rel_err']*100:.3f}%")


def assess_reconstruction(xs_recon, xs_ref, rtol=1e-3, atol=1e-5):
    """
    Assess reconstruction quality using VF/WMP criteria.

    rtol: relative tolerance (default 1e-3 = 0.1%)
    atol: absolute tolerance (default 1e-5 barns)
    """
    abserr = np.abs(xs_recon - xs_ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        relerr = abserr / xs_ref
    # Check for NaN (shouldn't happen but guard against it)
    if np.any(np.isnan(abserr)):
        return {"maxre": np.inf, "ratio": 0.0, "ratio2": 0.0, "status": "FAILED - NaN"}
    # If all points satisfy absolute tolerance, perfect
    if np.all(abserr <= atol):
        return {"maxre": 0.0, "ratio": 1.0, "ratio2": 1.0, "status": "PERFECT"}
    # Max relative error (only where abs error > atol)
    # This ignores relative error at near-zero cross sections
    maxre = np.max(relerr[abserr > atol])
    # Fraction of points satisfying EITHER relative OR absolute tolerance
    ratio = np.sum((relerr < rtol) | (abserr < atol)) / relerr.size
    ratio2 = np.sum((relerr < 10 * rtol) | (abserr < atol)) / relerr.size
    # Status
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

# Choose which reconstruction and reference to assess based on temp
recon_to_assess = xs_recon_T_total if temp > 0 else xs_recon_0K_total
ref_to_assess = original_T if temp > 0 else original_0K

for i, channel in enumerate(channels):
    result = assess_reconstruction(
        recon_to_assess[i], ref_to_assess[i], rtol=1e-3, atol=1e-5
    )
    print(f"\n{channel.upper()}:")
    print(f"  Max relative error: {result['maxre'] * 100:.3f}% (where abs err > 1e-5)")
    print(f"  Points within either tol:  {result['ratio'] * 100:.1f}%")
    print(f"  Points within 10x rtol and atol:  {result['ratio2'] * 100:.1f}%")
    print(f"  Status: {result['status']}")
