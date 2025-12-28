import pickle
import sys
from pathlib import Path

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

# Load reference data (same as before)
ref_data = generate_temperature_references(
    endf_file="data/input/ENDF/ENDF-VIII-data/n-092_U_238.endf",
    name="U238",
    temperatures=[0],
    cache_dir="data/input/NJOY_pickles",
    njoy_error=5e-4,
    log=0,
)

ref_0K = ref_data[0]
energy_full = ref_0K["energy"]  # FULL grid

channels = ["elastic", "absorption"]
fissionable = ref_0K["fissionable"]
if fissionable:
    channels.append("fission")
k = len(channels)

# Build sigE on FULL grid (sigma * E for comparison with bcf)
sigE_njoy_full = np.vstack(
    [
        ref_0K["elastic_xs"] * energy_full,
        ref_0K["absorption_xs"] * energy_full,
        ref_0K["fission_xs"] * energy_full
        if fissionable
        else np.zeros(len(energy_full)),
    ]
)

sqrt_awr = np.sqrt(mp_data["AWR"])
alpha = mp_data["AWR"] / (K_BOLTZMANN * TEMPERATURE_LIMIT)
piece_width = (np.sqrt(E_max) - np.sqrt(E_min)) / vf_pieces
Z_full = np.sqrt(energy_full)

import openmc.data.vectfit as vf

print("=" * 60)
print("PIECE-BY-PIECE ANALYSIS")
print("=" * 60)

for i_piece, poles in enumerate(poles_list):
    residues = residues_list[i_piece]
    c0 = mp_data["poly_info_list"][i_piece]["c0"]
    bcf_i = mp_data["bcf_list"][i_piece]

    mp_poles, mp_residues = to_wmp_form(poles, residues, tol=1e-9)

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

    sigE_poles = np.real(
        vf.evaluate(Z_i, mp_poles, mp_residues, poly_coefficients=None)
    )
    sigE_recon_c0 = sigE_poles + c0[:, None]

    print(
        f"\nPiece {i_piece}: [{energy_i[0]:.2f}, {energy_i[-1]:.2f}] eV, {len(energy_i)} pts, {len(mp_poles)} poles"
    )

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

        # NEW: Fit polynomial to NJOY residual
        residual_njoy = njoy_ch - poles_ch

        # Try different polynomial degrees
        for n_poly in [1, 2, 3, 5]:
            # Polynomial basis in sqrt(E): 1, z, z^2, ...
            poly_basis = np.vstack([Z_i**j for j in range(n_poly)]).T

            # Fit
            coeffs, _, _, _ = np.linalg.lstsq(poly_basis, residual_njoy, rcond=None)

            # Reconstruct
            poly_fit = poly_basis @ coeffs
            recon_polyfit = poles_ch + poly_fit

            err_recon_polyfit_njoy = np.max(
                np.abs(recon_polyfit - njoy_ch) / (np.abs(njoy_ch) + eps)
            )

            if n_poly == 1:
                print(f"  {ch_name}:")
                print(f"    bcf vs NJOY:           {err_bcf_njoy:.2e}")
                print(f"    poles+c0 vs bcf:       {err_recon_c0_bcf:.2e}")
                print(f"    poles+c0 vs NJOY:      {err_recon_c0_njoy:.2e}")
                print(
                    f"    poles+poly1 vs NJOY:   {err_recon_polyfit_njoy:.2e}  (coeffs: {coeffs[0]:.4e})"
                )
            else:
                print(f"    poles+poly{n_poly} vs NJOY:   {err_recon_polyfit_njoy:.2e}")

