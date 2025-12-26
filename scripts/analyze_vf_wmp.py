import pickle
import sys
from pathlib import Path

K_BOLTZMANN = 8.617333262e-5  # eV/K
TEMPERATURE_LIMIT = 3000  # K
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

from aaa_wmp.core.conversion import (
    evaluate_openmc_T,
)
from aaa_wmp.io.njoy_interface import generate_temperature_references

mp_file = "data/output/U238/mp_data/U238_mp-VF.pickle"
vf_pieces = 328
# temp = 1500
temp = 0

with open(mp_file, "rb") as f:
    obj = pickle.load(f)

mp_data = (
    obj["mp_data"] if isinstance(obj, dict) and set(obj.keys()) == {"mp_data"} else obj
)

# Handle multi-piece format (concatenate if needed)
poles_list = mp_data["poles"]
residues_list = mp_data["residues"]


ref_data = generate_temperature_references(
    endf_file="data/input/ENDF/ENDF-VIII-data/n-092_U_238.endf",
    name="U238",
    temperatures=[294, 600, 900, 1200, 1500],
    cache_dir="data/input/NJOY_pickles",
    njoy_error=5e-4,
    log=1,
)
ref = ref_data[temp]
energy = ref["energy"]
# Apply energy bounds if specified
E_min, E_max = mp_data["E_min"], mp_data["E_max"]
mask = (energy >= E_min) & (energy <= E_max)
energy = energy[mask]

# Reference cross sections
xs_ref = {
    "elastic": ref["elastic_xs"][mask]
    if len(ref["elastic_xs"]) > sum(mask)
    else ref["elastic_xs"],
    "absorption": ref["absorption_xs"][mask]
    if len(ref["absorption_xs"]) > sum(mask)
    else ref["absorption_xs"],
}
if ref["fissionable"]:
    xs_ref["fission"] = (
        ref["fission_xs"][mask]
        if len(ref["fission_xs"]) > sum(mask)
        else ref["fission_xs"]
    )

Z = np.sqrt(energy)
awr = np.sqrt(mp_data["AWR"])
alpha = mp_data["AWR"] / (K_BOLTZMANN * TEMPERATURE_LIMIT)
piece_width = (np.sqrt(mp_data["E_max"]) - np.sqrt(mp_data["E_min"])) / vf_pieces

channels = ["elastic", "absorption"]
if ref["fissionable"]:
    channels.append("fission")
k = len(channels)
# Global reconstruction accumulator
xs_recon_total = np.zeros((k, energy.size))
xs_weight = np.zeros(energy.size)  # for overlap-safe accumulation

# each window is probably going to have a different set of pseudo poles, hence why we do this
for i_piece, poles in enumerate(poles_list):
    sqrt_E_left = np.sqrt(E_min) + i_piece * piece_width
    sqrt_E_right = min(np.sqrt(E_max), sqrt_E_left + piece_width)
    E_left = sqrt_E_left**2
    E_right = sqrt_E_right**2
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
    mp_poles, mp_residues = poles, residues
    # SOME HWO GET ENERGY SLICE FOR THAT WINDOW.
    xs_poles_0K = evaluate_openmc_T(
        energy_i, 0.0, mp_poles, mp_residues / 1, sqrtAWR=awr, poly_coeffs=None
    )

    def interp_xs(energy_src, xs_src, energy_dst):
        return np.interp(energy_dst, energy_src, xs_src)

    energy_0K = ref_data[0]["energy"]
    energy_eval = energy_i
    xs_ref_0K_interp = np.vstack(
        [
            interp_xs(energy_0K, ref_data[0]["elastic_xs"], energy_eval),
            interp_xs(energy_0K, ref_data[0]["absorption_xs"], energy_eval),
            interp_xs(energy_0K, ref_data[0]["fission_xs"], energy_eval),
        ]
    )
    # Background remainder (pointwise)
    xs_poles_0K = np.asarray(xs_poles_0K)
    remainder = xs_ref_0K_interp - xs_poles_0K

    # Reconstruction at any temperature + same background
    xs_recon = evaluate_openmc_T(
        energy_i, temp, mp_poles, mp_residues / 1, sqrtAWR=awr, poly_coeffs=None
    )
    xs_recon_i = np.asarray(xs_recon)
    xs_recon_total[:, piece_mask] += xs_recon_i.real
    xs_weight[piece_mask] += 1.0

nonzero = xs_weight > 0
xs_recon_total[:, nonzero] /= xs_weight[nonzero]

original = np.vstack([xs_ref[c] for c in channels])
reconstructed = xs_recon_total

for i, name in enumerate(channels):
    mask = original[i] != 0
    err = np.full_like(original[i], np.nan)
    err[mask] = (
        np.abs((reconstructed[i, mask] - original[i, mask]) / original[i, mask]) * 100
    )
    if np.any(mask):
        print(
            f"{name:<12} | Max rel error = {np.nanmax(err):.2e}%  | RMS rel error = {np.sqrt(np.nanmean(err**2)):.2e}%"
        )
