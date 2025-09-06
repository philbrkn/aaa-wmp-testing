from openmc.data.plot_vf import plot_like_trainer
import os
from pathlib import Path

# Example usage — adjust these:
pickle_path = Path(__file__).parent / "mp-data-VF-output" / "U238_mp_data_VIII_20250815_175017.pickle"
endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
# E_lo, E_hi = 0.0, 20000.0
E_lo, E_hi = 0.0, 65.0

plot_like_trainer(str(pickle_path), str(endf_file), E_lo, E_hi, out_dir="plots_vf_original")
# plot_like_trainer(str(pickle_path), str(endf_file), E_lo, E_hi, out_dir="plots_final_AAA")
