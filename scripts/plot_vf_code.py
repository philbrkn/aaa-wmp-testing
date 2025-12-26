from openmc.data.plot_vf import plot_like_trainer

# Example usage — adjust these:
pickle_path = (
    "/home/philip/Documents/aaa-wmp-testing/data/output/U238/mp_data/U238_mp-VF.pickle"
)
endf_file = "data/input/ENDF/ENDF-VIII-data/n-092_U_238"
# E_lo, E_hi = 0.0, 20000.0
E_lo, E_hi = 0.0, 65.0

plot_like_trainer(
    str(pickle_path), str(endf_file), E_lo, E_hi, out_dir="plots_vf_original"
)
# plot_like_trainer(str(pickle_path), str(endf_file), E_lo, E_hi, out_dir="plots_final_AAA")
