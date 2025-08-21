import glob
import time
from pathlib import Path
import openmc.data as data
import pickle
from openmc.data.multipole import _windowing

# vectfit_file = Path("/home/philip/Research/wmp_testing/"
#                     "vf_figures/O16_mp_pseudo.pickle")

# vectfit_file = Path("/home/philip/Research/wmp_testing/"
#                     "vf_figures/Fe56_mp_pseudo.pickle")
vectfit_file = Path("/home/philip/Research/wmp_testing/"
                    "mp-data-VF-output/U238_mp_data_VIII_20250815_175017.pickle")

with open(vectfit_file, "rb") as f:
    obj = pickle.load(f)
  
mp_data = obj["mp_data"] if isinstance(obj, dict) and set(obj.keys()) == {"mp_data"} else obj


# analyze mp data structure:
print("Keys in mp_data:", mp_data.keys())

start_time = time.time()
wmp = data.WindowedMultipole.from_multipole(
  mp_data,
  log=1,
)

# w_pp  = _windowing(mp_data,
#                    n_cf=1,          # kill polynomial
#                    n_win=120,       # much wider windows
#                    n_pp_max=8,      # allow up to 3 conjugate pairs
#                    rtol=1e-3, atol=1e-5,
#                    log=2)
# wroks for ncf2, nwin 1020, nppmax 10.

print("Time taken to create WMP:", time.time() - start_time)

out_file = Path(__file__).parent / 'U238-vf-wmp-VIII.h5'
wmp.export_to_hdf5(str(out_file), mode='w')
print("Wrote:", out_file)
