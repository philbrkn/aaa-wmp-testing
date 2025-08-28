from pathlib import Path
# import pickle
from openmc.data.aaa import vectfit_nuclide
import time

endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
njoy_input_path = Path(__file__).parent / "NJOY_pickles" / "U238_NJOY.pickle"
njoy_input = njoy_input_path if njoy_input_path.exists() else None
path_out = Path(__file__).parent / "aaa_test"

time1 = time.perf_counter()
mp_data = vectfit_nuclide(
    endf_file,
    vf_pieces=1,
    mmax=1000,
    rtol=1e-3,
    # path_out=path_out,
    log=2,
    njoy_input=njoy_input,
    njoy_error=5e-4,
    # bounds={'E_min': 0, 'E_max': 200},
    bounds={'E_min': 0, 'E_max': 1000},
    space='E',
    # method='randomized_svd'
)

print(f"Time taken: {time.perf_counter()-time1:.3f} s")
