from pathlib import Path
# import pickle
from openmc.data.aaa import vectfit_nuclide

endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
njoy_input = Path(__file__).parent / "NJOY_pickles" / "U238_NJOY.pickle"  # error 5e-4
path_out = Path(__file__).parent / "aaa_test"

mp_data = vectfit_nuclide(
    endf_file,
    vf_pieces=10,
    mmax=200,
    rtol=1e-3,
    path_out=path_out,
    log=2,
    njoy_input=njoy_input,
    # njoy_error=5e-4,
)
