from pathlib import Path
import pickle
from openmc.data.miaaa import vectfit_nuclide

# endf_file = Path(
#     "/home/philip/Research/endf-data"
#     "/ENDF-B-VII.1-neutrons/neutrons"
#     "/n-008_O_017.endf"
# )

# endf_file = Path(
#     "/home/philip/Research/endf-data"
#     "/ENDF-B-VII.2-neutrons/neutrons"
#     "/n-026_Fe_056.endf"
# )
# U238
endf_file = Path(
    "/home/philip/Research/endf-data"
    "/ENDF-B-VIII.0_neutrons"
    "/n-092_U_238.endf"
)

path_out = Path(__file__).parent / "aaa_test"

mp_data = vectfit_nuclide(
    endf_file,
    vf_pieces=1,
    mmax=100,
    rtol=1e-7,
    path_out=path_out,
    log=2,
    njoy_input='/home/philip/Research/wmp_testing/aaa_test/U238_NJOY.pickle',
    lawson_iter=10,
    # njoy_error=5e-4,
)

# # print poles and residues:
# # mp data ['poles', 'residues']:
# print("Poles:", mp_data['poles'])
#