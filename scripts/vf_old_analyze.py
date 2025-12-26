import pickle
import time

import numpy as np

# vectfit_file = Path("/home/philip/Research/wmp_testing/"
#                     "vf_figures/O16_mp_pseudo.pickle")

# vectfit_file = Path("/home/philip/Research/wmp_testing/"
#                     "vf_figures/Fe56_mp_pseudo.pickle")
# vectfit_file = Path("mp-data-VF-output/U238_mp_data_VIII_20250815_175017.pickle")
# vectfit_file = Path("aaa_analyze_constant/U238_mp.pickle")
# vectfit_file = Path("data/output/WMP_Lib_viii.0/U238/U238_mp-VF.pickle")
vectfit_file = "data/output/U238/mp_data/U238_mp-VF.pickle"

with open(vectfit_file, "rb") as f:
    obj = pickle.load(f)

mp_data = (
    obj["mp_data"] if isinstance(obj, dict) and set(obj.keys()) == {"mp_data"} else obj
)


# analyze mp data structure:
print("Keys in mp_data:", mp_data.keys())
print(f"E min {mp_data['E_min']}  E max {mp_data['E_max']}")
print(f" num residues {len(mp_data['residues'])}")
print(f"number of vf pieces {len(mp_data['poles'])} ")
total_poles = sum(len(pole_list) for pole_list in mp_data["poles"])
print(f"total number of poles: {total_poles}")
start_time = time.time()


def count_wmp_poles_in_range(poles, bounds):
    """
    Count how many poles are in a given energy range.

    Parameters
    ----------
    wmp_data : WindowedMultipole
        The WMP data object
    E_min : float
        Lower energy bound in eV
    E_max : float
        Upper energy bound in eV

    Returns
    -------
    int
        Number of poles in the energy range
    """
    # Convert pole energies from sqrt(E) to E
    # wmp_data.data[:, 0] contains poles in sqrt(E) format
    # print(len(poles), len(poles[0]))
    flattened_poles = [item for sublist in poles for item in sublist]
    pole_energies = np.array(flattened_poles)

    E_min = np.sqrt(bounds["E_min"])
    E_max = np.sqrt(bounds["E_max"])
    # Count poles in the range
    in_range = (pole_energies >= E_min) & (pole_energies <= E_max)
    poles_in_range = pole_energies[in_range]
    poles_in_range = poles_in_range[np.argsort(poles_in_range.real)]
    # for p in poles_in_range:
    #     print(f"  pole: real {p.real:.2f}  | imag {p.imag:.2e}")
    n_poles = np.sum(in_range)
    print("----- VF analyze -----")
    print(f"Energy range: {bounds['E_min']:.2e} to {bounds['E_max']:.2e} eV")
    print(f"Number of poles: {n_poles}")
    print(f"Total poles in MP Data: {len(pole_energies)}")

    return n_poles


# bounds = {'E_min': 0, 'E_max': 30}
# bounds={"E_min": 17200, "E_max": 17275}
bounds = {"E_min": 0, "E_max": 20000}
count_wmp_poles_in_range(mp_data["poles"], bounds)

print("test", len(mp_data["poles"]))
print("Time taken to create WMP:", time.time() - start_time)
