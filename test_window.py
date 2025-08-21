from pathlib import Path
import pickle
from openmc.data.wmp import _windowing


njoy_input='/home/philip/Research/wmp_testing/aaa_test/U238_NJOY.pickle'
nuc_ce = pickle.load(open(njoy_input, "rb"))

energy = nuc_ce.energy["0K"]
mpd = {
    "name": nuc_ce.name,
    "AWR":  nuc_ce.atomic_weight_ratio,
    "energy": energy,
    "sigma_s": nuc_ce[2].xs["0K"](energy),
    "sigma_a": nuc_ce[27].xs["0K"](energy),
    "sigma_f": nuc_ce[18].xs["0K"](energy) if 18 in nuc_ce.reactions else None
}

wmp = _windowing(mpd, n_cf=0, log=2, aaa_tol=5e-2, mmax=500, n_win=64, pseudo_L=0)



# export
wmp.export_to_hdf5(path=Path(__file__).parent / "wmp_test.h5")