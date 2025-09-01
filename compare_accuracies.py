import openmc.data
import numpy as np
from openmc.data.wmp import vectfit_nuclide
from pathlib import Path


def compare_accuracy(energy_test, xs_reference, wmp_result, aaa_result):
    # Evaluate both methods on same grid
    xs_wmp = evaluate_wmp(energy_test, wmp_result)  # poles + polynomials
    xs_aaa = evaluate_aaa_poles(energy_test, aaa_result)  # pure poles

    # Compare errors
    wmp_error = np.abs(xs_wmp - xs_reference) / xs_reference
    aaa_error = np.abs(xs_aaa - xs_reference) / xs_reference

    return {
        "wmp_max_error": np.max(wmp_error),
        "aaa_max_error": np.max(aaa_error),
        "wmp_rms_error": np.sqrt(np.mean(wmp_error**2)),
        "aaa_rms_error": np.sqrt(np.mean(aaa_error**2)),
    }


def evaluate_wmp(energy, wmp_file_path):
    """Use OpenMC's WindowedMultipole class to evaluate"""

    wmp = openmc.data.WindowedMultipole.from_hdf5(wmp_file_path)

    # OpenMC evaluation at specific temperature
    temperature = 600  # K

    sigma_s = np.zeros_like(energy)
    sigma_a = np.zeros_like(energy)

    for i, E in enumerate(energy):
        if E >= wmp.E_min and E <= wmp.E_max:
            xs = wmp(E, temperature)
            sigma_s[i] = xs[0]  # scattering
            sigma_a[i] = xs[1]  # absorption
            # xs[2] would be fission if present

    return sigma_s, sigma_a


def evaluate_aaa_poles(energy, aaa_res):

    endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
    njoy_input_path = Path(__file__).parent / "NJOY_pickles" / "U238_NJOY.pickle"
    njoy_input = njoy_input_path if njoy_input_path.exists() else None
    path_out = Path(__file__).parent / "aaa_test"

    mp_data = vectfit_nuclide(
        endf_file,
        vf_pieces=2000,
        mmax=1000,
        rtol=1e-3,
        path_out=path_out,
        log=2,
        njoy_input=njoy_input,
        njoy_error=5e-4,
        # bounds={'E_min': 0, 'E_max': 200},
        # bounds={'E_min': 17465, 'E_max': 17596},
        space='E',
        # method='qr+svd',
        cleanup=False,
        cleanup_tol=1e-6,  # Only remove if pole-zero distance < 1e-6
        plot_each_slice=False,
        fit_mask_guard=0.0,
    )


