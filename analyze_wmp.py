import h5py
from pathlib import Path
from openmc.data.wmp import vectfit_nuclide
from openmc.data.aaa import plot_aaa_results


def open_wmp_h5(wmp_file):
    with h5py.File(wmp_file, "r") as f:
        # # List all options (groups and datasets) in the HDF5 file
        def list_all(name, obj):
            print(name, "(Group)" if isinstance(obj, h5py.Group) else "(Dataset)")

        f.visititems(list_all)

        # Find names
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset) and "windows" in name.lower():
                print("windows", name, obj.shape)

        f.visititems(walk)

        # expand data:
        def expand(name, obj):
            if isinstance(obj, h5py.Dataset) and "data" in name.lower():
                print("data", name, obj.shape)

        f.visititems(expand)

        # Print the number of inner windows
        windows = f["U238/windows"]
        n_windows = windows.shape[0]
        print("Number of inner windows:", n_windows)
        print(windows)
        poles = f["U238/data"]
        n_poles = poles.shape[0]
        print("Number of poles:", n_poles)


if __name__ == "__main__":

    endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
    njoy_input_path = Path(__file__).parent / "NJOY_pickles" / "U238_NJOY.pickle"
    njoy_input = njoy_input_path if njoy_input_path.exists() else None
    path_out = Path(__file__).parent / "aaa_analyze_constant"
    background_constants = {
        "elastic": 14.04452808,
        "absorption": -0.00053376,
        "fission":  4.2125586016855946e-05,
        }

    mp_data = vectfit_nuclide(
        endf_file,
        vf_pieces=1,
        mmax=1000,
        rtol=1e-3,
        path_out=path_out,
        log=2,
        fitter="miaaa",
        njoy_input=njoy_input,
        njoy_error=5e-4,
        # bounds={'E_min': 0, 'E_max': 30},
        # bounds={"E_min": 17465, "E_max": 17596},
        bounds={"E_min": 785, "E_max": 861},
        space="E",
        # method='qr+svd',
        cleanup=False,
        cleanup_tol=1e-6,  # Only remove if pole-zero distance < 1e-6
        plot_each_slice=True,
        fit_mask_guard=0.0,
        analyze_constant=True,
        # background_constants=background_constants
    )


    # poles = mp_data["poles"]
    # residues = mp_data["residues"]
    # Convert to OpenMC format
    # data_dict = poles_residues_to_openmc_data(poles, residues, name="U238", AWR=238.0)


    wmp_file = Path(__file__).parent / "windowing_h5_output" / "U238-vf-wmp-VIII.h5"
    # with h5py.File("/home/philip/Research/WMP_Library/092238.h5", "r") as f:
