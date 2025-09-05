from pathlib import Path
from openmc.data.wmp import fit_nuclide


if __name__ == "__main__":
    endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
    njoy_input_path = Path(__file__).parent / "NJOY_pickles" / "U238_NJOY.pickle"
    njoy_input = njoy_input_path if njoy_input_path.exists() else None
    path_out = Path(__file__).parent / "aaa_analyze_constant"

    mp_data = fit_nuclide(
        endf_file,
        vf_pieces=1,
        mmax=200,
        rtol=5e-4,
        path_out=path_out,
        log=2,
        fitter="miaaa",
        njoy_input=njoy_input,
        njoy_error=5e-4,
        # bounds={'E_min': 0, 'E_max': 30},
        bounds={"E_min": 17400, "E_max": 17475},
        # bounds={"E_min": 333, "E_max": 400},
        space="E",
        # method='qr+svd',
        # cleanup=True,
        # cleanup_tol=1e-12,  # Only remove if residue < tol
        plot_each_slice=True,
        fit_mask_guard=0.0,
    )

    # poles = mp_data["poles"]
    # residues = mp_data["residues"]
    # Convert to OpenMC format
    # data_dict = poles_residues_to_openmc_data(poles, residues, name="U238", AWR=238.0)

    wmp_file = Path(__file__).parent / "windowing_h5_output" / "U238-vf-wmp-VIII.h5"
    # with h5py.File("/home/philip/Research/WMP_Library/092238.h5", "r") as f:
