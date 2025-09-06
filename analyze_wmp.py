from pathlib import Path
from openmc.data.wmp import fit_nuclide


if __name__ == "__main__":
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
    name = "O16"
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-040_Zr_091.endf"
    endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-008_O_016.endf"
    # njoy_input = None
    njoy_input = Path(__file__).parent / "NJOY_pickles" / f"{name}_NJOY.pickle"
    path_out = Path(__file__).parent / "aaa_analyze_constant"

    mp_data = fit_nuclide(
        endf_file,
        name,
        vf_pieces=10,
        mmax=600,
        rtol=5e-4,
        path_out=path_out,
        log=2,
        fitter="miaaa",
        njoy_input=njoy_input,
        njoy_error=5e-4,
        # bounds={'E_min': 0, 'E_max': 30},
        # bounds={"E_min": 17400, "E_max": 17475},
        # bounds={"E_min": 17200, "E_max": 17275},
        # bounds={"E_min": 400000, "E_max": 2.00e6},
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
