from pathlib import Path
from openmc.data.wmp import fit_nuclide


if __name__ == "__main__":
    name = "U238"
    endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-092_U_238.endf"
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-040_Zr_091.endf"
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-008_O_016.endf"
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-026_Fe_056.endf"
    # njoy_input = None
    njoy_input = Path(__file__).parent / "NJOY_pickles" / f"{name}_NJOY.pickle"
    path_out = Path(__file__).parent / "aaa_analyze_constant"

    fit_nuclide(
        endf_file,
        name,
        vf_pieces=2,
        mmax=600,
        rtol=5e-4,
        path_out=path_out,
        log=2,
        fitter="miaaa",
        njoy_input=njoy_input,
        njoy_error=5e-4,
        bounds={'E_min': 1, 'E_max': 60},
        # bounds={"E_min": 17400, "E_max": 17475},
        # bounds={"E_min": 17400, "E_max": 17600},
        # bounds={"E_min": 17200, "E_max": 17275},
        # method='qr+svd',
        # cleanup=True,
        # cleanup_tol=1e-12,  # Only remove if residue < tol
        plot_each_slice=True,
        # pole_extraction="pseudo_pole",
        # pole_extraction="polynomial",
        # pole_extraction=None,
        max_poly_degree=1,
        fit_mask_guard=0.0,
        space="E",
        output_format="wmp",
    )

    # poles = mp_data["poles"]
    # residues = mp_data["residues"]
    # Convert to OpenMC format
    # data_dict = poles_residues_to_openmc_data(poles, residues, name="U238", AWR=238.0)