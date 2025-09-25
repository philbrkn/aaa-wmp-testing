from pathlib import Path

from aaa_wmp.core.wmp import fit_nuclide

if __name__ == "__main__":

    def get_paths(name):
        base = Path(__file__).parent

        # Convert U238 -> U_238, O16 -> O_016 for ENDF
        if name[-3:].isdigit():  # 3 digits like U238
            endf_name = f"{name[:-3]}_{name[-3:]}"
        elif name[-2:].isdigit():  # 2 digits like O16
            endf_name = f"{name[:-2]}_{name[-2:]:0>3}"  # pad to 3 digits
        elif name[-1:].isdigit():  # 1 digit like H1
            endf_name = f"{name[:-1]}_{name[-1:]:0>3}"
        else:
            endf_name = name

        return (
            base / "data/input/ENDF/ENDF-VIII-data" / f"n-092_{endf_name}.endf",
            base / "data/input/NJOY_pickles" / f"{name}_NJOY.pickle",
            base / "data/output",
        )

    name = "U238"
    endf_file, njoy_input, path_out = get_paths(name)
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-040_Zr_091.endf"
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-008_O_016.endf"
    # endf_file = Path(__file__).parent / "ENDF-VIII-data" / "n-026_Fe_056.endf"
    # njoy_input = None
    fit_nuclide(
        endf_file,
        name,
        vf_pieces=1,
        mmax=600,
        rtol=5e-3,
        path_out=path_out,
        log=2,
        fitter="miaaa",
        njoy_input=njoy_input,
        njoy_error=5e-4,
        bounds={"E_min": 1, "E_max": 30},
        # bounds={"E_min": 17400, "E_max": 17475},
        # bounds={"E_min": 17400, "E_max": 17600},
        # bounds={"E_min": 17200, "E_max": 17275},
        # method='qr+svd',
        # cleanup=True,
        # cleanup_tol=1e-10,  # Only remove if residue < tol
        plot_each_slice=True,
        # pole_extraction="pseudo_pole",
        pole_extraction="polynomial",
        # pole_extraction=None,
        max_poly_degree=1,
        fit_mask_guard=0.0,
        space="E",
        output_format="mp_data",
    )

    # poles = mp_data["poles"]
    # residues = mp_data["residues"]
    # Convert to OpenMC format
    # data_dict = poles_residues_to_openmc_data(poles, residues, name="U238", AWR=238.0)
