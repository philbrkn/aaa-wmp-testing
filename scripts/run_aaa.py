# scripts/fit_nuclide_simple.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aaa_wmp.processing.nuclide_fitter import NuclideFitter

# Direct configuration dictionary
config_dict = {
    # "vf_pieces": 50,
    "vf_pieces": 1,
    "mmax": 700,
    "rtol": 1e-3,
    "njoy_error": 5e-4,
    # "rerun_on_residual": True,
    "bounds": {"E_min": 1965, "E_max": 1985},
    # "bounds": {"E_min": 100, "E_max": 150},
    # "bounds": {"E_min": 1, "E_max": 20000},
    # "plot_each_slice": True,
    # "pole_extraction": "polynomial",
    "max_poly_degree": 1,
    # "pole_extraction": "pseudo_pole",
    "fit_mask_guard": 0.0,
    "space": "sqrt_E",
    "output_format": "mp_data",
    "log": 2,
    "method": "full_svd",
    "normalize": True,
    "lawson_iter": 0,
    "cleanup": False,
    # "cleanup": True,
    "fit_E_sigma": True,
}


def get_paths(name, base_dir):
    """Helper to get file paths."""
    if name[-3:].isdigit():
        endf_name = f"n-092_{name[:-3]}_{name[-3:]}"
    else:
        endf_name = name

    return {
        "endf": base_dir / "data/input/ENDF/ENDF-VIII-data" / f"{endf_name}.endf",
        "njoy": base_dir / "data/input/NJOY_pickles" / f"{name}_NJOY.pickle",
        "output": base_dir / "data/output" / name,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    name = "U238"
    paths = get_paths(name, base_dir)

    # Create fitter (can pass config dict or use defaults)
    fitter = NuclideFitter(config_dict)

    # Run fitting
    results = fitter.fit_nuclide(
        endf_file=str(paths["endf"]),
        name=name,
        njoy_input=str(paths["njoy"]) if paths["njoy"].exists() else None,
        path_out=str(paths["output"]),
        **config_dict,
    )
