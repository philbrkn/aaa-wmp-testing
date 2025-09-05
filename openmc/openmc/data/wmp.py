from pathlib import Path

import os
import pickle
import numpy as np
from scipy.signal import find_peaks

from .data import K_BOLTZMANN
from .neutron import IncidentNeutron
from .resonance import ResonanceRange
from .aaa import (aaa_xs)
from .multipole.fitting import (miaaa_xs, evaluate_miaaa)
from .multipole.conversion import (proper_rational)
from .multipole.plotting import (plot_reconstruction, plot_aaa_results, plot_miaaa_convergence)
from .multipole.cleanup import spurious_cleanup

# Constants that determine which value to access
_MP_EA = 0  # Pole

# Residue indices
_MP_RS = 1  # Residue scattering
_MP_RA = 2  # Residue absorption
_MP_RF = 3  # Residue fission

# Polynomial fit indices
_FIT_S = 0  # Scattering
_FIT_A = 1  # Absorption
_FIT_F = 2  # Fission

# Upper temperature limit (K)
TEMPERATURE_LIMIT = 3000

# Logging control
DETAILED_LOGGING = 2


def fit_nuclide(
    endf_file,
    njoy_error=5e-4,
    vf_pieces=None,
    log=False,
    path_out=None,
    mp_filename=None,
    njoy_input=None,
    bounds=None,
    plot_each_slice=True,
    fit_mask_guard=0,
    cleanup=False,
    cleanup_tol=1e-6,
    **kwargs,
):
    r"""Generate multipole data for a nuclide from ENDF.

    Parameters
    ----------
    endf_file : str
        Path to ENDF evaluation
    njoy_error : float, optional
        Fractional error tolerance for processing point-wise data with NJOY
    vf_pieces : integer, optional
        Number of equal-in-momentum spaced energy pieces for data fitting
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to write out mutipole data file and vector fitting figures
    mp_filename : str, optional
        File name to write out multipole data
    **kwargs
        Keyword arguments passed to :func:`openmc.data.multipole._vectfit_xs`

    Returns
    -------
    mp_data
        Dictionary containing necessary multipole data of the nuclide

    """

    # ======================================================================
    # PREPARE POINT-WISE XS
    # make 0K ACE data using njoy
    if njoy_input is None:
        if log:
            print(f"Running NJOY to get 0K point-wise data (error={njoy_error})...")

        nuc_ce = IncidentNeutron.from_njoy(
            endf_file,
            temperatures=[0.0],
            error=njoy_error,
            broadr=False,
            heatr=False,
            purr=False,
        )
        # dump the NJOY input for later use
        base_dir = Path(
            path_out
        ).parent  # TODO: this assumes path_out is a subdirectory
        njoy_path_out = base_dir / "NJOY_pickles"
        njoy_path_out.mkdir(parents=True, exist_ok=True)
        with open(njoy_path_out / "U238_NJOY.pickle", "wb") as f:
            pickle.dump(nuc_ce, f)
    else:
        # pickle in
        nuc_ce = pickle.load(open(njoy_input, "rb"))

    if log:
        print("Parsing cross sections within resolved resonance range...")

    # Determine upper energy: the lower of RRR upper bound and first threshold
    endf_res = IncidentNeutron.from_endf(endf_file).resonances
    if (
        hasattr(endf_res, "resolved")
        and hasattr(endf_res.resolved, "energy_max")
        and type(endf_res.resolved) is not ResonanceRange
    ):
        E_max = endf_res.resolved.energy_max
    elif hasattr(endf_res, "unresolved") and hasattr(endf_res.unresolved, "energy_min"):
        E_max = endf_res.unresolved.energy_min
    else:
        E_max = nuc_ce.energy["0K"][-1]
    E_max_idx = np.searchsorted(nuc_ce.energy["0K"], E_max, side="right") - 1
    for mt in nuc_ce.reactions:
        if hasattr(nuc_ce.reactions[mt].xs["0K"], "_threshold_idx"):
            threshold_idx = nuc_ce.reactions[mt].xs["0K"]._threshold_idx
            if 0 < threshold_idx < E_max_idx:
                E_max_idx = threshold_idx

    # parse energy and cross sections
    energy = nuc_ce.energy["0K"][: E_max_idx + 1]
    if bounds:
        E_min = bounds["E_min"]
        E_max = bounds["E_max"]
    else:
        E_min, E_max = energy[0], energy[-1]

    n_points = energy.size
    total_xs = nuc_ce[1].xs["0K"](energy)
    elastic_xs = nuc_ce[2].xs["0K"](energy)

    try:
        absorption_xs = nuc_ce[27].xs["0K"](energy)
    except KeyError:
        absorption_xs = np.zeros_like(total_xs)

    fissionable = False
    try:
        fission_xs = nuc_ce[18].xs["0K"](energy)
        fissionable = True
    except KeyError:
        pass

    # make vectors
    if fissionable:
        ce_xs = np.vstack((elastic_xs, absorption_xs, fission_xs))
        mts = [2, 27, 18]
    else:
        ce_xs = np.vstack((elastic_xs, absorption_xs))
        mts = [2, 27]

    if log:
        print(f"  MTs: {mts}")
        i0 = np.searchsorted(energy, E_min, side="left")
        i1 = np.searchsorted(energy, E_max, side="right")
        bound_pts = max(0, i1 - i0)  # number of points in [E_min, E_max]
        print(f"  Energy range: {E_min:.3e} to {E_max:.3e} eV ({bound_pts} points)")
        peaks, _ = find_peaks(total_xs)
        print(f"  There are {peaks.size} peaks in this range.")

    # ======================================================================
    # PERFORM VECTOR FITTING
    if vf_pieces is None:
        # divide into pieces for complex nuclides
        peaks, _ = find_peaks(total_xs)
        n_peaks = peaks.size
        if n_peaks > 200 or n_points > 30000 or n_peaks * n_points > 100 * 10000:
            vf_pieces = max(5, n_peaks // 10, n_points // 2000)
        else:
            vf_pieces = 1

    piece_width = (E_max - E_min) / vf_pieces
    # print(f"Piece width {piece_width}")
    alpha = nuc_ce.atomic_weight_ratio / (K_BOLTZMANN * TEMPERATURE_LIMIT)
    space = kwargs.get("space", "E")

    poles, residues = [], []
    # VF piece by piece
    for i_piece in range(vf_pieces):
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{vf_pieces}...")
        # start E of this piece
        E_left = E_min + i_piece * piece_width
        E_right = min(E_max, E_left + piece_width)

        s_left = np.sqrt(alpha * max(E_left, 0.0))
        s_right = np.sqrt(alpha * max(E_right, 0.0))
        if i_piece == 0 or s_left < 4:
            e_start = E_min
        else:
            e_start = max(E_min, (s_left - 4) ** 2 / alpha)

        e_end = min(E_max, (s_right + 4) ** 2 / alpha)

        # --- 3) Slice indices on the *E* grid ---
        e_start_idx = max(0, np.searchsorted(energy, e_start, side="right") - 1)
        e_end_idx = min(n_points, np.searchsorted(energy, e_end, side="left") + 1)
        if e_end_idx <= e_start_idx + 1:
            # ensure at least two points; mildly expand if needed
            e_start_idx = max(0, e_start_idx - 1)
            e_end_idx = min(n_points, e_end_idx + 1)

        e_idx = range(e_start_idx, e_end_idx)

        if log:
            print(
                f"  Piece {i_piece + 1}: E={energy[e_start_idx]:.3e} to "
                f"{energy[e_end_idx - 1]:.3e} eV"
            )

        if fit_mask_guard > 0:
            g = 0.1 * (e_end - e_start)
            mask_fit = (energy >= e_start - g) & (energy <= e_end + g)
            mask_core = (energy >= e_start) & (energy <= e_end)
            E_piece = energy[mask_fit]
            sig_s_piece = elastic_xs[mask_fit]
            sig_a_piece = absorption_xs[mask_fit]
            sig_f_piece = fission_xs[mask_fit] if fissionable else None

            w, z, fsz, faz, *rest = aaa_xs(
                E_piece,
                sig_s_piece,
                sig_a_piece,
                sigma_f=sig_f_piece,
                rtol=kwargs.get("rtol", 1e-13),
                mmax=kwargs.get("mmax", 100),
                log=log,
                fit_mask=np.ones_like(E_piece, dtype=bool),
                core_mask=mask_core[mask_fit],
                space=space,
            )
        else:
            E_piece = energy[e_idx]
            sig_s_piece = ce_xs[0, e_idx]
            sig_a_piece = ce_xs[1, e_idx]
            sig_f_piece = ce_xs[2, e_idx] if fissionable else None

            w, z, fz, R, err_hist = miaaa_xs(
                E_piece,
                [sig_s_piece, sig_a_piece, sig_f_piece],
                method=kwargs.get("method", "full_svd"),
                rtol=kwargs.get("rtol", 1e-13),
                mmax=kwargs.get("mmax", 100),
                greedy_metric="relative",  # relative or absolute_sum
                log=log,
                space=space,
                normalize=True,
                lawson_iter=0,
            )

            if cleanup:
                pol, res, pra, pr_handles, _ = proper_rational(
                    z, w, w, fz, R, E_piece,
                    pole_extraction=None, max_poly_degree=0,
                )
                z, fz, w = spurious_cleanup(pol, res, z, fz, w, E_piece, R.T, cleanup_tol=cleanup_tol)
            if len(w) == 2 * len(z):  # YES LAWSON
                m = len(z)
                w_num = w[m:2*m]
                w_den = w[:m]
                poles_s, residues_list, pra, pr_handles, polycoeffs = proper_rational(
                    z, w_num, w_den, fz, R, E_piece,
                    # pole_extraction="polynomial", max_poly_degree=2,
                    pole_extraction="pseudo_pole", n_pseudo_poles=2,
                )
            else:  # NO LAWSON
                poles_s, residues_list, pra, pr_handles, polycoeffs = proper_rational(
                    z, w, w, fz, R, E_piece,
                    # pole_extraction="polynomial", max_poly_degree=2,
                    pole_extraction="pseudo_pole", n_pseudo_poles=4,
                )
            # print(polycoeffs)

        poles.append(poles_s)
        residues.append(residues_list)

        if plot_each_slice:
            if len(w) == 2 * len(z):  # YES LAWSON
                R_pieces = evaluate_miaaa(E_piece, w, z, fz, space=space, w_num=w_num, w_den=w_den)
            else:  # NO LAWSON
                R_pieces = evaluate_miaaa(E_piece, w, z, fz, space=space)
            plot_aaa_results(
                E_piece,
                sig_s_piece,
                sig_a_piece,
                R_pieces[0],
                R_pieces[1],
                sigma_f=sig_f_piece,
                R_f=R_pieces[2],
                path_out=path_out,
            )

    # print number of poles
    n_poles = sum([p.size for p in poles])
    if log:
        # for p in poles[0]:
        #     p = np.sqrt(p)
        #     print(f"pole:  real {p.real:.2e} | imag {p.imag:.2e}")
        print(f"Total number of poles: {n_poles}")

    if vf_pieces == 1:
        channels_data = {"elastic": sig_s_piece,
                         "absorption": sig_a_piece,
                         "fission": sig_f_piece
                         }
        # Main reconstruction plot with remainder in subplot
        poles = poles[0]
        residues = residues[0]
        plot_reconstruction(E_piece, channels_data, poles, residues, name="U238", path_out="./plots",
                            plot_type="loglog", show_error=True, error_type="relative",
                            poly_info=polycoeffs)
        plot_reconstruction(E_piece, channels_data, poles, residues, name="U238", path_out="./plots",
                            plot_type="loglog", show_error=True, error_type="absolute",
                            poly_info=polycoeffs)
        plot_miaaa_convergence(err_hist, rtol=None, path_out="./plots")


    # collect multipole data into a dictionary
    mp_data = {
        "name": nuc_ce.name,
        "AWR": nuc_ce.atomic_weight_ratio,
        "E_min": E_min,
        "E_max": E_max,
        "poles": poles,
        "residues": residues,
    }

    # dump multipole data to file
    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if not mp_filename:
            mp_filename = f"{nuc_ce.name}_mp.pickle"
        mp_filename = os.path.join(path_out, mp_filename)
        with open(mp_filename, "wb") as f:
            pickle.dump(mp_data, f)
        if log:
            print(f"Dumped multipole data to file: {mp_filename}")

    return mp_data
