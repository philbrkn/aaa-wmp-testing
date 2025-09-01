#!/usr/bin/env python3
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from openmc.data import IncidentNeutron
from openmc.data.vectfit import evaluate  # raw vectfit evaluator (f = sigma * E)

# --- constants to match your fitter ---
K_BOLTZMANN_eV_per_K = 8.617333262145e-5  # eV/K
TEMPERATURE_LIMIT = 3000  # K


# ---------- helpers ----------
def load_mp(pickle_path: str) -> dict:
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    return d["mp_data"] if isinstance(d, dict) and "mp_data" in d else d


def compute_piece_windows(mp: dict) -> List[Tuple[float, float, float]]:
    """
    Replicate vectfit_nuclide() piece ranges with ±4/sqrt(alpha) padding.
    Returns [(E_start, E_end, s_center), ...] for each piece.
    """
    E_min = float(mp["E_min"]); E_max = float(mp["E_max"])
    n_pieces = len(mp["poles"])
    s_min, s_max = np.sqrt(E_min), np.sqrt(E_max)
    width = (s_max - s_min)/n_pieces

    alpha = float(mp["AWR"]) / (K_BOLTZMANN_eV_per_K * TEMPERATURE_LIMIT)

    wins = []
    for i in range(n_pieces):
        e_bound_start = (s_min + width*(i - 0.5))**2
        if i == 0 or np.sqrt(alpha*e_bound_start) < 4.0:
            E_start = E_min
        else:
            E_start = max(E_min, (np.sqrt(alpha*e_bound_start) - 4.0)**2 / alpha)

        e_bound_end = (s_min + width*(i + 1.0))**2
        E_end = min(E_max, (np.sqrt(alpha*e_bound_end) + 4.0)**2 / alpha)

        s_center = s_min + width*(i + 0.5)
        wins.append((float(E_start), float(E_end), float(s_center)))
    return wins


def refined_grid(E: np.ndarray, factor: int = 10) -> np.ndarray:
    """Interpolate to a 10× refined grid exactly like _vectfit_xs."""
    ne = E.size
    ne_test = (ne - 1)*factor + 1
    E_test = np.interp(np.arange(ne_test), np.arange(ne_test, step=factor), E)
    E_test[0], E_test[-1] = E[0], E[-1]  # guard like in _vectfit_xs
    return E_test


def unmerge_mp_to_vectfit(poles_mp: np.ndarray, residues_mp: np.ndarray):
    """
    Invert the merge you do in _vectfit_xs:

        mp_poles = [real poles, first-of-each-conj-pair]
        mp_residues = concatenate([best_residues[:, real_idx],
                                   2*best_residues[:, conj_idx]], axis=1) / 1j

    So, for each channel k:
      - real poles: best_r = 1j * mp_r
      - complex pole p: create pair (p, p*) with residues:
            r1 = (1j/2) * mp_r
            r2 = conj(r1)
    Returns (best_poles, best_residues) suitable for evaluate(s, ...).
    """
    poles_mp = np.asarray(poles_mp)
    residues_mp = np.asarray(residues_mp)  # shape (K, M')
    K, Mprime = residues_mp.shape

    best_poles = []
    best_residues = []

    for j in range(Mprime):
        p = poles_mp[j]
        if np.isclose(p.imag, 0.0):  # real pole
            best_poles.append(p)
            # column vector of residues for this pole across channels
            rcol = (1j) * residues_mp[:, j]  # invert the "/ 1j"
            best_residues.append(rcol)
        else:  # this represents a complex conjugate pair compressed into one
            p1 = p
            p2 = np.conj(p)
            r1 = (1j/2.0) * residues_mp[:, j]  # invert "*2" and "/1j"
            r2 = np.conj(r1)
            best_poles.extend([p1, p2])
            best_residues.extend([r1, r2])

    best_poles = np.array(best_poles, dtype=np.complex128)
    best_residues = np.stack(best_residues, axis=1).astype(np.complex128)  # shape (K, M)
    return best_poles, best_residues


# ---------- main plotting ----------
def plot_like_trainer(
    pickle_path: str,
    endf_file: str,
    E_lo: float,
    E_hi: float,
    njoy_error: float = 5e-4,
    out_dir: Optional[str] = None,
    log: bool = True,
):
    mp = load_mp(pickle_path)

    # clip to mp domain
    E_lo = max(E_lo, float(mp["E_min"]))
    E_hi = min(E_hi, float(mp["E_max"]))
    if E_lo >= E_hi:
        raise ValueError(f"Requested [{E_lo}, {E_hi}] eV outside mp range [{mp['E_min']}, {mp['E_max']}] eV.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # CE truth on coarse native grid within window

    # pickle dump:
    # if pickle exists dont do it
    # if out_dir and not os.path.exists(os.path.join(out_dir, "njoy_endfviii.pickle")):
    #     nuc_ce = IncidentNeutron.from_njoy(
    #         endf_file, temperatures=[0.0], error=njoy_error, broadr=False, heatr=False, purr=False
    #     )
    #     with open(os.path.join(out_dir, "njoy_endfviii.pickle"), "wb") as f:
    #         pickle.dump(nuc_ce, f)
    # else:
    with open("NJOY_pickles/U238_NJOY.pickle", "rb") as f:
        nuc_ce = pickle.load(f)
    
    E_full = nuc_ce.energy["0K"]
    mwin = (E_full >= E_lo) & (E_full <= E_hi)
    if not np.any(mwin):
        raise RuntimeError("Window has no CE grid points.")
    E_coarse = E_full[mwin]

    # Channels on coarse grid
    CE_s = nuc_ce[2].xs["0K"](E_coarse)
    try:
        CE_a = nuc_ce[27].xs["0K"](E_coarse)
    except KeyError:
        CE_a = np.zeros_like(CE_s)
    CE_f = None
    try:
        CE_f = nuc_ce[18].xs["0K"](E_coarse)
    except KeyError:
        pass

    # Build *test* grid & CE reference exactly like _vectfit_xs (interpolation)
    E_test = refined_grid(E_coarse, factor=10)
    xs_ref_s = np.interp(E_test, E_coarse, CE_s)
    xs_ref_a = np.interp(E_test, E_coarse, CE_a)
    xs_ref_f = np.interp(E_test, E_coarse, CE_f) if CE_f is not None else None

    # Reconstruct VF on E_test:
    #  - route each E to one piece whose padded [Estart,Eend] contains it
    #  - per piece, UNMERGE to raw vectfit form, then evaluate with evaluate(s,...)
    wins = compute_piece_windows(mp)
    K = np.asarray(mp["residues"][0]).shape[0]
    have_fis = (K == 3)

    R_s = np.full_like(E_test, np.nan, dtype=float)
    R_a = np.full_like(E_test, np.nan, dtype=float)
    R_f = np.full_like(E_test, np.nan, dtype=float) if have_fis else None

    # Precompute unmerged (best) poles/residues per piece
    best_per_piece = []
    for i in range(len(mp["poles"])):
        bp, br = unmerge_mp_to_vectfit(mp["poles"][i], mp["residues"][i])
        best_per_piece.append((bp, br))
    print_poles_in_range(mp, E_lo, E_hi)

    # Assign & evaluate
    for i, (E_start, E_end, s_center) in enumerate(wins):
        mask = (E_test >= E_start) & (E_test <= E_end)
        if not np.any(mask):
            continue
        s = np.sqrt(E_test[mask])
        poles_i, res_i = best_per_piece[i]
        f = evaluate(s, poles_i, res_i)     # shape (K, |mask|), f = sigma * E
        # print(f"Slice {i}: E=({E_start:.2f}, {E_end:.2f}), Poles={poles_i}")
        sig = f / E_test[mask][None, :]     # back to sigma
        if have_fis:
            R_s[mask], R_a[mask], R_f[mask] = sig[0], sig[1], sig[2]
        else:
            R_s[mask], R_a[mask] = sig[0], sig[1]

    # Relative error like trainer (no floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_s = np.abs(R_s - xs_ref_s) / xs_ref_s
        rel_a = np.abs(R_a - xs_ref_a) / xs_ref_a
        rel_f = np.abs(R_f - xs_ref_f) / xs_ref_f if (xs_ref_f is not None and have_fis) else None

    # Plot per MT with twin y-axes, exactly your style
    def plot_one(mt, xs_ref, xs_fit, rel):
        if xs_ref is None or not np.any(np.isfinite(xs_ref)):
            return
        fig, ax1 = plt.subplots()
        lns1 = ax1.semilogy(E_test, xs_ref, 'g', label="ACE xs")
        lns2 = ax1.semilogy(E_test, xs_fit, 'b', label="VF xs")
        ax2 = ax1.twinx()
        lns3 = ax2.semilogy(E_test, rel, 'r', label="Relative error", alpha=0.5)
        lns = lns1 + lns2 + lns3
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc='best')
        ax1.set_xlabel('energy (eV)')
        ax1.set_ylabel('cross section (b)', color='b')
        ax1.tick_params('y', colors='b')
        ax2.set_ylabel('relative error', color='r')
        ax2.tick_params('y', colors='r')
        plt.title(f"MT {mt} vector fitted — {E_lo:.0f}–{E_hi:.0f} eV")
        fig.tight_layout()
        if out_dir:
            fname = os.path.join(out_dir, f"{E_lo:.0f}-{E_hi:.0f}_MT{mt}.png")
            plt.savefig(fname, dpi=200)
            if log: print(f"Saved {fname}")
            plt.close()
        else:
            plt.show()

    plot_one(2,  xs_ref_s, R_s, rel_s)
    plot_one(27, xs_ref_a, R_a, rel_a)
    if have_fis and xs_ref_f is not None:
        plot_one(18, xs_ref_f, R_f, rel_f)


def print_poles_in_range(mp_data, E_lo, E_hi):
    s_lo, s_hi = np.sqrt(E_lo), np.sqrt(E_hi)
    collected = []
    for piece_idx, poles in enumerate(mp_data["poles"]):
        for p in poles:
            if s_lo <= p.real <= s_hi:
                collected.append((p.real, piece_idx, p))
    for _, piece_idx, p in sorted(collected, key=lambda x: x[0]):
        print(f"piece {piece_idx:02d} | E≈{p.real**2:.2e} eV | "
              f"real={p.real:.2f} imag={p.imag:.2e}")
    # print number of poles
    print(f"Total poles in [{E_lo}, {E_hi}] eV: {len(collected)} poles")

