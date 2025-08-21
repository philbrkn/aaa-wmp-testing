import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def _stack_poles_and_residues(poles_list, residues_list):
    """
    Concatenate all pieces as-is (no merging).
    residues_list: list over pieces; each element is [res_s, res_a, (res_f?)]
    Returns:
        P : (N_total,) complex
        R : (n_chan, N_total) complex
    """
    all_poles = []
    all_res_by_chan = None
    for p, res_chans in zip(poles_list, residues_list):
        p = np.asarray(p, dtype=complex)
        if all_res_by_chan is None:
            n_chan = len(res_chans)
            all_res_by_chan = [[] for _ in range(n_chan)]
        for k in range(n_chan):
            all_res_by_chan[k].append(np.asarray(res_chans[k], dtype=complex))
        all_poles.append(p)

    if not all_poles:
        return np.array([], dtype=complex), np.zeros((0, 0), dtype=complex)

    P = np.concatenate(all_poles, axis=0)
    R_list = [np.concatenate(rs, axis=0) for rs in all_res_by_chan]  # list of (N_total,)
    R = np.vstack(R_list)  # (n_chan, N_total)
    return P, R


def _eval_from_poles(E, poles_s, residues_by_chan):
    """
    Evaluate channels on energy grid E using s-plane poles/residues.
    residues_by_chan: (n_chan, M) complex aligned with poles_s
    Returns list of arrays [xs_elastic, xs_absorption, (xs_fission?)]
    """
    s = np.sqrt(E).astype(float)
    # Avoid accidental exact hits (extremely unlikely, but stay safe)
    # Add a tiny jitter where |s - p| is machine close
    CC = 1.0 / (s[:, None] - poles_s[None, :])
    out = []
    for k in range(residues_by_chan.shape[0]):
        xs = (CC @ residues_by_chan[k]).real
        out.append(xs)
    return out


def plot_poles_s_plane(mp_data, per_piece=False, ax=None, out_path=None, show=False, dpi=200):
    """
    Scatter-plot all poles in the s-plane (Re(p) vs Im(p)) with no merge.
    If per_piece=True, colors by piece.
    """
    poles_list = mp_data["poles"]

    if ax is None:
        fig, ax = plt.subplots()
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    if per_piece:
        for i, p in enumerate(poles_list):
            p = np.asarray(p, dtype=complex)
            if p.size == 0: 
                continue
            ax.scatter(p.real, p.imag, s=10, label=f"piece {i+1}", alpha=0.7)
        ax.legend(loc="best", fontsize="small")
    else:
        P = np.concatenate([np.asarray(p, dtype=complex) for p in poles_list]) if poles_list else np.array([], complex)
        if P.size:
            ax.scatter(P.real, P.imag, s=10, alpha=0.7)

    ax.axhline(0, lw=0.5, color="k")
    ax.set_xlabel("Re(p)")
    ax.set_ylabel("Im(p)")
    ax.set_title(f"{mp_data.get('name','')} — all poles (s-plane, no merge)")
    ax.grid(True, ls=":")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    elif owns_fig:
        plt.close(fig)


def plot_mp_global(mp_data, npts=4000, ax=None, out_path=None, show=False, dpi=200):
    """
    Evaluate/plot global xs using all poles/residues concatenated (no merge).
    """
    E_min = float(mp_data["E_min"])
    E_max = float(mp_data["E_max"])
    if not np.isfinite(E_min) or not np.isfinite(E_max) or E_max <= 0:
        raise ValueError(f"Bad energy bounds: E_min={E_min}, E_max={E_max}")
    E_lo = E_min if E_min > 0 else 1e-12
    E = np.geomspace(E_lo, E_max, npts)

    P, R = _stack_poles_and_residues(mp_data["poles"], mp_data["residues"])
    if P.size == 0:
        raise ValueError("No poles in mp_data.")

    xs_channels = _eval_from_poles(E, P, R)
    labels = ["elastic (MT=2)", "absorption (MT=27)", "fission (MT=18)"][: len(xs_channels)]

    if ax is None:
        fig, ax = plt.subplots()
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    for k, xs in enumerate(xs_channels):
        ax.semilogy(E, np.maximum(xs, 0.0), label=labels[k])
    ax.set_xlabel("energy (eV)")
    ax.set_ylabel("cross section (b)")
    ax.set_title(f"{mp_data.get('name','')} — global xs (no merge)")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    if out_path:
        os.makedirs(out_path, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    elif owns_fig:
        plt.close(fig)

    return E, {lab.split()[0].lower(): xs for lab, xs in zip(labels, xs_channels)}



if __name__ == "__main__":

    mp_data_path = "aaa_test/U238_mp.pickle"
    with open(mp_data_path, "rb") as f:
        mp_data = pickle.load(f)
    plot_mp_global(mp_data, npts=6000, out_path="plots_final_AAA")
