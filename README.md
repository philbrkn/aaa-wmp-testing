# AAA-WMP

A research implementation of the Windowed Multipole (WMP) method using the Adaptive Antoulas-Anderson (AAA) algorithm for nuclear cross-section representation in OpenMC.

## Overview

This project reformulates the Windowed Multipole library for OpenMC using AAA rational approximation instead of Vector Fitting (VF). The goal is to create a unified, temperature-independent multipole representation spanning the entire energy spectrum, enabling efficient on-the-fly Doppler broadening with minimal memory footprint.

### Why AAA over Vector Fitting?

The current WMP implementation uses Vector Fitting, which:

- Requires iterative pole relocation
- Fits nuclides in many segments (e.g., 304 pieces for U-238)
- Uses polynomial backgrounds from nearby resonances

AAA offers potential advantages:

- Greedy algorithm operating in barycentric form
- Near-optimal rational approximation in fewer iterations
- Natural multi-channel fitting with shared poles
- Pseudopole approach for background handling

### Project Status

**Current**: Resolved Resonance Range (RRR) fitting for U-238 at 0K  
**Next**: Doppler broadening validation at multiple temperatures  
**Future**: Extension to URR (with NIG distributions) and fast range

## Installation

```bash
# Create conda environment with all dependencies
conda create -n aaa-wmp-env -c conda-forge openmc scipy h5py numpy matplotlib njoy2016
conda activate aaa-wmp-env

# Install OpenMC in development mode (from repository root)
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENMC=0.0+local python -m pip install -e ./openmc
```

## Project Structure

```
├── aaa_wmp/                    # Main package
│   ├── core/                   # Core algorithms
│   │   ├── aaa_fitting.py      # MIAAA implementation (multi-channel AAA)
│   │   ├── conversion.py       # Barycentric → poles/residues conversion
│   │   └── cleanup.py          # Spurious pole removal
│   ├── processing/             # Fitting pipeline
│   │   ├── nuclide_fitter.py   # Main fitting orchestration
│   │   └── piece_fitting.py    # Single-piece fitting logic
│   ├── io/                     # Input/output
│   │   ├── njoy_interface.py   # NJOY data extraction and caching
│   │   └── output_writer.py    # WMP HDF5 and pickle output
│   ├── visualization/          # Plotting utilities
│   │   └── plotting.py
│   └── constants.py            # Physical constants, WMP version
│
├── scripts/                    # Entry points
│   ├── run_aaa.py              # Main fitting script
│   ├── run_wmp_8.0_v3.py       # Official WMP generation (for comparison)
│   ├── analyze_wmp.py          # WMP file analysis
│   └── plot_*.py               # Various plotting scripts
│
└── data/
    ├── input/
    │   ├── ENDF/               # ENDF-VIII evaluation files
    │   ├── NJOY_pickles/       # Cached 0K pointwise cross sections
    │   └── official_WMP_h5s/   # Reference WMP files for comparison
    └── output/
        ├── mp_data_output/     # Multipole data (pickle format)
        ├── aaa_in_h5wmp_format/ # WMP HDF5 files from AAA
        └── plots/              # Generated figures
```

## Usage

### Basic Fitting

```python
python scripts/run_aaa.py
```

Configuration is set in `run_aaa.py`:

```python
config_dict = {
    "vf_pieces": 1,              # Number of energy windows (1 = fit entire range)
    "mmax": 600,                 # Maximum number of poles
    "rtol": 5e-3,                # Relative tolerance for AAA convergence
    "njoy_error": 5e-4,          # NJOY linearization tolerance
    "bounds": {"E_min": 1, "E_max": 30},  # Energy bounds (eV)
    "space": "E",                # Fitting space: "E" or "sqrt_E"
    "pole_extraction": "polynomial",  # or "pseudopole"
    "output_format": "mp_data",  # "mp_data" (pickle) or "wmp" (HDF5)
}
```

### Key Parameters

| Parameter         | Description                 | Typical Values                                       |
| ----------------- | --------------------------- | ---------------------------------------------------- |
| `space`           | Energy variable for fitting | `"E"` (linear) or `"sqrt_E"` (better for resonances) |
| `rtol`            | Convergence tolerance       | 1e-3 to 5e-4 (lower = more poles)                    |
| `mmax`            | Maximum poles allowed       | 100-2000 depending on energy range                   |
| `pole_extraction` | How to handle remainder     | `"polynomial"` or `"pseudopole"`                     |

## Technical Background

### Windowed Multipole Method

Cross sections in multipole form at 0K:

$$\sigma(E, T=0) = \frac{1}{E} \sum_j \text{Re} \left[ \frac{i r_j}{\sqrt{E} - p_j} \right]$$

With Doppler broadening (free-gas model):

$$\sigma(E, T) = \frac{1}{2E\sqrt{\xi}} \sum_j \text{Re} \left[ r_j \sqrt{\pi} W_i(z) \right]$$

where $W_i(z)$ involves the Faddeeva function and $\xi = k_B T / 4A$.

### Multi-Channel AAA (MIAAA)

The implementation fits multiple reaction channels (elastic, absorption, fission) simultaneously with shared poles. This ensures physical consistency since all channels share the same resonance structure.

Key files:

- `aaa_wmp/core/aaa_fitting.py`: `miaaa_xs()` performs the fitting
- `aaa_wmp/core/conversion.py`: `proper_rational()` extracts poles/residues from barycentric form

## Validation Plan

### Doppler Broadening Validation

Temperatures to test: 293.6K, 600K, 900K, 1200K, 1500K, 1800K, 2400K

Strategy:

1. Generate NJOY reference data at each temperature
2. Reconstruct cross sections from AAA poles using Faddeeva-based broadening
3. Compare on standardized energy grids

Metrics:

- Maximum relative error in RRR
- RMS error across energy range
- Peak position accuracy
- Wing behavior at resonance edges

### Nuclides for Validation

| Nuclide | Complexity | Notes                        |
| ------- | ---------- | ---------------------------- |
| U-238   | High       | Many resonances, fissionable |
| Fe-56   | Medium     | Structural material          |
| O-16    | Low        | Few resonances               |
| Zr-91   | Medium     | Cladding material            |

## Research Roadmap

### Phase 1: RRR Validation (Current)

- [x] AAA fitting implementation
- [x] Multi-channel support
- [x] HDF5 WMP output format
- [ ] Doppler broadening validation
- [ ] Comparison with official WMP library

### Phase 2: URR Extension

- [ ] Fit continuum with AAA
- [ ] Integrate NIG distribution for statistical fluctuations
- [ ] Fallback to probability tables if needed

### Phase 3: Fast Range

- [ ] AAA for tabulated fast-range data
- [ ] Temperature-dependent threshold reactions (n,2n), (n,p)

### Phase 4: Production Library

- [ ] Process all 556 nuclides from ENDF/B-VIII.0
- [ ] OpenMC C++ integration
- [ ] Comprehensive benchmark validation

## Known Issues and Open Questions

1. **Physical Poles**: AAA produces mathematically optimal but not necessarily physically meaningful poles. May need post-processing or initialization from known resonance energies.

2. **Overfitting**: AAA accuracy is limited by ENDF linearization quality. Using `njoy_error` smaller than ENDF uncertainty is counterproductive.

3. **Window Optimization**: Trade-off between pole count per window and polynomial background complexity. Current approach uses fixed windows; adaptive windowing may improve efficiency.

4. **Threshold Reactions**: These are fitted (not physical) in current ENDF. Doppler broadening treatment needs care.

## References

1. Josey, C., Ducru, P., Forget, B., & Smith, K. (2016). Windowed multipole for cross section Doppler broadening. _Journal of Computational Physics_, 307, 715-727.

2. Ridley, G., & Forget, B. (2025). AAA algorithm for nuclear cross sections. _Proceedings of M&C 2025_, 1464-1473.

3. Hwang, R. N. (1987). A rigorous pole representation of multilevel cross sections and its practical applications. _Nuclear Science and Engineering_, 96(3), 192-209.

4. Nakagawa, Y., & Tamagawa, K. (1962). The pole representation of low energy resonances. _Nuclear Physics_, 39, 133-142.

## Contact

For questions about this research, contact the development team or Professor Benoit Forget at MIT.
