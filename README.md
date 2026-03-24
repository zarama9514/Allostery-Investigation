This repository contains code from user zarama9514.

## Repository Structure

`scripts/AllIn_geometry.py`
- Class-based MDAnalysis geometry analysis.
- Includes `RMSDAnalyzer`, `RMSFAnalyzer`, and `HelicityAnalyzer`.

`scripts/AllIn_plot.py`
- Reusable plotting blocks for RMSD/RMSF/helicity.
- Includes `|Delta RMSD|`, `|Delta RMSF|`, and DCCM heatmap plotting.
- Supports normalized DCCM heatmap difference plotting as `(H1 - H2) / 2` with range `[-1, 1]`.

`scripts/AllIn_DCCM.py`
- DCCM analysis module for one pair (`psf1+dcd1` vs itself) or two pairs (`psf1+dcd1` vs `psf2+dcd2`).
- Supports initial-frame cutoff via `skip_first_n_frames`.

`scripts/AllIn_community.py`
- Community analysis module based on DCCM correlations.
- Takes `psf + dcd` and writes an output `.pdb` of frame 0 with all atoms.
- Encodes protein community IDs in the `beta-factor` column for visualization.
- Typical use: call `CommunityAnalyzer(...).run(output_pdb=...)` from your system-specific run script.

`scripts/AllIn_psf_cleaner.py`
- PSF cleanup helper for protein-only topology workflows.

`requierements/pyproject.toml`
`requierements/uv.lock`
`requierements/requirements.txt`
- Dependency manifests for the project.

## Executable Paths

- Geometry/analysis module: `scripts/AllIn_geometry.py`
- Plot module: `scripts/AllIn_plot.py`
- DCCM module: `scripts/AllIn_DCCM.py`
- Community module: `scripts/AllIn_community.py`
- PSF cleaner: `scripts/AllIn_psf_cleaner.py`
- System-specific run scripts should be named by your internal system labels and added under `scripts/` when needed.
