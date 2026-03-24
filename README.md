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

`scripts/AllIn_psf_cleaner.py`
- PSF cleanup helper for protein-only topology workflows.

`scripts/AllIn_run_DCCM_local_check.py`
- Local verification script for:
  `C:\Users\Daniil\IT_projects\Mika_project\results\A\step5_input.psf`
  and
  `C:\Users\Daniil\IT_projects\Mika_project\results\A\step7_productionRUN5.nowat.dcd`
- Writes test heatmap into `plots/`.

`requierements/pyproject.toml`
`requierements/uv.lock`
`requierements/requirements.txt`
- Dependency manifests for the project.

## Executable Paths

- Geometry/analysis module: `scripts/AllIn_geometry.py`
- Plot module: `scripts/AllIn_plot.py`
- DCCM module: `scripts/AllIn_DCCM.py`
- PSF cleaner: `scripts/AllIn_psf_cleaner.py`
- DCCM local check run: `scripts/AllIn_run_DCCM_local_check.py`
- System-specific run scripts should be named by your internal system labels and added under `scripts/` when needed.
