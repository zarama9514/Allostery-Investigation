This repository contains code from user zarama9514.

## Repository Structure

`scripts/AllIn_geometry.py`
- Class-based MDAnalysis geometry analysis.
- Includes `RMSDAnalyzer`, `RMSFAnalyzer`, and `HelicityAnalyzer`.

`scripts/AllIn_plot.py`
- Reusable plotting blocks for RMSD/RMSF/helicity.
- Includes `|Delta RMSD|` and `|Delta RMSF|` plotting.

`scripts/AllIn_psf_cleaner.py`
- PSF cleanup helper for protein-only topology workflows.

`scripts/AllIn_run_geometry_plots.py`
- Main executable for running calculations and generating plots.

`requierements/pyproject.toml`
`requierements/uv.lock`
`requierements/requirements.txt`
- Dependency manifests for the project.

`plots/`
- Output folder for generated figures and helper PDB files.

## Executable Paths

- Geometry/analysis module: `scripts/AllIn_geometry.py`
- Plot module: `scripts/AllIn_plot.py`
- PSF cleaner: `scripts/AllIn_psf_cleaner.py`
- End-to-end plotting run: `scripts/AllIn_run_geometry_plots.py`
