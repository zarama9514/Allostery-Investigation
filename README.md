This repository contains code from user zarama9514.

## Scripts

`AllIn_geometry.py`
- Provides class-based MDAnalysis geometry analysis blocks.
- Includes `RMSDAnalyzer`, `RMSFAnalyzer`, and `HelicityAnalyzer`.
- Accepts `psf` and `dcd` inputs and returns numeric arrays for downstream plotting.

`AllIn_plot.py`
- Provides reusable plotting classes for analysis outputs.
- Builds RMSD/RMSF profiles, `|Delta RMSD|`, `|Delta RMSF|`, and helicity plots.
- Saves figures to PNG and supports headless plotting (`Agg` backend).
