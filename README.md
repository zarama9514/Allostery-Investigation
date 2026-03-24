This repository contains code from user zarama9514.

## Repository Structure

`scripts/AllIn_geometry.py`
- Class-based MDAnalysis geometry analysis.
- Includes `RMSDAnalyzer`, `RMSFAnalyzer`, and `HelicityAnalyzer`.

`scripts/AllIn_plot.py`
- Reusable plotting blocks for RMSD/RMSF/helicity.
- Includes `|Delta RMSD|`, `|Delta RMSF|`, and DCCM heatmap plotting.
- Supports normalized DCCM heatmap difference plotting as `(H1 - H2) / 2` with range `[-1, 1]`.
- Includes `HydrogenBondPlotter` for top-`N` protein-protein H-bond lifetime barplots (`x`: lifetime, `y`: residue pair + subunits).

`scripts/AllIn_hbond.py`
- Protein-protein hydrogen bond analysis from `psf + dcd` using MDAnalysis.
- Excludes ligand-like small segments by default and supports `skip_first_n_frames`.
- Writes `.csv` with contact identity (residue + subunit + atom) and lifetime metrics.
- Example analysis call: `ProteinHydrogenBondAnalyzer(...).run(output_csv=..., skip_first_n_frames=50, step=10)`.
- Result plotting: `HydrogenBondPlotter(...).plot_top_contacts(csv_path=..., top_n=15)`.

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

`scripts/AllIn_run_AB_combined.py`
- End-to-end A/B analysis runner with per-RUN cutoff and combined aggregation across matching RUN ids.
- Uses `skip_first_n_frames` for each RUN before aggregation.
- Helicity is computed for mGlu3 alpha-loop target residues `665-670` (configurable via `--helicity-resids`).
- Builds community PDB outputs from combined mean DCCM:
  - `results/AB_results/01_system_A/community_A_combined_frame0.pdb`
  - `results/AB_results/02_system_B/community_B_combined_frame0.pdb`

`scripts/AllIn_run_MUT_vs_A_combined.py`
- End-to-end mutant-vs-A runner with per-RUN cutoff and combined aggregation across matching RUN ids.
- For each mutant, computes A-vs-mutant plots and matrices, bound/unbound mGlu3 blocks for A and mutant, bound mGlu3 A-vs-mutant block, H-bond csv/top15 plots, and community PDBs from combined mean DCCM.
- Uses `skip_first_n_frames` for each RUN and helicity residues via `--helicity-resids`.
- Does not include A-vs-B arrestin asymmetry/symmetry comparison logic.

`requierements/pyproject.toml`
`requierements/uv.lock`
`requierements/requirements.txt`
- Dependency manifests for the project.

`results/AB_results/`
- Output directory for combined A/B analysis across all matched RUNs.
- Includes:
  - `01_system_A/`: system A outputs (`community_A_combined_frame0.pdb`, H-bond csv/png).
  - `02_system_B/`: system B outputs (`community_B_combined_frame0.pdb`, H-bond csv/png).
  - `03_A_vs_B/`: A vs B plots (RMSD/RMSF/helicity/DCCM and delta variants).
  - `04_arrestin_A_vs_B/`: arrestin-focused comparison plots.
  - `05_mGlu3_bound_vs_unbound_A/`: mGlu3 bound/unbound comparisons inside system A.
  - `summary_combined.json`: aggregated metadata and key assignments for combined runs.

`results/mut_results/`
- Output directory for combined mutant-vs-A analysis.
- Includes one folder per mutant (for example `R668G/`, `I669G/`, `F670G/`), each containing:
  - `01_A_vs_mutant/`: A vs mutant plots and DCCM/delta DCCM.
  - `02_mGlu_bound_vs_unbound_A/`: bound/unbound mGlu3 in A.
  - `03_mGlu_bound_vs_unbound_mutant/`: bound/unbound mGlu3 in the mutant.
  - `04_bound_mGlu_A_vs_mutant/`: bound mGlu3 A vs mutant comparison.
  - `05_hbond/`: protein-protein H-bond csv and top15 barplots for A and mutant.
  - `06_community/`: community PDB outputs for A and mutant from combined DCCM.
  - `summary_mutant.json`: per-mutant metadata and assignments.
- `summary_mut_vs_A_combined.json`: global execution summary across all mutants.
- `mutants_info/`: compact mutant metadata and quick-reference table.
  - `mutants_info/README.md`: human-readable summary for mutant set.
  - `mutants_info/mutant_metadata.json`: machine-readable extracted metadata.

## Executable Paths

- Geometry/analysis module: `scripts/AllIn_geometry.py`
- Plot module: `scripts/AllIn_plot.py`
- Hydrogen bonds module: `scripts/AllIn_hbond.py`
- DCCM module: `scripts/AllIn_DCCM.py`
- Community module: `scripts/AllIn_community.py`
- PSF cleaner: `scripts/AllIn_psf_cleaner.py`
- AB combined runner: `scripts/AllIn_run_AB_combined.py`
- MUT vs A combined runner: `scripts/AllIn_run_MUT_vs_A_combined.py`
- System-specific run scripts should be named by your internal system labels and added under `scripts/` when needed.
