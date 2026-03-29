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

`scripts/AllIn_phospho_coupling.py`
- Phospho-tail coupling analysis for catalytic/scaffold questions.
- Computes phospho-residue SASA distributions for `856, 857, 859, 860` in tail segids such as `T` and `L`.
- Computes phospho-tail (`856-864`) contacts with arrestin `N-domain 1-180`.
- Computes phosphate oxygen (`O1P/O2P/OT`) salt bridges to arrestin `Lys/Arg` sidechain hydrogens with cutoff `<= 2.5 Å`.
- Supports both `skip_first_n_frames` and frame subsampling `step` with default `10`.

`scripts/AllIn_run_phospho_tail_coupling.py`
- End-to-end runner for phospho-tail coupling analysis across systems `A`, `B`, `F670G`, `I669G`, and `R668G`.
- For `B`, analyzes both phospho-tail/arrestin pairs: `T_to_A` and `L_to_C`.
- Aggregates across matched `RUN` ids and writes summary metadata plus CSV/PNG outputs for SASA, tail contacts, and salt bridges.

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

`scripts/AllIn_run_RMSF_v2.py`
- RMSF v2 runner for the common A-like comparison space `A/B/R/S/T`.
- Builds pairwise comparisons for `A vs B`, `A vs F670G`, `A vs I669G`, and `A vs R668G`.
- In each comparison folder, writes exactly two figures:
  - `overlay_RMSF.png`
  - `delta_RMSF_abs.png`
- Uses the current annotated plotting logic with:
  - explicit `System` legend for blue/red overlay lines,
  - colored domain/chain legend below the plot,
  - mGlu3 domain split `VFT 30-508`, `CRD 509-574`, `TMD 575-831`, `C-tail 832-837`,
  - separate phospho-tail block `856-864`.

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

`results_2/RMSF_v2/`
- RMSF-only comparison block with the current annotated plot style.
- Contains the plotting wrapper:
  - `results_2/RMSF_v2/plot_rmsf_v2.py`
- Contains one subfolder per comparison:
  - `results_2/RMSF_v2/A vs B/`
  - `results_2/RMSF_v2/A vs F670G/`
  - `results_2/RMSF_v2/A vs I669G/`
  - `results_2/RMSF_v2/A vs R668G/`
- Each comparison subfolder contains:
  - `overlay_RMSF.png`
  - `delta_RMSF_abs.png`
- `summary.json` stores run ids and output paths for the RMSF v2 block.

`results_2/phospho_tail_coupling/`
- Phospho-tail catalytic/scaffold coupling block.
- Contains the runnable wrapper:
  - `results_2/phospho_tail_coupling/run_phospho_tail_coupling.py`
- Contains one system folder per analyzed system:
  - `results_2/phospho_tail_coupling/A/`
  - `results_2/phospho_tail_coupling/B/`
  - `results_2/phospho_tail_coupling/F670G/`
  - `results_2/phospho_tail_coupling/I669G/`
  - `results_2/phospho_tail_coupling/R668G/`
- Each system contains one or more pair folders such as `T_to_A/` or `L_to_C/`.
- Each pair folder contains:
  - `01_sasa/`
    - per-frame phospho-residue SASA csv
    - summary SASA csv
    - phospho-residue SASA distribution png
    - total phospho-SASA distribution png
  - `02_tail_n_domain_contacts/`
    - tail/N-domain contact lifetime csv
    - per-frame contact count csv
    - top-15 contact lifetime barplot
    - contact-count timeseries
  - `03_salt_bridges/`
    - atom-level salt-bridge csv
    - residue-level salt-bridge csv
    - candidate `Lys/Arg` donor csv
    - per-frame salt-bridge count csv
    - top-15 salt-bridge lifetime barplot
    - salt-bridge-count timeseries
- Also contains phospho-SASA comparison folders for mutant-vs-A:
  - `results_2/phospho_tail_coupling/A vs F670G/`
  - `results_2/phospho_tail_coupling/A vs I669G/`
  - `results_2/phospho_tail_coupling/A vs R668G/`
- Each comparison folder contains:
  - `delta_mean_sasa_A_minus_mutant.csv`
  - `delta_mean_sasa_A_minus_mutant.png`
- `delta_mean_sasa_A_minus_mutant.csv` stores signed deltas `A - mutant` for mean and median SASA.
- `summary.json` stores system, pair, run, and output-path metadata for the whole phospho-tail coupling block.
- `explanation/`
  - `README.md`: written interpretation of how the phospho-tail results agree or disagree with the paper.
  - `key_site_metrics.csv`: compact comparison table for pS857, pS859, pT860, and total phospho-tail SASA.
  - `phospho_key_site_summary.png`: summary heatmap of SASA, tail-contact occupancy, and salt-bridge occupancy.
  - `A_minus_mutants_signed_delta_sasa.png`: signed `A - mutant` mean SASA comparison for the key phosphosites.

## Executable Paths

- Geometry/analysis module: `scripts/AllIn_geometry.py`
- Plot module: `scripts/AllIn_plot.py`
- Hydrogen bonds module: `scripts/AllIn_hbond.py`
- DCCM module: `scripts/AllIn_DCCM.py`
- Community module: `scripts/AllIn_community.py`
- PSF cleaner: `scripts/AllIn_psf_cleaner.py`
- Phospho coupling module: `scripts/AllIn_phospho_coupling.py`
- Phospho explanation builder: `scripts/AllIn_build_phospho_explanation.py`
- AB combined runner: `scripts/AllIn_run_AB_combined.py`
- MUT vs A combined runner: `scripts/AllIn_run_MUT_vs_A_combined.py`
- RMSF v2 runner: `scripts/AllIn_run_RMSF_v2.py`
- Phospho-tail coupling runner: `scripts/AllIn_run_phospho_tail_coupling.py`
- RMSF v2 plotting wrapper: `results_2/RMSF_v2/plot_rmsf_v2.py`
- Phospho-tail coupling wrapper: `results_2/phospho_tail_coupling/run_phospho_tail_coupling.py`
- System-specific run scripts should be named by your internal system labels and added under `scripts/` when needed.
