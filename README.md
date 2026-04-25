# Allostery Investigation

MD analysis and figure-generation code for the current project snapshot.

## Repository Layout

```text
.
├── README.md
├── requierements/
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── uv.lock
├── results_4/
│   └── mfti_2026_ru_v2/
├── scripts/
│   ├── calculations/
│   └── graphics/
└── nastya/
```

## Code Organization

### `scripts/calculations/`
Reusable analysis modules used by the plotting runners.

- `AllIn_geometry.py`
  - RMSD, RMSF, and helicity analyzers.
- `AllIn_DCCM.py`
  - DCCM calculations and comparison helpers.
- `AllIn_community.py`
  - Community analysis on top of DCCM matrices.
- `AllIn_hbond.py`
  - Protein-protein H-bond analysis.
- `AllIn_phospho_coupling.py`
  - Phospho-tail SASA, contacts, and salt-bridge analysis.
- `AllIn_psf_cleaner.py`
  - PSF cleanup helpers for protein-only workflows.
- `AllIn_run_AB_combined.py`
  - Shared helpers for A/B-style combined analyses.
- `AllIn_run_MUT_vs_A_combined.py`
  - Shared helpers for mutant-vs-reference analyses.
- `AllIn_plot.py`
  - Shared plotting helpers.
- `AllIn_signal_path.py`
  - Signal-path analysis utilities.

### `scripts/graphics/`
System-specific scripts that generate the published figures and tables.

- `AllIn_make_mglu3_rmsf_runs.py`
- `AllIn_make_rmsd_panels_mfti_ru.py`
- `AllIn_make_rmsf_cluster_runs.py`
- `AllIn_make_system_rmsf_runs.py`
- `AllIn_refresh_arrestin_delta_rmsf_overlay.py`
- `AllIn_refresh_dccm_full_protein_v2.py`
- `AllIn_refresh_dccm_v2.py`
- `AllIn_refresh_hbond_mglu3_arrestin_barplot.py`
- `AllIn_refresh_phospho_distribution_plots.py`
- `AllIn_refresh_phospho_key_site_summary_v2.py`
- `AllIn_run_contact_filtered_communities.py`
- `AllIn_run_mfti_mutants_ru.py`
- `AllIn_run_mutants_mfti_ru_v2.py`
- `AllIn_run_phospho_tail_coupling.py`
- `AllIn_run_phospho_tail_sasa_mfti_ru_v2.py`
- `AllIn_run_RMSF_v2.py`
- `AllIn_build_phospho_explanation.py`
- `analyze_signal_paths.py`
- `build_all_systems_comparison_bar.py`
- `build_arrestin_delta_rmsf_ru.py`
- `build_connectedness_vs_mobility.py`
- `build_delta_rmsf_all_mutants_ru.py`
- `build_delta_rmsf_arrestin_all_mutants_ru.py`
- `build_final_phospho_summary_3panel_ru.py`
- `build_handshake_interface_plots.py`
- `build_helicity_kde_ru.py`
- `build_interface_collapse_comparison.py`
- `build_interface_occupancy_final.py`
- `build_mglu3_delta_rmsf_three_state_ru.py`
- `build_phospho_key_site_summary_ru_v3.py`
- `compute_nam_rigidity_metrics.py`
- `export_path_vmd.py`
- `final_asymmetric_gate_visualization.py`
- `final_icl2_asymmetry.py`
- `final_icl2_asymmetry_physical.py`
- `final_mutant_comparison.py`
- `measure_mutant_proximity.py`

## Results

### `results_4/mfti_2026_ru_v2/`
Main curated results directory for the current MFTI figure set.

Top-level outputs include:

- `01_delta_rmsf_mutants_vs_wt_ru_all.png`
- `02_delta_dccm_mutants_vs_wt_ru_all.png`
- `03_local_hbonds_665_667_ru.png`
- `04_mglu3_no_arrestin_dccm_ru.png`
- `06_helicity/`
- `07_hbond/`
- `08_phospho_tail_coupling/`
- `09_mglu3_no_arrestin_dccm_ru.png`
- `10_mglu3_no_arrestin_community_frame0.pdb`
- `11_contact_filtered_community/`
- `12_signal_path/`
- `13_icl2_asymmetry/`
- `13_icl2_asymmetry_physical/`
- `rmsd/`
- `rmsf/`
- `rmsf_run_clustering.png`
- `rmsf_run_clustering_correlation.csv`
- `rmsf_run_matrix.csv`

Other useful folders:

- `results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/explanation/`
  - summary figures and comparison tables for phospho-tail coupling.
- `results_4/mfti_2026_ru_v2/11_contact_filtered_community/`
  - per-system community outputs with contact filtering.
- `results_4/mfti_2026_ru_v2/12_signal_path/`
  - signal-path summary figures and reports.
- `results_4/mfti_2026_ru_v2/13_icl2_asymmetry/`
  - asymmetry summary figures for the ICL2/gating analysis.

## Environment

The project is set up to run with `uv`.

Suggested workflow:

```bash
uv venv allostery_uv --python 3.11
uv pip install -r requierements/requirements.txt --python allostery_uv/Scripts/python.exe
```

## Notes

- Path references in this repository are intentionally relative.
- The `nastya/` subtree is treated as a separate project and should not be modified when working on the main analysis pipeline.
