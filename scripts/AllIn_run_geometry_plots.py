from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np

from AllIn_geometry import HelicityAnalyzer, RMSFAnalyzer
from AllIn_plot import HelicityPlotter, RMSDDifferencePlotter, RMSFDifferencePlotter, RMSProfilePlotter


def per_residue_rmsd(
    psf_path: str,
    dcd_path: str,
    selection: str,
    skip_first_n_frames: int = 0,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    u = mda.Universe(psf_path, dcd_path)
    atoms = u.select_atoms(selection)
    if atoms.n_atoms == 0:
        raise ValueError(f"Selection has no atoms: {selection}")
    u.trajectory[skip_first_n_frames]
    ref = atoms.positions.copy()
    acc = np.zeros(atoms.n_atoms, dtype=float)
    n = 0
    for _ in u.trajectory[skip_first_n_frames::step]:
        disp = atoms.positions - ref
        acc += np.sum(disp * disp, axis=1)
        n += 1
    if n == 0:
        raise ValueError("No frames selected after skipping")
    rmsd = np.sqrt(acc / n)
    return atoms.resids.copy(), rmsd


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(r"C:\Users\Daniil\IT_projects\Mika_project")
    psf_path = data_dir / "results" / "A" / "step5_input.psf"
    dcd_path = data_dir / "results" / "A" / "step7_productionRUN5.nowat.dcd"
    out_dir = project_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = out_dir / "step5_input_cleaned_first_frame.pdb"
    u = mda.Universe(str(psf_path), str(dcd_path))
    u.trajectory[0]
    u.atoms.write(str(pdb_path))
    mda.Universe(str(psf_path), str(pdb_path))

    skip_first_n_frames = 100
    step = 10

    rms_plotter = RMSProfilePlotter(figsize=(12, 5), dpi=300)
    drmsd_plotter = RMSDDifferencePlotter(figsize=(12, 5), dpi=300)
    drmsf_plotter = RMSFDifferencePlotter(figsize=(12, 5), dpi=300)
    helicity_plotter = HelicityPlotter(figsize=(12, 5), dpi=300)

    resids_b_rmsd, rmsd_b = per_residue_rmsd(
        str(psf_path),
        str(dcd_path),
        "segid B and name CA",
        skip_first_n_frames=skip_first_n_frames,
        step=step,
    )
    resids_r_rmsd, rmsd_r = per_residue_rmsd(
        str(psf_path),
        str(dcd_path),
        "segid R and name CA",
        skip_first_n_frames=skip_first_n_frames,
        step=step,
    )
    if len(resids_b_rmsd) != len(resids_r_rmsd):
        raise ValueError("RMSD comparison requires equal amino-acid counts in both systems/subunits")

    rms_plotter.plot_rmsd(
        amino_acid_numbers=resids_b_rmsd,
        rmsd_values=rmsd_b,
        title="RMSD Profile (Chain B, CA)",
        save_path=str(out_dir / "rmsd_profile_chain_B.png"),
    )
    rms_plotter.plot_rmsd(
        amino_acid_numbers=resids_r_rmsd,
        rmsd_values=rmsd_r,
        title="RMSD Profile (Chain R, CA)",
        save_path=str(out_dir / "rmsd_profile_chain_R.png"),
    )
    drmsd_plotter.plot_difference(
        amino_acid_numbers=resids_b_rmsd,
        rmsd_values_a=rmsd_b,
        rmsd_values_b=rmsd_r,
        title="|Delta RMSD| Profile (Chain B vs Chain R)",
        save_path=str(out_dir / "delta_rmsd_profile_B_vs_R.png"),
    )

    rmsf_analyzer = RMSFAnalyzer(str(psf_path), str(dcd_path))
    resids_b_rmsf, rmsf_b = rmsf_analyzer.calculate(
        target_selection="segid B and name CA",
        align_selection="(segid B or segid R) and backbone",
        ref_frame=0,
        step=step,
        skip_first_n_frames=skip_first_n_frames,
    )
    resids_r_rmsf, rmsf_r = rmsf_analyzer.calculate(
        target_selection="segid R and name CA",
        align_selection="(segid B or segid R) and backbone",
        ref_frame=0,
        step=step,
        skip_first_n_frames=skip_first_n_frames,
    )
    if len(resids_b_rmsf) != len(resids_r_rmsf):
        raise ValueError("RMSF comparison requires equal amino-acid counts in both systems/subunits")

    rms_plotter.plot_rmsf(
        amino_acid_numbers=resids_b_rmsf,
        rmsf_values=rmsf_b,
        title="RMSF Profile (Chain B, CA)",
        save_path=str(out_dir / "rmsf_profile_chain_B.png"),
    )
    rms_plotter.plot_rmsf(
        amino_acid_numbers=resids_r_rmsf,
        rmsf_values=rmsf_r,
        title="RMSF Profile (Chain R, CA)",
        save_path=str(out_dir / "rmsf_profile_chain_R.png"),
    )
    drmsf_plotter.plot_difference(
        amino_acid_numbers=resids_b_rmsf,
        rmsf_values_a=rmsf_b,
        rmsf_values_b=rmsf_r,
        title="|Delta RMSF| Profile (Chain B vs Chain R)",
        save_path=str(out_dir / "delta_rmsf_profile_B_vs_R.png"),
    )

    helicity_analyzer = HelicityAnalyzer(str(psf_path), str(dcd_path))
    helicity_data = helicity_analyzer.calculate(
        amino_acids=[665, 666, 667, 668, 669, 670],
        segids=("B", "R"),
        step=step,
        skip_first_n_frames=skip_first_n_frames,
    )
    if isinstance(helicity_data, str):
        raise ValueError(helicity_data)
    helicity_plotter.plot_from_geometry_output(
        helicity_data,
        title="Helicity (%) vs Molecular Dynamics Step",
        save_path=str(out_dir / "helicity_profile_B_R.png"),
    )

    print(f"plots_dir={out_dir}")
    print(f"pdb_path={pdb_path}")
    print("done=true")


if __name__ == "__main__":
    main()
