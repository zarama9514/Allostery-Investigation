from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from AllIn_geometry import RMSFAnalyzer
from AllIn_plot import RMSFDifferencePlotter, RMSProfilePlotter, build_residue_annotation_blocks
from AllIn_run_AB_combined import collect_runs, progress_iter, save_json


COMMON_SEGIDS = ["A", "B", "R", "S", "T"]
DEFAULT_MUTANTS = ["F670G", "I669G", "R668G"]
COMMON_DEFINITIONS = [
    {"segid": "A", "start_resid": 1, "end_resid": 376, "label": "A arrestin", "color": "#d00000", "alpha": 0.18},
    {"segid": "B", "start_resid": 30, "end_resid": 508, "label": "B mGlu3 VFT", "color": "#005f73", "alpha": 0.18},
    {"segid": "B", "start_resid": 509, "end_resid": 574, "label": "B mGlu3 CRD", "color": "#0a9396", "alpha": 0.18},
    {"segid": "B", "start_resid": 575, "end_resid": 831, "label": "B mGlu3 TMD", "color": "#ee9b00", "alpha": 0.18},
    {"segid": "B", "start_resid": 832, "end_resid": 837, "label": "B mGlu3 C-tail", "color": "#ca6702", "alpha": 0.18},
    {"segid": "R", "start_resid": 30, "end_resid": 508, "label": "R mGlu3 VFT", "color": "#1d3557", "alpha": 0.18},
    {"segid": "R", "start_resid": 509, "end_resid": 574, "label": "R mGlu3 CRD", "color": "#457b9d", "alpha": 0.18},
    {"segid": "R", "start_resid": 575, "end_resid": 831, "label": "R mGlu3 TMD", "color": "#e76f51", "alpha": 0.18},
    {"segid": "R", "start_resid": 832, "end_resid": 837, "label": "R mGlu3 C-tail", "color": "#f4a261", "alpha": 0.18},
    {"segid": "S", "start_resid": 1, "end_resid": 251, "label": "S scFv30", "color": "#6c757d", "alpha": 0.18},
    {"segid": "T", "start_resid": 856, "end_resid": 864, "label": "T phospho tail", "color": "#6a4c93", "alpha": 0.18},
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_results = Path(r"C:\Users\Daniil\IT_projects\Mika_project\results")
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-psf", default=str(default_results / "A" / "step5_input_protein.psf"))
    parser.add_argument("--a-dir", default=str(default_results / "A"))
    parser.add_argument("--b-psf", default=str(default_results / "B" / "B_step5_input_protein.psf"))
    parser.add_argument("--b-dir", default=str(default_results / "B"))
    parser.add_argument("--mut-root", default=str(default_results))
    parser.add_argument("--mutants", default=",".join(DEFAULT_MUTANTS))
    parser.add_argument("--results-root", default=str(project_root / "results_2" / "RMSF_v2"))
    parser.add_argument("--skip-first-n-frames", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def selection_for(segids: Sequence[str], atom_clause: str) -> str:
    return f"segid {' '.join(segids)} and protein and {atom_clause}"


def axis_keys(segids: Sequence[str] | np.ndarray, resids: Sequence[int] | np.ndarray) -> list[tuple[str, int]]:
    return [(str(segid), int(resid)) for segid, resid in zip(np.asarray(segids, dtype=str), np.asarray(resids, dtype=int))]


def aggregate_rmsf_detailed(
    psf: str,
    dcd_files: Sequence[Path],
    target_selection: str,
    align_selection: str,
    skip: int,
    label: str,
) -> dict[str, np.ndarray]:
    ref: dict[str, np.ndarray] | None = None
    values: list[np.ndarray] = []
    for dcd in progress_iter(dcd_files, desc=f"{label}: runs", unit="run"):
        out = RMSFAnalyzer(psf=psf, dcd=str(dcd)).calculate_detailed(
            target_selection=target_selection,
            align_selection=align_selection,
            skip_first_n_frames=skip,
            step=1,
        )
        if ref is None:
            ref = out
        elif axis_keys(ref["segids"], ref["resids"]) != axis_keys(out["segids"], out["resids"]):
            raise ValueError(f"RMSF axis mismatch in {label}")
        values.append(np.asarray(out["rmsf"], dtype=float))
    if ref is None:
        raise ValueError(f"No RMSF values collected for {label}")
    return {
        "segids": np.asarray(ref["segids"], dtype=str),
        "resids": np.asarray(ref["resids"], dtype=int),
        "rmsf": np.mean(np.asarray(values, dtype=float), axis=0),
    }


def align_rmsf_to_reference(reference: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], label: str) -> dict[str, np.ndarray]:
    ref_keys = axis_keys(reference["segids"], reference["resids"])
    other_map = {
        key: float(value)
        for key, value in zip(axis_keys(other["segids"], other["resids"]), np.asarray(other["rmsf"], dtype=float))
    }
    missing = [key for key in ref_keys if key not in other_map]
    if missing:
        preview = ", ".join(f"{segid}:{resid}" for segid, resid in missing[:5])
        raise ValueError(f"RMSF alignment mismatch in {label}; missing keys: {preview}")
    return {
        "segids": np.asarray(reference["segids"], dtype=str),
        "resids": np.asarray(reference["resids"], dtype=int),
        "rmsf": np.asarray([other_map[key] for key in ref_keys], dtype=float),
    }


def comparison_folder_name(system_name: str) -> str:
    return f"A vs {system_name}"


def render_comparison(
    output_dir: Path,
    label_other: str,
    rmsf_a: Mapping[str, np.ndarray],
    rmsf_other: Mapping[str, np.ndarray],
    annotation_blocks: Sequence[Mapping[str, object]],
    plotter: RMSProfilePlotter,
    diff_plotter: RMSFDifferencePlotter,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay_RMSF.png"
    delta_path = output_dir / "delta_RMSF_abs.png"
    plotter.plot_rmsf_overlay(
        rmsf_a["resids"],
        rmsf_a["rmsf"],
        rmsf_other["rmsf"],
        label_a="A",
        label_b=label_other,
        color_a="blue",
        color_b="red",
        title=f"RMSF Overlay, A vs {label_other}",
        residue_annotation_blocks=annotation_blocks,
        save_path=str(overlay_path),
    )
    diff_plotter.plot_difference(
        rmsf_a["resids"],
        rmsf_a["rmsf"],
        rmsf_other["rmsf"],
        title=f"|Delta RMSF|, A vs {label_other}",
        residue_annotation_blocks=annotation_blocks,
        save_path=str(delta_path),
    )
    return {
        "overlay": str(overlay_path),
        "delta_abs": str(delta_path),
    }


def run() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    skip = int(args.skip_first_n_frames)
    target_selection = selection_for(COMMON_SEGIDS, "name CA")
    align_selection = selection_for(COMMON_SEGIDS, "backbone")
    annotation_blocks: list[Mapping[str, object]] | None = None

    plotter = RMSProfilePlotter(figsize=(20.0, 8.0), dpi=300)
    diff_plotter = RMSFDifferencePlotter(figsize=(20.0, 8.0), dpi=300)

    a_runs_all = collect_runs(Path(args.a_dir))
    a_reference_runs = sorted(a_runs_all)
    if args.max_runs and int(args.max_runs) > 0:
        a_reference_runs = a_reference_runs[: int(args.max_runs)]
    if not a_reference_runs:
        raise ValueError("No A runs detected")

    a_runs = [a_runs_all[run_id] for run_id in a_reference_runs]
    a_common_rmsf = aggregate_rmsf_detailed(args.a_psf, a_runs, target_selection, align_selection, skip, "A common RMSF")
    annotation_blocks = build_residue_annotation_blocks(
        a_common_rmsf["segids"],
        a_common_rmsf["resids"],
        COMMON_DEFINITIONS,
        include_unmatched=True,
    )

    summary: dict[str, object] = {
        "results_root": str(results_root),
        "skip_first_n_frames_per_run": skip,
        "common_segids": COMMON_SEGIDS,
        "comparisons": {},
    }

    b_runs_all = collect_runs(Path(args.b_dir))
    run_ids_ab = sorted(set(a_runs_all).intersection(b_runs_all))
    if args.max_runs and int(args.max_runs) > 0:
        run_ids_ab = run_ids_ab[: int(args.max_runs)]
    if not run_ids_ab:
        raise ValueError("No matching RUN ids found between A and B")
    a_ab = aggregate_rmsf_detailed(args.a_psf, [a_runs_all[run_id] for run_id in run_ids_ab], target_selection, align_selection, skip, "A vs B: A RMSF")
    b_ab_raw = aggregate_rmsf_detailed(args.b_psf, [b_runs_all[run_id] for run_id in run_ids_ab], target_selection, align_selection, skip, "A vs B: B RMSF")
    b_ab = align_rmsf_to_reference(a_ab, b_ab_raw, "A vs B")
    ab_dir = results_root / comparison_folder_name("B")
    summary["comparisons"]["A vs B"] = {
        "runs_used": run_ids_ab,
        "outputs": render_comparison(ab_dir, "B", a_ab, b_ab, annotation_blocks, plotter, diff_plotter),
    }

    for mutant_name in parse_csv(args.mutants):
        mutant_dir = Path(args.mut_root) / mutant_name
        mutant_psf = mutant_dir / f"{mutant_name}_step5_input_protein.psf"
        if not mutant_psf.exists():
            raise FileNotFoundError(f"Missing protein-only psf for {mutant_name}: {mutant_psf}")
        mutant_runs_all = collect_runs(mutant_dir)
        run_ids = sorted(set(a_runs_all).intersection(mutant_runs_all))
        if args.max_runs and int(args.max_runs) > 0:
            run_ids = run_ids[: int(args.max_runs)]
        if not run_ids:
            raise ValueError(f"No matching RUN ids found between A and {mutant_name}")
        a_mut = aggregate_rmsf_detailed(args.a_psf, [a_runs_all[run_id] for run_id in run_ids], target_selection, align_selection, skip, f"A vs {mutant_name}: A RMSF")
        mut_raw = aggregate_rmsf_detailed(str(mutant_psf), [mutant_runs_all[run_id] for run_id in run_ids], target_selection, align_selection, skip, f"A vs {mutant_name}: {mutant_name} RMSF")
        mut = align_rmsf_to_reference(a_mut, mut_raw, f"A vs {mutant_name}")
        out_dir = results_root / comparison_folder_name(mutant_name)
        summary["comparisons"][f"A vs {mutant_name}"] = {
            "runs_used": run_ids,
            "outputs": render_comparison(out_dir, mutant_name, a_mut, mut, annotation_blocks, plotter, diff_plotter),
        }

    save_json(results_root / "summary.json", summary)
    print(f"RMSF_v2 completed: {results_root}")


if __name__ == "__main__":
    run()
