from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from AllIn_phospho_coupling import (
    PhosphoCouplingPlotter,
    PhosphoSASAAnalyzer,
    PhosphoTailContactAnalyzer,
    PhosphoTailSaltBridgeAnalyzer,
    TailPairSpec,
)
from AllIn_run_AB_combined import collect_runs, progress_iter, save_json


PHOSPHO_RESIDS = [856, 857, 859, 860]
TAIL_RESIDS = list(range(856, 865))
N_DOMAIN_RANGE = (1, 180)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    psf: Path
    run_dir: Path
    pairs: tuple[TailPairSpec, ...]


def parse_csv_names(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_system_specs(results_root: Path) -> dict[str, SystemSpec]:
    return {
        "A": SystemSpec(
            name="A",
            psf=results_root / "A" / "step5_input_protein.psf",
            run_dir=results_root / "A",
            pairs=(TailPairSpec(arrestin_segid="A", tail_segid="T", label="T_to_A"),),
        ),
        "B": SystemSpec(
            name="B",
            psf=results_root / "B" / "B_step5_input_protein.psf",
            run_dir=results_root / "B",
            pairs=(
                TailPairSpec(arrestin_segid="A", tail_segid="T", label="T_to_A"),
                TailPairSpec(arrestin_segid="C", tail_segid="L", label="L_to_C"),
            ),
        ),
        "F670G": SystemSpec(
            name="F670G",
            psf=results_root / "F670G" / "F670G_step5_input_protein.psf",
            run_dir=results_root / "F670G",
            pairs=(TailPairSpec(arrestin_segid="A", tail_segid="T", label="T_to_A"),),
        ),
        "I669G": SystemSpec(
            name="I669G",
            psf=results_root / "I669G" / "I669G_step5_input_protein.psf",
            run_dir=results_root / "I669G",
            pairs=(TailPairSpec(arrestin_segid="A", tail_segid="T", label="T_to_A"),),
        ),
        "R668G": SystemSpec(
            name="R668G",
            psf=results_root / "R668G" / "R668G_step5_input_protein.psf",
            run_dir=results_root / "R668G",
            pairs=(TailPairSpec(arrestin_segid="A", tail_segid="T", label="T_to_A"),),
        ),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_results = project_root.parent / "Mika_project" / "results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(project_root / "results_2" / "phospho_tail_coupling"))
    parser.add_argument("--source-root", default=str(default_results))
    parser.add_argument("--systems", default="A,B,F670G,I669G,R668G")
    parser.add_argument("--skip-first-n-frames", type=int, default=100)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--sasa-chunk-size", type=int, default=200)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--salt-cutoff", type=float, default=2.5)
    return parser.parse_args()


def ensure_pair_dirs(system_root: Path, pair: TailPairSpec) -> dict[str, Path]:
    pair_root = system_root / pair.label
    dirs = {
        "root": pair_root,
        "sasa": pair_root / "01_sasa",
        "contacts": pair_root / "02_tail_n_domain_contacts",
        "salt": pair_root / "03_salt_bridges",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def merge_lifetime_records(record_sets: Sequence[Sequence[Mapping[str, object]]], key_field: str) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for rows in record_sets:
        for row in rows:
            key = str(row[key_field])
            frames_observed = int(row["frames_observed"])
            if key not in merged:
                merged[key] = dict(row)
                merged[key]["_n_frames_weight"] = int(row["n_frames_analyzed"])
                merged[key]["_distance_weighted_sum"] = float(row["mean_distance_angstrom"]) * float(frames_observed)
            else:
                acc = merged[key]
                acc["frames_observed"] = int(acc["frames_observed"]) + frames_observed
                acc["lifetime_ps"] = float(acc["lifetime_ps"]) + float(row["lifetime_ps"])
                acc["max_continuous_frames"] = max(int(acc["max_continuous_frames"]), int(row["max_continuous_frames"]))
                acc["max_continuous_ps"] = max(float(acc["max_continuous_ps"]), float(row["max_continuous_ps"]))
                acc["segments_count"] = int(acc["segments_count"]) + int(row["segments_count"])
                acc["first_observed_frame"] = -1
                acc["last_observed_frame"] = -1
                acc["_n_frames_weight"] = int(acc["_n_frames_weight"]) + int(row["n_frames_analyzed"])
                acc["_distance_weighted_sum"] = float(acc["_distance_weighted_sum"]) + (
                    float(row["mean_distance_angstrom"]) * float(frames_observed)
                )
    out: list[dict[str, object]] = []
    for row in merged.values():
        total_frames = max(1, int(row["_n_frames_weight"]))
        observed = max(1, int(row["frames_observed"]))
        row["n_frames_analyzed"] = int(total_frames)
        row["occupancy_fraction"] = float(row["frames_observed"]) / float(total_frames)
        row["occupancy_percent"] = float(row["occupancy_fraction"]) * 100.0
        row["mean_distance_angstrom"] = float(row["_distance_weighted_sum"]) / float(observed)
        row.pop("_n_frames_weight", None)
        row.pop("_distance_weighted_sum", None)
        out.append(row)
    out.sort(key=lambda row: float(row["lifetime_ps"]), reverse=True)
    return out


def merge_candidate_records(record_sets: Sequence[Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for rows in record_sets:
        for row in rows:
            key = str(row["donor_residue_label"])
            if key not in merged:
                merged[key] = dict(row)
            else:
                acc = merged[key]
                acc["frames_observed"] = int(acc["frames_observed"]) + int(row["frames_observed"])
                acc["lifetime_ps"] = float(acc["lifetime_ps"]) + float(row["lifetime_ps"])
                acc["max_continuous_ps"] = max(float(acc["max_continuous_ps"]), float(row["max_continuous_ps"]))
                acc["segments_count"] = int(acc["segments_count"]) + int(row["segments_count"])
    out = list(merged.values())
    out.sort(key=lambda row: float(row["lifetime_ps"]), reverse=True)
    return out


def concat_count_series(run_ids: Sequence[str], series_sets: Sequence[Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    combined_index = 0
    for run_id, series in zip(run_ids, series_sets, strict=False):
        for row in series:
            item = dict(row)
            item["run_id"] = str(run_id)
            item["combined_frame_index"] = int(combined_index)
            rows.append(item)
            combined_index += 1
    return rows


def ensure_comparison_dir(results_root: Path, comparison_name: str) -> Path:
    path = results_root / comparison_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_sasa_delta_rows(
    reference_rows: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, object]],
    reference_system: str,
    target_system: str,
) -> list[dict[str, object]]:
    reference_map = {str(row["label"]): dict(row) for row in reference_rows}
    target_map = {str(row["label"]): dict(row) for row in target_rows}
    labels = [str(row["label"]) for row in reference_rows if str(row["label"]) in target_map]
    rows: list[dict[str, object]] = []
    for label in labels:
        ref = reference_map[label]
        tgt = target_map[label]
        delta_mean = float(ref["mean_sasa_angstrom2"]) - float(tgt["mean_sasa_angstrom2"])
        delta_median = float(ref["median_sasa_angstrom2"]) - float(tgt["median_sasa_angstrom2"])
        rows.append(
            {
                "label": label,
                "reference_system": reference_system,
                "target_system": target_system,
                "reference_mean_sasa_angstrom2": float(ref["mean_sasa_angstrom2"]),
                "target_mean_sasa_angstrom2": float(tgt["mean_sasa_angstrom2"]),
                "delta_mean_sasa_angstrom2": float(delta_mean),
                "reference_median_sasa_angstrom2": float(ref["median_sasa_angstrom2"]),
                "target_median_sasa_angstrom2": float(tgt["median_sasa_angstrom2"]),
                "delta_median_sasa_angstrom2": float(delta_median),
            }
        )
    return rows


def write_sasa_delta_csv(rows: Sequence[Mapping[str, object]], output_csv: str) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "reference_system",
        "target_system",
        "reference_mean_sasa_angstrom2",
        "target_mean_sasa_angstrom2",
        "delta_mean_sasa_angstrom2",
        "reference_median_sasa_angstrom2",
        "target_median_sasa_angstrom2",
        "delta_median_sasa_angstrom2",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output_path


def aggregate_sasa(
    psf: str,
    run_ids: Sequence[str],
    run_map: Mapping[str, Path],
    tail_segid: str,
    step: int,
    skip: int,
    chunk_size: int,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    residue_labels: list[str] | None = None
    for run_id in progress_iter(run_ids, desc=f"{tail_segid} SASA runs", unit="run"):
        out = PhosphoSASAAnalyzer(psf=psf, dcd=str(run_map[run_id])).calculate(
            tail_segid=tail_segid,
            phospho_resids=PHOSPHO_RESIDS,
            step=step,
            skip_first_n_frames=skip,
            chunk_size=chunk_size,
        )
        residue_labels = list(out["residue_labels"])
        for row in out["records"]:
            item = dict(row)
            item["run_id"] = str(run_id)
            records.append(item)
    if residue_labels is None:
        raise ValueError(f"No SASA records collected for tail {tail_segid}")
    return {"records": records, "residue_labels": residue_labels}


def aggregate_contacts(
    psf: str,
    run_ids: Sequence[str],
    run_map: Mapping[str, Path],
    pair: TailPairSpec,
    step: int,
    skip: int,
    contact_cutoff: float,
) -> dict[str, object]:
    record_sets: list[list[dict[str, object]]] = []
    series_sets: list[list[dict[str, object]]] = []
    for run_id in progress_iter(run_ids, desc=f"{pair.label} contacts runs", unit="run"):
        out = PhosphoTailContactAnalyzer(psf=psf, dcd=str(run_map[run_id])).calculate(
            tail_segid=pair.tail_segid,
            arrestin_segid=pair.arrestin_segid,
            tail_resids=TAIL_RESIDS,
            n_domain_range=N_DOMAIN_RANGE,
            heavy_atom_cutoff=contact_cutoff,
            step=step,
            skip_first_n_frames=skip,
        )
        record_sets.append(list(out["contacts"]))
        series_sets.append(list(out["count_series"]))
    return {
        "contacts": merge_lifetime_records(record_sets, key_field="contact_residue_label"),
        "count_series": concat_count_series(run_ids, series_sets),
    }


def aggregate_salt_bridges(
    psf: str,
    run_ids: Sequence[str],
    run_map: Mapping[str, Path],
    pair: TailPairSpec,
    step: int,
    skip: int,
    salt_cutoff: float,
) -> dict[str, object]:
    atom_sets: list[list[dict[str, object]]] = []
    residue_sets: list[list[dict[str, object]]] = []
    candidate_sets: list[list[dict[str, object]]] = []
    series_sets: list[list[dict[str, object]]] = []
    for run_id in progress_iter(run_ids, desc=f"{pair.label} salt runs", unit="run"):
        out = PhosphoTailSaltBridgeAnalyzer(psf=psf, dcd=str(run_map[run_id])).calculate(
            tail_segid=pair.tail_segid,
            arrestin_segid=pair.arrestin_segid,
            phospho_resids=PHOSPHO_RESIDS,
            n_domain_range=N_DOMAIN_RANGE,
            hydrogen_oxygen_cutoff=salt_cutoff,
            step=step,
            skip_first_n_frames=skip,
        )
        atom_sets.append(list(out["salt_bridges_atom"]))
        residue_sets.append(list(out["salt_bridges_residue"]))
        candidate_sets.append(list(out["candidate_residues"]))
        series_sets.append(list(out["count_series"]))
    return {
        "salt_bridges_atom": merge_lifetime_records(atom_sets, key_field="salt_bridge_atom_label"),
        "salt_bridges_residue": merge_lifetime_records(residue_sets, key_field="salt_bridge_residue_label"),
        "candidate_residues": merge_candidate_records(candidate_sets),
        "count_series": concat_count_series(run_ids, series_sets),
    }


def run() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    source_root = Path(args.source_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    systems = build_system_specs(source_root)
    selected_systems = parse_csv_names(args.systems)
    plotter = PhosphoCouplingPlotter(figsize=(12.0, 6.0), dpi=300)
    summary: dict[str, object] = {
        "results_root": str(results_root),
        "systems": [],
        "sasa_comparisons": [],
        "phospho_resids": PHOSPHO_RESIDS,
        "tail_resids": TAIL_RESIDS,
        "n_domain_range": list(N_DOMAIN_RANGE),
        "skip_first_n_frames_per_run": int(args.skip_first_n_frames),
        "step": int(args.step),
        "contact_cutoff_angstrom": float(args.contact_cutoff),
        "salt_bridge_hydrogen_oxygen_cutoff_angstrom": float(args.salt_cutoff),
    }
    sasa_summaries_by_system_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
    for system_name in progress_iter(selected_systems, desc="Systems", unit="system", leave=True):
        spec = systems[system_name]
        run_map = collect_runs(spec.run_dir)
        run_ids = sorted(run_map)
        if args.max_runs and int(args.max_runs) > 0:
            run_ids = run_ids[: int(args.max_runs)]
        if not run_ids:
            raise ValueError(f"No RUN files found for {system_name}")
        system_root = results_root / spec.name
        system_root.mkdir(parents=True, exist_ok=True)
        system_summary: dict[str, object] = {"system": spec.name, "runs_used": run_ids, "pairs": []}
        for pair in progress_iter(spec.pairs, desc=f"{system_name} pairs", unit="pair"):
            dirs = ensure_pair_dirs(system_root, pair)
            sasa = aggregate_sasa(str(spec.psf), run_ids, run_map, pair.tail_segid, int(args.step), int(args.skip_first_n_frames), int(args.sasa_chunk_size))
            sasa_summary_rows = PhosphoSASAAnalyzer.summarize_records(sasa["records"], sasa["residue_labels"])
            sasa_summaries_by_system_pair[(spec.name, pair.label)] = list(sasa_summary_rows)
            sasa_csv = PhosphoSASAAnalyzer.write_frame_csv(
                sasa["records"],
                output_csv=str(dirs["sasa"] / f"{pair.tail_segid}_phospho_sasa_frames.csv"),
            )
            sasa_summary_csv = PhosphoSASAAnalyzer.write_summary_csv(
                sasa_summary_rows,
                output_csv=str(dirs["sasa"] / f"{pair.tail_segid}_phospho_sasa_summary.csv"),
            )
            sasa_dist_png = dirs["sasa"] / f"{pair.tail_segid}_phospho_residue_sasa_distribution.png"
            sasa_total_png = dirs["sasa"] / f"{pair.tail_segid}_phospho_total_sasa_distribution.png"
            plotter.plot_sasa_distribution(
                sasa["records"],
                sasa["residue_labels"],
                title=f"{system_name} {pair.tail_segid} Phospho Residue SASA Distribution",
                save_path=str(sasa_dist_png),
            )
            plotter.plot_total_sasa_distribution(
                sasa["records"],
                title=f"{system_name} {pair.tail_segid} Total Phospho SASA Distribution",
                save_path=str(sasa_total_png),
            )

            contacts = aggregate_contacts(str(spec.psf), run_ids, run_map, pair, int(args.step), int(args.skip_first_n_frames), float(args.contact_cutoff))
            contact_csv = PhosphoTailContactAnalyzer.write_contacts_csv(
                contacts["contacts"],
                output_csv=str(dirs["contacts"] / "tail_n_domain_contacts.csv"),
            )
            contact_counts_csv = PhosphoTailContactAnalyzer.write_count_series_csv(
                contacts["count_series"],
                output_csv=str(dirs["contacts"] / "tail_n_domain_contact_counts.csv"),
            )
            contact_top_png = dirs["contacts"] / "top15_tail_n_domain_contacts_by_lifetime.png"
            contact_count_png = dirs["contacts"] / "tail_n_domain_contact_counts_timeseries.png"
            if contacts["contacts"]:
                plotter.plot_top_lifetime(
                    contacts["contacts"],
                    label_field="contact_residue_label",
                    title=f"{system_name} {pair.label} Top 15 Tail-N-domain Contacts",
                    save_path=str(contact_top_png),
                )
            plotter.plot_count_timeseries(
                contacts["count_series"],
                count_fields=["n_contact_pairs"],
                title=f"{system_name} {pair.label} Tail-N-domain Contact Counts",
                y_label="Number of Contact Pairs",
                save_path=str(contact_count_png),
            )

            salt = aggregate_salt_bridges(str(spec.psf), run_ids, run_map, pair, int(args.step), int(args.skip_first_n_frames), float(args.salt_cutoff))
            salt_atom_csv = PhosphoTailSaltBridgeAnalyzer.write_salt_bridge_csv(
                salt["salt_bridges_atom"],
                output_csv=str(dirs["salt"] / "salt_bridges_atom_level.csv"),
            )
            salt_residue_csv = PhosphoTailSaltBridgeAnalyzer.write_salt_bridge_csv(
                salt["salt_bridges_residue"],
                output_csv=str(dirs["salt"] / "salt_bridges_residue_level.csv"),
            )
            salt_candidate_csv = PhosphoTailSaltBridgeAnalyzer.write_candidate_csv(
                salt["candidate_residues"],
                output_csv=str(dirs["salt"] / "candidate_n_domain_lys_arg.csv"),
            )
            salt_counts_csv = PhosphoTailSaltBridgeAnalyzer.write_count_series_csv(
                salt["count_series"],
                output_csv=str(dirs["salt"] / "salt_bridge_counts.csv"),
            )
            salt_top_png = dirs["salt"] / "top15_salt_bridges_by_lifetime.png"
            salt_count_png = dirs["salt"] / "salt_bridge_counts_timeseries.png"
            if salt["salt_bridges_atom"]:
                plotter.plot_top_lifetime(
                    salt["salt_bridges_atom"],
                    label_field="salt_bridge_atom_label",
                    title=f"{system_name} {pair.label} Top 15 Salt Bridges",
                    save_path=str(salt_top_png),
                )
            plotter.plot_count_timeseries(
                salt["count_series"],
                count_fields=["n_salt_bridge_atom_pairs", "n_salt_bridge_residue_pairs"],
                title=f"{system_name} {pair.label} Salt Bridge Counts",
                y_label="Number of Salt Bridges",
                save_path=str(salt_count_png),
            )

            pair_summary = {
                "pair_label": pair.label,
                "tail_segid": pair.tail_segid,
                "arrestin_segid": pair.arrestin_segid,
                "outputs": {
                    "sasa_frames_csv": str(sasa_csv),
                    "sasa_summary_csv": str(sasa_summary_csv),
                    "sasa_residue_distribution_png": str(sasa_dist_png),
                    "sasa_total_distribution_png": str(sasa_total_png),
                    "contacts_csv": str(contact_csv),
                    "contact_counts_csv": str(contact_counts_csv),
                    "contact_top_png": str(contact_top_png),
                    "contact_counts_png": str(contact_count_png),
                    "salt_bridges_atom_csv": str(salt_atom_csv),
                    "salt_bridges_residue_csv": str(salt_residue_csv),
                    "salt_bridge_candidates_csv": str(salt_candidate_csv),
                    "salt_bridge_counts_csv": str(salt_counts_csv),
                    "salt_bridge_top_png": str(salt_top_png),
                    "salt_bridge_counts_png": str(salt_count_png),
                },
                "n_contact_pairs": int(len(contacts["contacts"])),
                "n_salt_bridge_atom_pairs": int(len(salt["salt_bridges_atom"])),
                "n_salt_bridge_residue_pairs": int(len(salt["salt_bridges_residue"])),
                "n_candidate_lys_arg": int(len(salt["candidate_residues"])),
            }
            system_summary["pairs"].append(pair_summary)
        save_json(
            system_root / "summary.json",
            system_summary,
            repo_root=project_root,
            workspace_root=workspace_root,
        )
        summary["systems"].append(system_summary)

    reference_key = ("A", "T_to_A")
    if reference_key in sasa_summaries_by_system_pair:
        reference_rows = sasa_summaries_by_system_pair[reference_key]
        for target_system in ["F670G", "I669G", "R668G"]:
            target_key = (target_system, "T_to_A")
            if target_key not in sasa_summaries_by_system_pair:
                continue
            delta_rows = build_sasa_delta_rows(
                reference_rows=reference_rows,
                target_rows=sasa_summaries_by_system_pair[target_key],
                reference_system="A",
                target_system=target_system,
            )
            comparison_name = f"A vs {target_system}"
            comparison_root = ensure_comparison_dir(results_root, comparison_name)
            delta_csv = write_sasa_delta_csv(
                delta_rows,
                output_csv=str(comparison_root / "delta_mean_sasa_A_minus_mutant.csv"),
            )
            delta_png = comparison_root / "delta_mean_sasa_A_minus_mutant.png"
            plotter.plot_delta_sasa(
                delta_rows,
                title=f"Δ mean SASA: A - {target_system}",
                save_path=str(delta_png),
            )
            summary["sasa_comparisons"].append(
                {
                    "comparison": comparison_name,
                    "reference_system": "A",
                    "target_system": target_system,
                    "pair_label": "T_to_A",
                    "outputs": {
                        "delta_mean_sasa_csv": str(delta_csv),
                        "delta_mean_sasa_png": str(delta_png),
                    },
                }
            )
    save_json(
        results_root / "summary.json",
        summary,
        repo_root=project_root,
        workspace_root=workspace_root,
    )
    print(f"Phospho tail coupling analysis completed: {results_root}")


if __name__ == "__main__":
    run()
