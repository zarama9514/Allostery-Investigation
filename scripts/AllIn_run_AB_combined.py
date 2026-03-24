from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array
from tqdm.auto import tqdm

from AllIn_DCCM import DCCMAnalyzer
from AllIn_geometry import HelicityAnalyzer, RMSDAnalyzer, RMSFAnalyzer
from AllIn_hbond import ProteinHydrogenBondAnalyzer
from AllIn_plot import (
    DCCMPlotter,
    HelicityPlotter,
    HydrogenBondPlotter,
    RMSDDifferencePlotter,
    RMSFDifferencePlotter,
    RMSProfilePlotter,
)
from AllIn_community import CommunityAnalyzer


@dataclass
class SystemRoles:
    mglu_segids: list[str]
    arrestin_segids: list[str]
    core_segids: list[str]


def progress_iter(iterable, desc: str, unit: str = "item", leave: bool = False):
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True, mininterval=1.0, leave=leave)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_results = Path(r"C:\Users\Daniil\IT_projects\Mika_project\results")
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-psf", default=str(default_results / "A" / "step5_input_protein.psf"))
    parser.add_argument("--b-psf", default=str(default_results / "B" / "B_step5_input_protein.psf"))
    parser.add_argument("--a-dir", default=str(default_results / "A"))
    parser.add_argument("--b-dir", default=str(default_results / "B"))
    parser.add_argument("--ab-results", default=str(project_root / "results" / "AB_results"))
    parser.add_argument("--mut-results", default=str(project_root / "mut_results"))
    parser.add_argument("--skip-first-n-frames", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--helicity-resids", default="665,666,667,668,669,670")
    return parser.parse_args()


def ensure_dirs(ab_results: Path, mut_results: Path) -> None:
    ab_results.mkdir(parents=True, exist_ok=True)
    mut_results.mkdir(parents=True, exist_ok=True)
    for name in [
        "01_system_A",
        "02_system_B",
        "03_A_vs_B",
        "04_arrestin_A_vs_B",
        "05_mGlu3_bound_vs_unbound_A",
    ]:
        (ab_results / name).mkdir(parents=True, exist_ok=True)


def run_id_from_name(filename: str) -> str | None:
    match = re.search(r"RUN(\d+)\.nowat\.dcd$", filename)
    if match is None:
        return None
    return f"RUN{int(match.group(1))}"


def collect_runs(directory: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for dcd in sorted(directory.glob("*.dcd")):
        run_id = run_id_from_name(dcd.name)
        if run_id is not None:
            out[run_id] = dcd
    return out


def detect_roles(psf: str, dcd: str) -> SystemRoles:
    u = mda.Universe(psf, dcd)
    protein = u.select_atoms("protein")
    mglu: list[str] = []
    arrestin: list[str] = []
    core: list[str] = []
    for segment in protein.segments:
        n_res = int(segment.residues.n_residues)
        segid = str(segment.segid).strip()
        if not segid:
            continue
        if n_res >= 250:
            core.append(segid)
        if n_res >= 700:
            mglu.append(segid)
        elif 300 <= n_res < 700:
            arrestin.append(segid)
    if len(mglu) < 2:
        raise ValueError(f"Cannot detect 2 mGlu3-like segments in {psf}")
    if len(arrestin) < 1:
        raise ValueError(f"Cannot detect arrestin-like segments in {psf}")
    return SystemRoles(
        mglu_segids=sorted(mglu),
        arrestin_segids=sorted(arrestin),
        core_segids=sorted(core),
    )


def build_selection(segids: Sequence[str], atom_part: str) -> str:
    return f"segid {' '.join(segids)} and {atom_part}"


def common_resids(psf: str, dcd: str, segids: Sequence[str]) -> list[int]:
    u = mda.Universe(psf, dcd)
    sets: list[set[int]] = []
    for segid in segids:
        atoms = u.select_atoms(f"segid {segid} and protein")
        sets.append(set(int(x) for x in atoms.residues.resids))
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def truncate_pair(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(x), len(a), len(b))
    return x[:n], a[:n], b[:n]


def align_rmsf(
    resids_a: np.ndarray,
    vals_a: np.ndarray,
    resids_b: np.ndarray,
    vals_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map_a = {int(r): float(v) for r, v in zip(resids_a, vals_a)}
    map_b = {int(r): float(v) for r, v in zip(resids_b, vals_b)}
    common = sorted(set(map_a).intersection(map_b))
    if not common:
        raise ValueError("No common residues for RMSF alignment")
    x = np.asarray(common, dtype=int)
    a = np.asarray([map_a[r] for r in common], dtype=float)
    b = np.asarray([map_b[r] for r in common], dtype=float)
    return x, a, b


def detect_bound_map(
    psf: str,
    dcd_files: Sequence[Path],
    mglu_segids: Sequence[str],
    arrestin_segids: Sequence[str],
    skip_first_n_frames: int,
    step: int = 10,
    progress_label: str = "bound-map",
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    score_sum: dict[str, dict[str, float]] = {a: {m: 0.0 for m in mglu_segids} for a in arrestin_segids}
    score_cnt: dict[str, dict[str, int]] = {a: {m: 0 for m in mglu_segids} for a in arrestin_segids}
    for dcd in progress_iter(dcd_files, desc=f"{progress_label}: runs", unit="run"):
        u = mda.Universe(psf, str(dcd))
        arr_atoms = {a: u.select_atoms(f"segid {a} and name CA") for a in arrestin_segids}
        mglu_atoms = {m: u.select_atoms(f"segid {m} and name CA") for m in mglu_segids}
        for ts in progress_iter(
            u.trajectory[skip_first_n_frames::step],
            desc=f"{progress_label}: {dcd.name} frames",
            unit="frame",
        ):
            _ = ts.frame
            for a in arrestin_segids:
                arr_pos = arr_atoms[a].positions
                for m in mglu_segids:
                    mglu_pos = mglu_atoms[m].positions
                    dmin = float(np.min(distance_array(arr_pos, mglu_pos)))
                    score_sum[a][m] += dmin
                    score_cnt[a][m] += 1
    mean_scores: dict[str, dict[str, float]] = {}
    mapping: dict[str, str] = {}
    for a in arrestin_segids:
        mean_scores[a] = {}
        for m in mglu_segids:
            cnt = score_cnt[a][m]
            if cnt == 0:
                mean_scores[a][m] = float("inf")
            else:
                mean_scores[a][m] = score_sum[a][m] / float(cnt)
        mapping[a] = min(mean_scores[a], key=mean_scores[a].get)
    return mapping, mean_scores


def aggregate_rmsd(
    psf: str,
    dcd_files: Sequence[Path],
    selection_backbone: str,
    skip_first_n_frames: int,
    progress_label: str = "RMSD",
) -> dict[str, np.ndarray]:
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    offset = 0.0
    for dcd in progress_iter(dcd_files, desc=f"{progress_label}: runs", unit="run"):
        out = RMSDAnalyzer(psf=psf, dcd=str(dcd)).calculate(
            selection=selection_backbone,
            skip_first_n_frames=skip_first_n_frames,
            step=1,
        )
        x = np.asarray(out["md_step"], dtype=float)
        y = np.asarray(out["rmsd"], dtype=float)
        if x.size == 0:
            continue
        x = (x - x[0]) + offset
        offset = float(x[-1] + 1.0)
        x_all.append(x)
        y_all.append(y)
    if not x_all:
        raise ValueError("No RMSD values collected")
    return {"md_step": np.concatenate(x_all), "rmsd": np.concatenate(y_all)}


def aggregate_rmsf(
    psf: str,
    dcd_files: Sequence[Path],
    selection_ca: str,
    selection_backbone: str,
    skip_first_n_frames: int,
    progress_label: str = "RMSF",
) -> tuple[np.ndarray, np.ndarray]:
    maps: list[dict[int, float]] = []
    for dcd in progress_iter(dcd_files, desc=f"{progress_label}: runs", unit="run"):
        resids, vals = RMSFAnalyzer(psf=psf, dcd=str(dcd)).calculate(
            target_selection=selection_ca,
            align_selection=selection_backbone,
            skip_first_n_frames=skip_first_n_frames,
            step=1,
        )
        maps.append({int(r): float(v) for r, v in zip(resids, vals)})
    common = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())
    if not common:
        raise ValueError("No common residues for aggregated RMSF")
    x = np.asarray(sorted(common), dtype=int)
    stacked = np.asarray([[m[int(r)] for r in x] for m in maps], dtype=float)
    return x, stacked.mean(axis=0)


def aggregate_helicity_mean(
    psf: str,
    dcd_files: Sequence[Path],
    segids: Sequence[str],
    amino_acids: Sequence[int],
    skip_first_n_frames: int,
    progress_label: str = "Helicity",
) -> tuple[np.ndarray, np.ndarray]:
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    offset = 0.0
    for dcd in progress_iter(dcd_files, desc=f"{progress_label}: runs", unit="run"):
        out = HelicityAnalyzer(psf=psf, dcd=str(dcd)).calculate(
            amino_acids=amino_acids,
            segids=tuple(segids),
            skip_first_n_frames=skip_first_n_frames,
            step=1,
        )
        if isinstance(out, str):
            raise ValueError(out)
        frame = np.asarray(out["frame"], dtype=float)
        keys = [k for k in out if k.startswith("helicity_")]
        series = np.asarray([np.asarray(out[k], dtype=float) for k in keys], dtype=float)
        mean_vals = series.mean(axis=0)
        frame = (frame - frame[0]) + offset
        offset = float(frame[-1] + 1.0) if frame.size else offset
        x_all.append(frame)
        y_all.append(mean_vals)
    if not x_all:
        raise ValueError("No helicity values collected")
    return np.concatenate(x_all), np.concatenate(y_all)


def aggregate_dccm_mean(
    psf: str,
    dcd_files: Sequence[Path],
    selection_ca: str,
    selection_backbone: str,
    skip_first_n_frames: int,
    progress_label: str = "DCCM",
) -> dict[str, object]:
    mats: list[np.ndarray] = []
    x_resids_ref: np.ndarray | None = None
    y_resids_ref: np.ndarray | None = None
    x_chain_ranges = None
    y_chain_ranges = None
    for dcd in progress_iter(dcd_files, desc=f"{progress_label}: runs", unit="run"):
        out = DCCMAnalyzer(psf1=psf, dcd1=str(dcd)).calculate(
            selection1=selection_ca,
            align_selection1=selection_backbone,
            skip_first_n_frames=skip_first_n_frames,
            step=10,
        )
        dccm = np.asarray(out["dccm"], dtype=float)
        x_resids = np.asarray(out["x_resids"], dtype=int)
        y_resids = np.asarray(out["y_resids"], dtype=int)
        if x_resids_ref is None:
            x_resids_ref = x_resids
            y_resids_ref = y_resids
            x_chain_ranges = out.get("x_chain_ranges")
            y_chain_ranges = out.get("y_chain_ranges")
        else:
            if not np.array_equal(x_resids_ref, x_resids) or not np.array_equal(y_resids_ref, y_resids):
                raise ValueError("DCCM residue axes mismatch between runs")
        mats.append(dccm)
    if not mats:
        raise ValueError("No DCCM matrices collected")
    return {
        "dccm": np.mean(np.asarray(mats, dtype=float), axis=0),
        "x_resids": x_resids_ref,
        "y_resids": y_resids_ref,
        "x_chain_ranges": x_chain_ranges,
        "y_chain_ranges": y_chain_ranges,
    }


def aggregate_hbond_contacts(
    psf: str,
    dcd_files: Sequence[Path],
    skip_first_n_frames: int,
    allowed_segids: Sequence[str],
    progress_label: str = "HBond",
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for dcd in progress_iter(dcd_files, desc=f"{progress_label}: runs", unit="run"):
        out = ProteinHydrogenBondAnalyzer(psf=psf, dcd=str(dcd)).calculate(
            donor_selection="protein",
            acceptor_selection="protein",
            inter_subunit_only=True,
            allowed_segids=allowed_segids,
            skip_first_n_frames=skip_first_n_frames,
            step=10,
        )
        for row in out["contacts"]:
            key = str(row["contact_atom_label"])
            if key not in merged:
                merged[key] = dict(row)
                merged[key]["_n_frames_weight"] = int(row["n_frames_analyzed"])
                merged[key]["_n_runs"] = 1
            else:
                acc = merged[key]
                acc["frames_observed"] = int(acc["frames_observed"]) + int(row["frames_observed"])
                acc["lifetime_ps"] = float(acc["lifetime_ps"]) + float(row["lifetime_ps"])
                acc["max_continuous_frames"] = max(int(acc["max_continuous_frames"]), int(row["max_continuous_frames"]))
                acc["max_continuous_ps"] = max(float(acc["max_continuous_ps"]), float(row["max_continuous_ps"]))
                acc["segments_count"] = int(acc["segments_count"]) + int(row["segments_count"])
                acc["first_observed_frame"] = min(int(acc["first_observed_frame"]), int(row["first_observed_frame"]))
                acc["last_observed_frame"] = max(int(acc["last_observed_frame"]), int(row["last_observed_frame"]))
                acc["mean_distance_angstrom"] = (
                    float(acc["mean_distance_angstrom"]) + float(row["mean_distance_angstrom"])
                ) / 2.0
                acc["mean_angle_degree"] = (
                    float(acc["mean_angle_degree"]) + float(row["mean_angle_degree"])
                ) / 2.0
                acc["_n_frames_weight"] = int(acc["_n_frames_weight"]) + int(row["n_frames_analyzed"])
                acc["_n_runs"] = int(acc["_n_runs"]) + 1
    contacts: list[dict[str, object]] = []
    for row in merged.values():
        row["n_frames_analyzed"] = int(row["_n_frames_weight"])
        denom = max(1, int(row["n_frames_analyzed"]))
        row["occupancy_fraction"] = float(row["frames_observed"]) / float(denom)
        row["occupancy_percent"] = float(row["occupancy_fraction"]) * 100.0
        row.pop("_n_frames_weight", None)
        row.pop("_n_runs", None)
        contacts.append(row)
    contacts.sort(key=lambda r: float(r["lifetime_ps"]), reverse=True)
    return contacts


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_resids_csv(text: str) -> list[int]:
    items = [x.strip() for x in text.split(",")]
    out: list[int] = []
    for item in items:
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("helicity residue list is empty")
    return out


def write_community_from_combined_dccm(
    psf: str,
    dcd_for_coordinates: str,
    selection: str,
    dccm: np.ndarray,
    output_pdb: str,
    threshold: float = 0.5,
    seed: int = 42,
    output_frame_index: int = 0,
) -> dict[str, int | str]:
    graph = CommunityAnalyzer._graph_from_dccm(np.asarray(dccm, dtype=float), threshold=threshold)
    community_map = CommunityAnalyzer._communities(graph, seed=seed)
    universe = mda.Universe(psf, dcd_for_coordinates)
    universe.trajectory[output_frame_index]
    try:
        _ = universe.atoms.tempfactors
    except Exception:
        universe.add_TopologyAttr("tempfactors")
    universe.atoms.tempfactors = 0.0
    ca_atoms = universe.select_atoms(selection)
    if ca_atoms.n_atoms != int(dccm.shape[0]):
        raise ValueError("Selected atom count does not match combined DCCM size")
    for idx, atom in enumerate(ca_atoms):
        atom.residue.atoms.tempfactors = float(int(community_map.get(idx, -1)))
    output_path = Path(output_pdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.atoms.write(str(output_path))
    return {
        "output_pdb": str(output_path),
        "n_communities": int(len(set(community_map.values()))),
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "dccm_source": "combined_runs_mean",
    }


def run() -> None:
    args = parse_args()
    a_psf = str(Path(args.a_psf))
    b_psf = str(Path(args.b_psf))
    a_dir = Path(args.a_dir)
    b_dir = Path(args.b_dir)
    ab_results = Path(args.ab_results)
    mut_results = Path(args.mut_results)
    skip = int(args.skip_first_n_frames)
    helicity_resids = parse_resids_csv(args.helicity_resids)
    ensure_dirs(ab_results=ab_results, mut_results=mut_results)
    a_runs_all = collect_runs(a_dir)
    b_runs_all = collect_runs(b_dir)
    common_ids = sorted(set(a_runs_all).intersection(b_runs_all))
    if not common_ids:
        raise ValueError("No matching RUN ids in A and B directories")
    if args.max_runs and args.max_runs > 0:
        common_ids = common_ids[: int(args.max_runs)]
    a_runs = [a_runs_all[rid] for rid in common_ids]
    b_runs = [b_runs_all[rid] for rid in common_ids]
    stage = tqdm(total=9, desc="AB pipeline", unit="stage", dynamic_ncols=True, mininterval=1.0)
    roles_a = detect_roles(a_psf, str(a_runs[0]))
    roles_b = detect_roles(b_psf, str(b_runs[0]))
    stage.update(1)
    bound_map_a, bound_scores_a = detect_bound_map(
        psf=a_psf,
        dcd_files=a_runs,
        mglu_segids=roles_a.mglu_segids,
        arrestin_segids=roles_a.arrestin_segids,
        skip_first_n_frames=skip,
        step=10,
        progress_label="A bound-map",
    )
    stage.update(1)
    bound_map_b, bound_scores_b = detect_bound_map(
        psf=b_psf,
        dcd_files=b_runs,
        mglu_segids=roles_b.mglu_segids,
        arrestin_segids=roles_b.arrestin_segids,
        skip_first_n_frames=skip,
        step=10,
        progress_label="B bound-map",
    )
    stage.update(1)
    arrestin_a_bound = sorted(bound_map_a.keys())[0]
    mglu_a_bound = bound_map_a[arrestin_a_bound]
    mglu_a_unbound = sorted([seg for seg in roles_a.mglu_segids if seg != mglu_a_bound])[0]
    arrestin_b_bound = sorted(bound_map_b.keys())
    plot_rms = RMSProfilePlotter(figsize=(10.0, 5.0), dpi=300)
    plot_rmsd_diff = RMSDDifferencePlotter(figsize=(10.0, 5.0), dpi=300)
    plot_rmsf_diff = RMSFDifferencePlotter(figsize=(10.0, 5.0), dpi=300)
    plot_helicity = HelicityPlotter(figsize=(10.0, 5.0), dpi=300)
    plot_dccm = DCCMPlotter(figsize=(10.0, 8.0), dpi=300)
    plot_hbond = HydrogenBondPlotter(figsize=(12.0, 6.0), dpi=300)

    a_sel_bb = build_selection(roles_a.mglu_segids, "backbone")
    b_sel_bb = build_selection(roles_b.mglu_segids, "backbone")
    a_sel_ca = build_selection(roles_a.mglu_segids, "name CA")
    b_sel_ca = build_selection(roles_b.mglu_segids, "name CA")

    a_rmsd = aggregate_rmsd(psf=a_psf, dcd_files=a_runs, selection_backbone=a_sel_bb, skip_first_n_frames=skip, progress_label="A mGlu RMSD")
    b_rmsd = aggregate_rmsd(psf=b_psf, dcd_files=b_runs, selection_backbone=b_sel_bb, skip_first_n_frames=skip, progress_label="B mGlu RMSD")
    x, a_y, b_y = truncate_pair(np.asarray(a_rmsd["md_step"]), np.asarray(a_rmsd["rmsd"]), np.asarray(b_rmsd["rmsd"]))

    out_ab = ab_results / "03_A_vs_B"
    plot_rms.plot_rmsd(x, a_y, title="A Combined RMSD vs Molecular Dynamics Step", save_path=str(out_ab / "A_rmsd_combined.png"))
    plot_rms.plot_rmsd(x, b_y, title="B Combined RMSD vs Molecular Dynamics Step", save_path=str(out_ab / "B_rmsd_combined.png"))
    plot_rmsd_diff.plot_difference(x, a_y, b_y, title="|Delta RMSD| A vs B (Combined)", save_path=str(out_ab / "delta_rmsd_A_vs_B_combined.png"))

    a_rmsf_resids, a_rmsf_vals = aggregate_rmsf(
        psf=a_psf,
        dcd_files=a_runs,
        selection_ca=a_sel_ca,
        selection_backbone=a_sel_bb,
        skip_first_n_frames=skip,
        progress_label="A mGlu RMSF",
    )
    b_rmsf_resids, b_rmsf_vals = aggregate_rmsf(
        psf=b_psf,
        dcd_files=b_runs,
        selection_ca=b_sel_ca,
        selection_backbone=b_sel_bb,
        skip_first_n_frames=skip,
        progress_label="B mGlu RMSF",
    )
    rx, ra, rb = align_rmsf(a_rmsf_resids, a_rmsf_vals, b_rmsf_resids, b_rmsf_vals)
    plot_rms.plot_rmsf(rx, ra, title="A Combined RMSF Profile", save_path=str(out_ab / "A_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx, rb, title="B Combined RMSF Profile", save_path=str(out_ab / "B_rmsf_combined.png"))
    plot_rmsf_diff.plot_difference(rx, ra, rb, title="|Delta RMSF| A vs B (Combined)", save_path=str(out_ab / "delta_rmsf_A_vs_B_combined.png"))

    a_hx, a_hy = aggregate_helicity_mean(a_psf, a_runs, roles_a.mglu_segids, helicity_resids, skip_first_n_frames=skip, progress_label="A mGlu helicity")
    b_hx, b_hy = aggregate_helicity_mean(b_psf, b_runs, roles_b.mglu_segids, helicity_resids, skip_first_n_frames=skip, progress_label="B mGlu helicity")
    hx, ha, hb = truncate_pair(a_hx, a_hy, b_hy)
    plot_helicity.plot(
        md_steps=hx,
        helicity_series={"A_mGlu3": ha, "B_mGlu3": hb},
        title="mGlu3 Helicity Comparison (A vs B, Combined)",
        save_path=str(out_ab / "helicity_mglu_A_vs_B_combined.png"),
    )

    a_dccm = aggregate_dccm_mean(a_psf, a_runs, a_sel_ca, a_sel_bb, skip_first_n_frames=skip, progress_label="A mGlu DCCM")
    b_dccm = aggregate_dccm_mean(b_psf, b_runs, b_sel_ca, b_sel_bb, skip_first_n_frames=skip, progress_label="B mGlu DCCM")
    plot_dccm.plot_from_community_output(a_dccm, title="A Combined DCCM (mGlu3)", save_path=str(out_ab / "A_dccm_mglu_combined.png"))
    plot_dccm.plot_from_community_output(b_dccm, title="B Combined DCCM (mGlu3)", save_path=str(out_ab / "B_dccm_mglu_combined.png"))
    plot_dccm.plot_difference_from_community_outputs(
        community_output_1=a_dccm,
        community_output_2=b_dccm,
        title="Normalized DCCM Difference (A - B) / 2, mGlu3 Combined",
        save_path=str(out_ab / "delta_dccm_mglu_A_vs_B_combined.png"),
    )
    stage.update(1)

    out_ar = ab_results / "04_arrestin_A_vs_B"
    a_arr_sel_bb = build_selection([arrestin_a_bound], "backbone")
    a_arr_sel_ca = build_selection([arrestin_a_bound], "name CA")
    a_arr_rmsd = aggregate_rmsd(a_psf, a_runs, a_arr_sel_bb, skip, progress_label="A arrestin RMSD")
    a_arr_rmsf_resids, a_arr_rmsf_vals = aggregate_rmsf(a_psf, a_runs, a_arr_sel_ca, a_arr_sel_bb, skip, progress_label="A arrestin RMSF")
    b_arr_rmsd_series: list[np.ndarray] = []
    b_arr_rmsf_maps: list[tuple[np.ndarray, np.ndarray]] = []
    for segid in progress_iter(arrestin_b_bound, desc="B arrestin chains", unit="chain"):
        b_arr_sel_bb = build_selection([segid], "backbone")
        b_arr_sel_ca = build_selection([segid], "name CA")
        b_arr_rmsd_series.append(
            aggregate_rmsd(
                b_psf,
                b_runs,
                b_arr_sel_bb,
                skip,
                progress_label=f"B arrestin {segid} RMSD",
            )["rmsd"]
        )
        b_arr_rmsf_maps.append(
            aggregate_rmsf(
                b_psf,
                b_runs,
                b_arr_sel_ca,
                b_arr_sel_bb,
                skip,
                progress_label=f"B arrestin {segid} RMSF",
            )
        )
    min_len_b = min(len(x) for x in b_arr_rmsd_series)
    b_arr_rmsd_mean = np.mean(np.asarray([x[:min_len_b] for x in b_arr_rmsd_series], dtype=float), axis=0)
    x_ar, a_ar, b_ar = truncate_pair(
        np.asarray(a_arr_rmsd["md_step"]),
        np.asarray(a_arr_rmsd["rmsd"]),
        b_arr_rmsd_mean,
    )
    plot_rms.plot_rmsd(x_ar, a_ar, title="A Bound Arrestin RMSD (Combined)", save_path=str(out_ar / "A_bound_arrestin_rmsd_combined.png"))
    plot_rms.plot_rmsd(x_ar, b_ar, title="B Bound Arrestin RMSD Mean (Combined)", save_path=str(out_ar / "B_bound_arrestin_rmsd_mean_combined.png"))
    plot_rmsd_diff.plot_difference(x_ar, a_ar, b_ar, title="|Delta RMSD| Arrestin A vs B (Combined)", save_path=str(out_ar / "delta_rmsd_arrestin_A_vs_B_combined.png"))

    b_arr_common_resids = b_arr_rmsf_maps[0][0]
    b_arr_vals_stack = []
    for resids, vals in b_arr_rmsf_maps:
        x_tmp, _, b_tmp = align_rmsf(b_arr_common_resids, b_arr_rmsf_maps[0][1], resids, vals)
        b_arr_common_resids = x_tmp
        b_arr_vals_stack.append(b_tmp)
    b_arr_rmsf_mean = np.mean(np.asarray(b_arr_vals_stack, dtype=float), axis=0)
    rx_ar, ra_ar, rb_ar = align_rmsf(a_arr_rmsf_resids, a_arr_rmsf_vals, b_arr_common_resids, b_arr_rmsf_mean)
    plot_rms.plot_rmsf(rx_ar, ra_ar, title="A Bound Arrestin RMSF (Combined)", save_path=str(out_ar / "A_bound_arrestin_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx_ar, rb_ar, title="B Bound Arrestin RMSF Mean (Combined)", save_path=str(out_ar / "B_bound_arrestin_rmsf_mean_combined.png"))
    plot_rmsf_diff.plot_difference(rx_ar, ra_ar, rb_ar, title="|Delta RMSF| Arrestin A vs B (Combined)", save_path=str(out_ar / "delta_rmsf_arrestin_A_vs_B_combined.png"))

    # Helicity is intentionally not computed for arrestin here.
    # The target alpha-helical loop is defined for mGlu3 region 665-670.
    stage.update(1)

    out_mglu = ab_results / "05_mGlu3_bound_vs_unbound_A"
    a_bound_sel_bb = build_selection([mglu_a_bound], "backbone")
    a_bound_sel_ca = build_selection([mglu_a_bound], "name CA")
    a_unbound_sel_bb = build_selection([mglu_a_unbound], "backbone")
    a_unbound_sel_ca = build_selection([mglu_a_unbound], "name CA")
    a_bound_rmsd = aggregate_rmsd(a_psf, a_runs, a_bound_sel_bb, skip, progress_label="A bound mGlu RMSD")
    a_unbound_rmsd = aggregate_rmsd(a_psf, a_runs, a_unbound_sel_bb, skip, progress_label="A unbound mGlu RMSD")
    x_m, y_bound, y_unbound = truncate_pair(
        np.asarray(a_bound_rmsd["md_step"]),
        np.asarray(a_bound_rmsd["rmsd"]),
        np.asarray(a_unbound_rmsd["rmsd"]),
    )
    plot_rms.plot_rmsd(x_m, y_bound, title=f"A Bound mGlu3 ({mglu_a_bound}) RMSD Combined", save_path=str(out_mglu / "A_bound_mglu_rmsd_combined.png"))
    plot_rms.plot_rmsd(x_m, y_unbound, title=f"A Unbound mGlu3 ({mglu_a_unbound}) RMSD Combined", save_path=str(out_mglu / "A_unbound_mglu_rmsd_combined.png"))
    plot_rmsd_diff.plot_difference(x_m, y_bound, y_unbound, title="|Delta RMSD| Bound vs Unbound mGlu3 in A (Combined)", save_path=str(out_mglu / "delta_rmsd_bound_vs_unbound_mglu_A_combined.png"))

    rb_resids, rb_vals = aggregate_rmsf(a_psf, a_runs, a_bound_sel_ca, a_bound_sel_bb, skip, progress_label="A bound mGlu RMSF")
    ru_resids, ru_vals = aggregate_rmsf(a_psf, a_runs, a_unbound_sel_ca, a_unbound_sel_bb, skip, progress_label="A unbound mGlu RMSF")
    rx_m, vb, vu = align_rmsf(rb_resids, rb_vals, ru_resids, ru_vals)
    plot_rms.plot_rmsf(rx_m, vb, title=f"A Bound mGlu3 ({mglu_a_bound}) RMSF Combined", save_path=str(out_mglu / "A_bound_mglu_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx_m, vu, title=f"A Unbound mGlu3 ({mglu_a_unbound}) RMSF Combined", save_path=str(out_mglu / "A_unbound_mglu_rmsf_combined.png"))
    plot_rmsf_diff.plot_difference(rx_m, vb, vu, title="|Delta RMSF| Bound vs Unbound mGlu3 in A (Combined)", save_path=str(out_mglu / "delta_rmsf_bound_vs_unbound_mglu_A_combined.png"))

    hx_bound, hy_bound = aggregate_helicity_mean(a_psf, a_runs, [mglu_a_bound], helicity_resids, skip, progress_label="A bound mGlu helicity")
    _, hy_unbound = aggregate_helicity_mean(a_psf, a_runs, [mglu_a_unbound], helicity_resids, skip, progress_label="A unbound mGlu helicity")
    hx_m, hy_bnd, hy_unb = truncate_pair(hx_bound, hy_bound, hy_unbound)
    plot_helicity.plot(
        md_steps=hx_m,
        helicity_series={f"bound_{mglu_a_bound}": hy_bnd, f"unbound_{mglu_a_unbound}": hy_unb},
        title="mGlu3 Helicity in A: Bound vs Unbound (Combined)",
        save_path=str(out_mglu / "helicity_bound_vs_unbound_mglu_A_combined.png"),
    )
    stage.update(1)

    hbond_a = aggregate_hbond_contacts(a_psf, a_runs, skip, roles_a.core_segids, progress_label="A HBond")
    hbond_b = aggregate_hbond_contacts(b_psf, b_runs, skip, roles_b.core_segids, progress_label="B HBond")
    hbond_a_csv = ProteinHydrogenBondAnalyzer.write_csv(hbond_a, str(ab_results / "01_system_A" / "hbond_contacts_combined.csv"))
    hbond_b_csv = ProteinHydrogenBondAnalyzer.write_csv(hbond_b, str(ab_results / "02_system_B" / "hbond_contacts_combined.csv"))
    plot_hbond.plot_top_contacts(csv_path=str(hbond_a_csv), top_n=15, title="A Combined Top 15 H-bond Contacts", save_path=str(ab_results / "01_system_A" / "hbond_top15_combined.png"))
    plot_hbond.plot_top_contacts(csv_path=str(hbond_b_csv), top_n=15, title="B Combined Top 15 H-bond Contacts", save_path=str(ab_results / "02_system_B" / "hbond_top15_combined.png"))
    stage.update(1)

    community_a_info = write_community_from_combined_dccm(
        psf=a_psf,
        dcd_for_coordinates=str(a_runs[0]),
        selection=a_sel_ca,
        dccm=np.asarray(a_dccm["dccm"], dtype=float),
        output_pdb=str(ab_results / "01_system_A" / "community_A_combined_frame0.pdb"),
    )
    community_b_info = write_community_from_combined_dccm(
        psf=b_psf,
        dcd_for_coordinates=str(b_runs[0]),
        selection=b_sel_ca,
        dccm=np.asarray(b_dccm["dccm"], dtype=float),
        output_pdb=str(ab_results / "02_system_B" / "community_B_combined_frame0.pdb"),
    )
    stage.update(1)

    summary = {
        "runs_used": common_ids,
        "skip_first_n_frames_per_run": skip,
        "roles_A": {
            "mglu_segids": roles_a.mglu_segids,
            "arrestin_segids": roles_a.arrestin_segids,
            "core_segids": roles_a.core_segids,
        },
        "roles_B": {
            "mglu_segids": roles_b.mglu_segids,
            "arrestin_segids": roles_b.arrestin_segids,
            "core_segids": roles_b.core_segids,
        },
        "bound_map_A": bound_map_a,
        "bound_map_B": bound_map_b,
        "bound_scores_A": bound_scores_a,
        "bound_scores_B": bound_scores_b,
        "A_bound_arrestin": arrestin_a_bound,
        "A_bound_mglu": mglu_a_bound,
        "A_unbound_mglu": mglu_a_unbound,
        "B_bound_arrestins": arrestin_b_bound,
        "helicity_target_resids": helicity_resids,
        "community_A": community_a_info,
        "community_B": community_b_info,
    }
    save_json(ab_results / "summary_combined.json", summary)
    stage.update(1)
    stage.close()
    print(f"AB combined analysis completed: {ab_results}")
    print(f"Runs used: {', '.join(common_ids)}")


if __name__ == "__main__":
    run()
