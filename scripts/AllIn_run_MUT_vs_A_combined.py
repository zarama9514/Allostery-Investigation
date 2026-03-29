from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import MDAnalysis as mda
import matplotlib.pyplot as plt
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


@dataclass
class MutantSystem:
    name: str
    psf: Path
    run_map: dict[str, Path]


def progress_iter(iterable, desc: str, unit: str = "item", leave: bool = False):
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True, mininterval=1.0, leave=leave)


def default_workspace_results(project_root: Path) -> Path:
    return project_root.parent / "Mika_project" / "results"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_results = default_workspace_results(project_root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-psf", default=str(default_results / "A" / "step5_input_protein.psf"))
    parser.add_argument("--a-dir", default=str(default_results / "A"))
    parser.add_argument("--mut-root", default=str(default_results))
    parser.add_argument("--mutants", default="")
    parser.add_argument("--mut-results", default=str(project_root / "results" / "mut_results"))
    parser.add_argument("--skip-first-n-frames", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-mutants", type=int, default=0)
    parser.add_argument("--helicity-resids", default="665,666,667,668,669,670")
    return parser.parse_args()


def ensure_dirs(mut_results: Path) -> None:
    mut_results.mkdir(parents=True, exist_ok=True)


def parse_csv_names(text: str) -> list[str]:
    if not text.strip():
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def ensure_mutant_dirs(mut_results: Path, mutant_name: str) -> dict[str, Path]:
    root = mut_results / mutant_name
    dirs = {
        "root": root,
        "a_vs_mut": root / "01_A_vs_mutant",
        "a_bound": root / "02_mGlu_bound_vs_unbound_A",
        "mut_bound": root / "03_mGlu_bound_vs_unbound_mutant",
        "bound_cmp": root / "04_bound_mGlu_A_vs_mutant",
        "hbond": root / "05_hbond",
        "community": root / "06_community",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def collect_mutants(mut_root: Path, explicit: Sequence[str], max_mutants: int) -> list[MutantSystem]:
    out: list[MutantSystem] = []
    reserved = {"A", "B"}
    candidates: list[Path] = []
    if explicit:
        candidates.extend([mut_root / name for name in explicit])
    else:
        for child in sorted(mut_root.iterdir()):
            if child.is_dir() and child.name not in reserved:
                candidates.append(child)
    for folder in candidates:
        if not folder.exists():
            continue
        name = folder.name
        psf_candidates = [folder / f"{name}_step5_input_protein.psf"]
        psf_candidates.extend(sorted(folder.glob("*_step5_input_protein.psf")))
        psf_path = next((p for p in psf_candidates if p.exists()), None)
        if psf_path is None:
            continue
        run_map = collect_runs(folder)
        if not run_map:
            continue
        out.append(MutantSystem(name=name, psf=psf_path, run_map=run_map))
    out.sort(key=lambda x: x.name)
    if max_mutants and max_mutants > 0:
        out = out[: int(max_mutants)]
    return out


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


def serialize_summary_path(path: Path, repo_root: Path | None = None, workspace_root: Path | None = None) -> str:
    resolved = path.resolve()
    bases = [repo_root.resolve()] if repo_root is not None else []
    if workspace_root is not None:
        bases.append(workspace_root.resolve())
    for base in bases:
        try:
            relative = resolved.relative_to(base).as_posix()
            if relative.startswith("AB_results/"):
                return f"results/{relative}"
            return relative
        except ValueError:
            continue
    return path.name


def normalize_summary_payload(
    payload: object,
    repo_root: Path | None = None,
    workspace_root: Path | None = None,
) -> object:
    if isinstance(payload, Mapping):
        return {
            key: normalize_summary_payload(value, repo_root=repo_root, workspace_root=workspace_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [normalize_summary_payload(item, repo_root=repo_root, workspace_root=workspace_root) for item in payload]
    if isinstance(payload, tuple):
        return [normalize_summary_payload(item, repo_root=repo_root, workspace_root=workspace_root) for item in payload]
    if isinstance(payload, Path):
        return serialize_summary_path(payload, repo_root=repo_root, workspace_root=workspace_root)
    if isinstance(payload, str):
        looks_like_path = "\\" in payload or "/" in payload or (len(payload) > 1 and payload[1] == ":")
        if looks_like_path:
            path = Path(payload)
            if path.is_absolute():
                return serialize_summary_path(path, repo_root=repo_root, workspace_root=workspace_root)
            return payload.replace("\\", "/")
    return payload


def save_json(
    path: Path,
    payload: Mapping[str, object],
    repo_root: Path | None = None,
    workspace_root: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            normalize_summary_payload(payload, repo_root=repo_root, workspace_root=workspace_root),
            handle,
            ensure_ascii=False,
            indent=2,
        )


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


def close_figs() -> None:
    plt.close("all")


def run_single_mutant(
    *,
    mutant: MutantSystem,
    a_psf: str,
    a_runs: list[Path],
    mut_runs: list[Path],
    helicity_resids: Sequence[int],
    skip: int,
    mut_results: Path,
    repo_root: Path,
    workspace_root: Path,
) -> dict[str, object]:
    roles_a = detect_roles(a_psf, str(a_runs[0]))
    roles_m = detect_roles(str(mutant.psf), str(mut_runs[0]))
    bound_map_a, bound_scores_a = detect_bound_map(
        psf=a_psf,
        dcd_files=a_runs,
        mglu_segids=roles_a.mglu_segids,
        arrestin_segids=roles_a.arrestin_segids,
        skip_first_n_frames=skip,
        step=10,
        progress_label=f"{mutant.name} A bound-map",
    )
    bound_map_m, bound_scores_m = detect_bound_map(
        psf=str(mutant.psf),
        dcd_files=mut_runs,
        mglu_segids=roles_m.mglu_segids,
        arrestin_segids=roles_m.arrestin_segids,
        skip_first_n_frames=skip,
        step=10,
        progress_label=f"{mutant.name} MUT bound-map",
    )
    arrestin_a = sorted(bound_map_a.keys())[0]
    mglu_a_bound = bound_map_a[arrestin_a]
    mglu_a_unbound = sorted([seg for seg in roles_a.mglu_segids if seg != mglu_a_bound])[0]
    arrestin_m = sorted(bound_map_m.keys())[0]
    mglu_m_bound = bound_map_m[arrestin_m]
    mglu_m_unbound = sorted([seg for seg in roles_m.mglu_segids if seg != mglu_m_bound])[0]

    dirs = ensure_mutant_dirs(mut_results=mut_results, mutant_name=mutant.name)
    plot_rms = RMSProfilePlotter(figsize=(10.0, 5.0), dpi=300)
    plot_rmsd_diff = RMSDDifferencePlotter(figsize=(10.0, 5.0), dpi=300)
    plot_rmsf_diff = RMSFDifferencePlotter(figsize=(10.0, 5.0), dpi=300)
    plot_helicity = HelicityPlotter(figsize=(10.0, 5.0), dpi=300)
    plot_dccm = DCCMPlotter(figsize=(10.0, 8.0), dpi=300)
    plot_hbond = HydrogenBondPlotter(figsize=(12.0, 6.0), dpi=300)
    stage = tqdm(total=7, desc=f"{mutant.name} pipeline", unit="stage", dynamic_ncols=True, mininterval=1.0)

    a_sel_bb = build_selection(roles_a.mglu_segids, "backbone")
    m_sel_bb = build_selection(roles_m.mglu_segids, "backbone")
    a_sel_ca = build_selection(roles_a.mglu_segids, "name CA")
    m_sel_ca = build_selection(roles_m.mglu_segids, "name CA")
    a_rmsd = aggregate_rmsd(psf=a_psf, dcd_files=a_runs, selection_backbone=a_sel_bb, skip_first_n_frames=skip, progress_label=f"{mutant.name} A mGlu RMSD")
    m_rmsd = aggregate_rmsd(psf=str(mutant.psf), dcd_files=mut_runs, selection_backbone=m_sel_bb, skip_first_n_frames=skip, progress_label=f"{mutant.name} MUT mGlu RMSD")
    x, a_y, m_y = truncate_pair(np.asarray(a_rmsd["md_step"]), np.asarray(a_rmsd["rmsd"]), np.asarray(m_rmsd["rmsd"]))
    plot_rms.plot_rmsd(x, a_y, title=f"A Combined RMSD vs Molecular Dynamics Step ({mutant.name} comparison)", save_path=str(dirs["a_vs_mut"] / "A_rmsd_combined.png"))
    plot_rms.plot_rmsd(x, m_y, title=f"{mutant.name} Combined RMSD vs Molecular Dynamics Step", save_path=str(dirs["a_vs_mut"] / f"{mutant.name}_rmsd_combined.png"))
    plot_rmsd_diff.plot_difference(x, a_y, m_y, title=f"|Delta RMSD| A vs {mutant.name} (Combined)", save_path=str(dirs["a_vs_mut"] / f"delta_rmsd_A_vs_{mutant.name}_combined.png"))
    a_rmsf_resids, a_rmsf_vals = aggregate_rmsf(
        psf=a_psf,
        dcd_files=a_runs,
        selection_ca=a_sel_ca,
        selection_backbone=a_sel_bb,
        skip_first_n_frames=skip,
        progress_label=f"{mutant.name} A mGlu RMSF",
    )
    m_rmsf_resids, m_rmsf_vals = aggregate_rmsf(
        psf=str(mutant.psf),
        dcd_files=mut_runs,
        selection_ca=m_sel_ca,
        selection_backbone=m_sel_bb,
        skip_first_n_frames=skip,
        progress_label=f"{mutant.name} MUT mGlu RMSF",
    )
    rx, ra, rm = align_rmsf(a_rmsf_resids, a_rmsf_vals, m_rmsf_resids, m_rmsf_vals)
    plot_rms.plot_rmsf(rx, ra, title=f"A Combined RMSF Profile ({mutant.name} comparison)", save_path=str(dirs["a_vs_mut"] / "A_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx, rm, title=f"{mutant.name} Combined RMSF Profile", save_path=str(dirs["a_vs_mut"] / f"{mutant.name}_rmsf_combined.png"))
    plot_rmsf_diff.plot_difference(rx, ra, rm, title=f"|Delta RMSF| A vs {mutant.name} (Combined)", save_path=str(dirs["a_vs_mut"] / f"delta_rmsf_A_vs_{mutant.name}_combined.png"))
    a_hx, a_hy = aggregate_helicity_mean(a_psf, a_runs, roles_a.mglu_segids, helicity_resids, skip_first_n_frames=skip, progress_label=f"{mutant.name} A mGlu helicity")
    m_hx, m_hy = aggregate_helicity_mean(str(mutant.psf), mut_runs, roles_m.mglu_segids, helicity_resids, skip_first_n_frames=skip, progress_label=f"{mutant.name} MUT mGlu helicity")
    hx, ha, hm = truncate_pair(a_hx, a_hy, m_hy)
    plot_helicity.plot(
        md_steps=hx,
        helicity_series={"A_mGlu3": ha, f"{mutant.name}_mGlu3": hm},
        title=f"mGlu3 Helicity Comparison (A vs {mutant.name}, Combined)",
        save_path=str(dirs["a_vs_mut"] / f"helicity_A_vs_{mutant.name}_combined.png"),
    )
    a_dccm = aggregate_dccm_mean(a_psf, a_runs, a_sel_ca, a_sel_bb, skip_first_n_frames=skip, progress_label=f"{mutant.name} A mGlu DCCM")
    m_dccm = aggregate_dccm_mean(str(mutant.psf), mut_runs, m_sel_ca, m_sel_bb, skip_first_n_frames=skip, progress_label=f"{mutant.name} MUT mGlu DCCM")
    plot_dccm.plot_from_community_output(a_dccm, title=f"A Combined DCCM (mGlu3, {mutant.name} comparison)", save_path=str(dirs["a_vs_mut"] / "A_dccm_mglu_combined.png"))
    plot_dccm.plot_from_community_output(m_dccm, title=f"{mutant.name} Combined DCCM (mGlu3)", save_path=str(dirs["a_vs_mut"] / f"{mutant.name}_dccm_mglu_combined.png"))
    plot_dccm.plot_difference_from_community_outputs(
        community_output_1=a_dccm,
        community_output_2=m_dccm,
        title=f"Normalized DCCM Difference (A - {mutant.name}) / 2, mGlu3 Combined",
        save_path=str(dirs["a_vs_mut"] / f"delta_dccm_mglu_A_vs_{mutant.name}_combined.png"),
    )
    close_figs()
    stage.update(1)

    a_bound_sel_bb = build_selection([mglu_a_bound], "backbone")
    a_bound_sel_ca = build_selection([mglu_a_bound], "name CA")
    a_unbound_sel_bb = build_selection([mglu_a_unbound], "backbone")
    a_unbound_sel_ca = build_selection([mglu_a_unbound], "name CA")
    a_bound_rmsd = aggregate_rmsd(a_psf, a_runs, a_bound_sel_bb, skip, progress_label=f"{mutant.name} A bound mGlu RMSD")
    a_unbound_rmsd = aggregate_rmsd(a_psf, a_runs, a_unbound_sel_bb, skip, progress_label=f"{mutant.name} A unbound mGlu RMSD")
    x_a, y_bound_a, y_unbound_a = truncate_pair(np.asarray(a_bound_rmsd["md_step"]), np.asarray(a_bound_rmsd["rmsd"]), np.asarray(a_unbound_rmsd["rmsd"]))
    plot_rms.plot_rmsd(x_a, y_bound_a, title=f"A Bound mGlu3 ({mglu_a_bound}) RMSD Combined", save_path=str(dirs["a_bound"] / "A_bound_mglu_rmsd_combined.png"))
    plot_rms.plot_rmsd(x_a, y_unbound_a, title=f"A Unbound mGlu3 ({mglu_a_unbound}) RMSD Combined", save_path=str(dirs["a_bound"] / "A_unbound_mglu_rmsd_combined.png"))
    plot_rmsd_diff.plot_difference(x_a, y_bound_a, y_unbound_a, title="|Delta RMSD| Bound vs Unbound mGlu3 in A (Combined)", save_path=str(dirs["a_bound"] / "delta_rmsd_bound_vs_unbound_mglu_A_combined.png"))
    rb_a_resids, rb_a_vals = aggregate_rmsf(a_psf, a_runs, a_bound_sel_ca, a_bound_sel_bb, skip, progress_label=f"{mutant.name} A bound mGlu RMSF")
    ru_a_resids, ru_a_vals = aggregate_rmsf(a_psf, a_runs, a_unbound_sel_ca, a_unbound_sel_bb, skip, progress_label=f"{mutant.name} A unbound mGlu RMSF")
    rx_a, vb_a, vu_a = align_rmsf(rb_a_resids, rb_a_vals, ru_a_resids, ru_a_vals)
    plot_rms.plot_rmsf(rx_a, vb_a, title=f"A Bound mGlu3 ({mglu_a_bound}) RMSF Combined", save_path=str(dirs["a_bound"] / "A_bound_mglu_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx_a, vu_a, title=f"A Unbound mGlu3 ({mglu_a_unbound}) RMSF Combined", save_path=str(dirs["a_bound"] / "A_unbound_mglu_rmsf_combined.png"))
    plot_rmsf_diff.plot_difference(rx_a, vb_a, vu_a, title="|Delta RMSF| Bound vs Unbound mGlu3 in A (Combined)", save_path=str(dirs["a_bound"] / "delta_rmsf_bound_vs_unbound_mglu_A_combined.png"))
    hx_ab, hy_ab = aggregate_helicity_mean(a_psf, a_runs, [mglu_a_bound], helicity_resids, skip, progress_label=f"{mutant.name} A bound mGlu helicity")
    _, hy_au = aggregate_helicity_mean(a_psf, a_runs, [mglu_a_unbound], helicity_resids, skip, progress_label=f"{mutant.name} A unbound mGlu helicity")
    hx_a2, hy_a_bnd, hy_a_unb = truncate_pair(hx_ab, hy_ab, hy_au)
    plot_helicity.plot(
        md_steps=hx_a2,
        helicity_series={f"bound_{mglu_a_bound}": hy_a_bnd, f"unbound_{mglu_a_unbound}": hy_a_unb},
        title="mGlu3 Helicity in A: Bound vs Unbound (Combined)",
        save_path=str(dirs["a_bound"] / "helicity_bound_vs_unbound_mglu_A_combined.png"),
    )
    close_figs()
    stage.update(1)

    m_bound_sel_bb = build_selection([mglu_m_bound], "backbone")
    m_bound_sel_ca = build_selection([mglu_m_bound], "name CA")
    m_unbound_sel_bb = build_selection([mglu_m_unbound], "backbone")
    m_unbound_sel_ca = build_selection([mglu_m_unbound], "name CA")
    m_bound_rmsd = aggregate_rmsd(str(mutant.psf), mut_runs, m_bound_sel_bb, skip, progress_label=f"{mutant.name} bound mGlu RMSD")
    m_unbound_rmsd = aggregate_rmsd(str(mutant.psf), mut_runs, m_unbound_sel_bb, skip, progress_label=f"{mutant.name} unbound mGlu RMSD")
    x_m2, y_bound_m, y_unbound_m = truncate_pair(np.asarray(m_bound_rmsd["md_step"]), np.asarray(m_bound_rmsd["rmsd"]), np.asarray(m_unbound_rmsd["rmsd"]))
    plot_rms.plot_rmsd(x_m2, y_bound_m, title=f"{mutant.name} Bound mGlu3 ({mglu_m_bound}) RMSD Combined", save_path=str(dirs["mut_bound"] / f"{mutant.name}_bound_mglu_rmsd_combined.png"))
    plot_rms.plot_rmsd(x_m2, y_unbound_m, title=f"{mutant.name} Unbound mGlu3 ({mglu_m_unbound}) RMSD Combined", save_path=str(dirs["mut_bound"] / f"{mutant.name}_unbound_mglu_rmsd_combined.png"))
    plot_rmsd_diff.plot_difference(x_m2, y_bound_m, y_unbound_m, title=f"|Delta RMSD| Bound vs Unbound mGlu3 in {mutant.name} (Combined)", save_path=str(dirs["mut_bound"] / f"delta_rmsd_bound_vs_unbound_mglu_{mutant.name}_combined.png"))
    rb_m_resids, rb_m_vals = aggregate_rmsf(str(mutant.psf), mut_runs, m_bound_sel_ca, m_bound_sel_bb, skip, progress_label=f"{mutant.name} bound mGlu RMSF")
    ru_m_resids, ru_m_vals = aggregate_rmsf(str(mutant.psf), mut_runs, m_unbound_sel_ca, m_unbound_sel_bb, skip, progress_label=f"{mutant.name} unbound mGlu RMSF")
    rx_m2, vb_m, vu_m = align_rmsf(rb_m_resids, rb_m_vals, ru_m_resids, ru_m_vals)
    plot_rms.plot_rmsf(rx_m2, vb_m, title=f"{mutant.name} Bound mGlu3 ({mglu_m_bound}) RMSF Combined", save_path=str(dirs["mut_bound"] / f"{mutant.name}_bound_mglu_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx_m2, vu_m, title=f"{mutant.name} Unbound mGlu3 ({mglu_m_unbound}) RMSF Combined", save_path=str(dirs["mut_bound"] / f"{mutant.name}_unbound_mglu_rmsf_combined.png"))
    plot_rmsf_diff.plot_difference(rx_m2, vb_m, vu_m, title=f"|Delta RMSF| Bound vs Unbound mGlu3 in {mutant.name} (Combined)", save_path=str(dirs["mut_bound"] / f"delta_rmsf_bound_vs_unbound_mglu_{mutant.name}_combined.png"))
    hx_mb, hy_mb = aggregate_helicity_mean(str(mutant.psf), mut_runs, [mglu_m_bound], helicity_resids, skip, progress_label=f"{mutant.name} bound mGlu helicity")
    _, hy_mu = aggregate_helicity_mean(str(mutant.psf), mut_runs, [mglu_m_unbound], helicity_resids, skip, progress_label=f"{mutant.name} unbound mGlu helicity")
    hx_m3, hy_m_bnd, hy_m_unb = truncate_pair(hx_mb, hy_mb, hy_mu)
    plot_helicity.plot(
        md_steps=hx_m3,
        helicity_series={f"bound_{mglu_m_bound}": hy_m_bnd, f"unbound_{mglu_m_unbound}": hy_m_unb},
        title=f"mGlu3 Helicity in {mutant.name}: Bound vs Unbound (Combined)",
        save_path=str(dirs["mut_bound"] / f"helicity_bound_vs_unbound_mglu_{mutant.name}_combined.png"),
    )
    close_figs()
    stage.update(1)

    a_bound_rmsd_for_cmp = aggregate_rmsd(a_psf, a_runs, a_bound_sel_bb, skip, progress_label=f"{mutant.name} A bound RMSD for cmp")
    m_bound_rmsd_for_cmp = aggregate_rmsd(str(mutant.psf), mut_runs, m_bound_sel_bb, skip, progress_label=f"{mutant.name} MUT bound RMSD for cmp")
    x_cmp, y_a_cmp, y_m_cmp = truncate_pair(
        np.asarray(a_bound_rmsd_for_cmp["md_step"]),
        np.asarray(a_bound_rmsd_for_cmp["rmsd"]),
        np.asarray(m_bound_rmsd_for_cmp["rmsd"]),
    )
    plot_rms.plot_rmsd(x_cmp, y_a_cmp, title=f"A bound mGlu3 ({mglu_a_bound}) RMSD Combined", save_path=str(dirs["bound_cmp"] / "A_bound_mglu_rmsd_combined.png"))
    plot_rms.plot_rmsd(x_cmp, y_m_cmp, title=f"{mutant.name} bound mGlu3 ({mglu_m_bound}) RMSD Combined", save_path=str(dirs["bound_cmp"] / f"{mutant.name}_bound_mglu_rmsd_combined.png"))
    plot_rmsd_diff.plot_difference(x_cmp, y_a_cmp, y_m_cmp, title=f"|Delta RMSD| bound mGlu3 A vs {mutant.name} (Combined)", save_path=str(dirs["bound_cmp"] / f"delta_rmsd_bound_mglu_A_vs_{mutant.name}_combined.png"))
    rb_ac_resids, rb_ac_vals = aggregate_rmsf(a_psf, a_runs, a_bound_sel_ca, a_bound_sel_bb, skip, progress_label=f"{mutant.name} A bound RMSF for cmp")
    rb_mc_resids, rb_mc_vals = aggregate_rmsf(str(mutant.psf), mut_runs, m_bound_sel_ca, m_bound_sel_bb, skip, progress_label=f"{mutant.name} MUT bound RMSF for cmp")
    rx_cmp, v_a_cmp, v_m_cmp = align_rmsf(rb_ac_resids, rb_ac_vals, rb_mc_resids, rb_mc_vals)
    plot_rms.plot_rmsf(rx_cmp, v_a_cmp, title=f"A bound mGlu3 ({mglu_a_bound}) RMSF Combined", save_path=str(dirs["bound_cmp"] / "A_bound_mglu_rmsf_combined.png"))
    plot_rms.plot_rmsf(rx_cmp, v_m_cmp, title=f"{mutant.name} bound mGlu3 ({mglu_m_bound}) RMSF Combined", save_path=str(dirs["bound_cmp"] / f"{mutant.name}_bound_mglu_rmsf_combined.png"))
    plot_rmsf_diff.plot_difference(rx_cmp, v_a_cmp, v_m_cmp, title=f"|Delta RMSF| bound mGlu3 A vs {mutant.name} (Combined)", save_path=str(dirs["bound_cmp"] / f"delta_rmsf_bound_mglu_A_vs_{mutant.name}_combined.png"))
    hx_a_bound, hy_a_bound = aggregate_helicity_mean(a_psf, a_runs, [mglu_a_bound], helicity_resids, skip, progress_label=f"{mutant.name} A bound helicity for cmp")
    _, hy_m_bound = aggregate_helicity_mean(str(mutant.psf), mut_runs, [mglu_m_bound], helicity_resids, skip, progress_label=f"{mutant.name} MUT bound helicity for cmp")
    hx_cmp_h, hy_ac, hy_mc = truncate_pair(hx_a_bound, hy_a_bound, hy_m_bound)
    plot_helicity.plot(
        md_steps=hx_cmp_h,
        helicity_series={f"A_bound_{mglu_a_bound}": hy_ac, f"{mutant.name}_bound_{mglu_m_bound}": hy_mc},
        title=f"mGlu3 Bound Helicity Comparison (A vs {mutant.name}, Combined)",
        save_path=str(dirs["bound_cmp"] / f"helicity_bound_mglu_A_vs_{mutant.name}_combined.png"),
    )
    close_figs()
    stage.update(1)

    hbond_a = aggregate_hbond_contacts(a_psf, a_runs, skip, roles_a.core_segids, progress_label=f"{mutant.name} A HBond")
    hbond_m = aggregate_hbond_contacts(str(mutant.psf), mut_runs, skip, roles_m.core_segids, progress_label=f"{mutant.name} MUT HBond")
    hbond_a_csv = ProteinHydrogenBondAnalyzer.write_csv(hbond_a, str(dirs["hbond"] / "A_hbond_contacts_combined.csv"))
    hbond_m_csv = ProteinHydrogenBondAnalyzer.write_csv(hbond_m, str(dirs["hbond"] / f"{mutant.name}_hbond_contacts_combined.csv"))
    plot_hbond.plot_top_contacts(csv_path=str(hbond_a_csv), top_n=15, title="A Combined Top 15 H-bond Contacts", save_path=str(dirs["hbond"] / "A_hbond_top15_combined.png"))
    plot_hbond.plot_top_contacts(csv_path=str(hbond_m_csv), top_n=15, title=f"{mutant.name} Combined Top 15 H-bond Contacts", save_path=str(dirs["hbond"] / f"{mutant.name}_hbond_top15_combined.png"))
    close_figs()
    stage.update(1)

    community_a_info = write_community_from_combined_dccm(
        psf=a_psf,
        dcd_for_coordinates=str(a_runs[0]),
        selection=a_sel_ca,
        dccm=np.asarray(a_dccm["dccm"], dtype=float),
        output_pdb=str(dirs["community"] / "community_A_combined_frame0.pdb"),
    )
    community_m_info = write_community_from_combined_dccm(
        psf=str(mutant.psf),
        dcd_for_coordinates=str(mut_runs[0]),
        selection=m_sel_ca,
        dccm=np.asarray(m_dccm["dccm"], dtype=float),
        output_pdb=str(dirs["community"] / f"community_{mutant.name}_combined_frame0.pdb"),
    )
    stage.update(1)
    stage.close()

    summary = {
        "mutant": mutant.name,
        "runs_used": [run_id_from_name(path.name) for path in a_runs],
        "skip_first_n_frames_per_run": skip,
        "roles_A": {
            "mglu_segids": roles_a.mglu_segids,
            "arrestin_segids": roles_a.arrestin_segids,
            "core_segids": roles_a.core_segids,
        },
        "roles_mutant": {
            "mglu_segids": roles_m.mglu_segids,
            "arrestin_segids": roles_m.arrestin_segids,
            "core_segids": roles_m.core_segids,
        },
        "bound_map_A": bound_map_a,
        "bound_map_mutant": bound_map_m,
        "bound_scores_A": bound_scores_a,
        "bound_scores_mutant": bound_scores_m,
        "A_bound_arrestin": arrestin_a,
        "A_bound_mglu": mglu_a_bound,
        "A_unbound_mglu": mglu_a_unbound,
        "mutant_bound_arrestin": arrestin_m,
        "mutant_bound_mglu": mglu_m_bound,
        "mutant_unbound_mglu": mglu_m_unbound,
        "helicity_target_resids": list(helicity_resids),
        "community_A": community_a_info,
        "community_mutant": community_m_info,
    }
    save_json(
        dirs["root"] / "summary_mutant.json",
        summary,
        repo_root=repo_root,
        workspace_root=workspace_root,
    )
    return summary


def run() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    a_psf = str(Path(args.a_psf))
    a_dir = Path(args.a_dir)
    mut_root = Path(args.mut_root)
    mut_results = Path(args.mut_results)
    skip = int(args.skip_first_n_frames)
    helicity_resids = parse_resids_csv(args.helicity_resids)
    ensure_dirs(mut_results=mut_results)
    a_runs_all = collect_runs(a_dir)
    if not a_runs_all:
        raise ValueError(f"No A RUN trajectories found in {a_dir}")
    mutants = collect_mutants(
        mut_root=mut_root,
        explicit=parse_csv_names(args.mutants),
        max_mutants=int(args.max_mutants),
    )
    if not mutants:
        raise ValueError(f"No mutant systems found in {mut_root}")

    global_summary: dict[str, object] = {
        "a_psf": a_psf,
        "mut_root": str(mut_root),
        "mutants_processed": [],
        "mutants_failed": [],
        "skip_first_n_frames_per_run": skip,
        "helicity_target_resids": helicity_resids,
    }

    for mutant in progress_iter(mutants, desc="Mutants", unit="mutant", leave=True):
        common_ids = sorted(set(a_runs_all).intersection(mutant.run_map))
        if not common_ids:
            global_summary["mutants_failed"].append({
                "mutant": mutant.name,
                "error": "No matching RUN ids with A",
            })
            continue
        if args.max_runs and int(args.max_runs) > 0:
            common_ids = common_ids[: int(args.max_runs)]
        a_runs = [a_runs_all[rid] for rid in common_ids]
        mut_runs = [mutant.run_map[rid] for rid in common_ids]
        try:
            summary = run_single_mutant(
                mutant=mutant,
                a_psf=a_psf,
                a_runs=a_runs,
                mut_runs=mut_runs,
                helicity_resids=helicity_resids,
                skip=skip,
                mut_results=mut_results,
                repo_root=project_root,
                workspace_root=workspace_root,
            )
            global_summary["mutants_processed"].append(summary)
        except Exception as exc:
            global_summary["mutants_failed"].append({
                "mutant": mutant.name,
                "runs": common_ids,
                "error": str(exc),
            })
        close_figs()

    save_json(
        mut_results / "summary_mut_vs_A_combined.json",
        global_summary,
        repo_root=project_root,
        workspace_root=workspace_root,
    )
    print(f"MUT vs A combined analysis completed: {mut_results}")
    print(f"Mutants completed: {len(global_summary['mutants_processed'])}")
    print(f"Mutants failed: {len(global_summary['mutants_failed'])}")


if __name__ == "__main__":
    run()
