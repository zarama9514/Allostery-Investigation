from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
import numpy as np
from MDAnalysis.lib.distances import capped_distance


@dataclass(frozen=True)
class TailPairSpec:
    arrestin_segid: str
    tail_segid: str
    label: str


class _TrajectoryBase:
    def __init__(self, psf: str, dcd: str | Sequence[str]):
        self.psf = psf
        self.dcd = dcd

    @staticmethod
    def _validate_step(step: int) -> int:
        if step < 1:
            raise ValueError("step must be >= 1")
        return step

    @staticmethod
    def _validate_skip(skip_first_n_frames: int) -> int:
        if skip_first_n_frames < 0:
            raise ValueError("skip_first_n_frames must be >= 0")
        return skip_first_n_frames

    def _build_universe(self) -> mda.Universe:
        if isinstance(self.dcd, str):
            return mda.Universe(self.psf, self.dcd)
        return mda.Universe(self.psf, list(self.dcd))

    @staticmethod
    def _frame_dt_ps(trajectory, step: int) -> float:
        dt_raw = float(getattr(trajectory, "dt", 1.0))
        if not np.isfinite(dt_raw) or dt_raw <= 0.0:
            dt_raw = 1.0
        return dt_raw * float(step)

    @staticmethod
    def _segment_lengths(indices: list[int]) -> list[int]:
        if not indices:
            return []
        lengths: list[int] = []
        current = 1
        previous = indices[0]
        for idx in indices[1:]:
            if idx == previous + 1:
                current += 1
            else:
                lengths.append(current)
                current = 1
            previous = idx
        lengths.append(current)
        return lengths

    @staticmethod
    def _safe_segid(atom) -> str:
        segid = str(atom.segid).strip()
        if segid:
            return segid
        chainid = str(getattr(atom, "chainID", "")).strip()
        if chainid:
            return chainid
        return "NA"

    @staticmethod
    def _residue_label(segid: str, resname: str, resid: int) -> str:
        return f"{segid}:{resname}{resid}"


class PhosphoSASAAnalyzer(_TrajectoryBase):
    @staticmethod
    def _target_residue_entries(
        topology: md.Topology,
        tail_segid: str,
        phospho_resids: Sequence[int],
    ) -> tuple[list[int], list[str], list[int]]:
        phospho_set = {int(resid) for resid in phospho_resids}
        targets: list[tuple[int, int, str, list[int]]] = []
        for residue in topology.residues:
            segment_id = str(getattr(residue, "segment_id", "")).strip()
            if segment_id != str(tail_segid):
                continue
            resid = int(residue.resSeq)
            if resid not in phospho_set:
                continue
            label = f"{segment_id}:{residue.name}{resid}"
            atom_indices = [int(atom.index) for atom in residue.atoms]
            targets.append((resid, int(residue.index), label, atom_indices))
        targets.sort(key=lambda item: item[0])
        if not targets:
            raise ValueError(f"No phospho residues found for segid {tail_segid}")
        flattened_atom_indices: list[int] = []
        for item in targets:
            flattened_atom_indices.extend(item[3])
        return [item[1] for item in targets], [item[2] for item in targets], flattened_atom_indices

    @staticmethod
    def write_frame_csv(records: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str]
        if records:
            fieldnames = list(dict(records[0]).keys())
        else:
            fieldnames = ["run_id", "frame", "time_ps"]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    @staticmethod
    def write_summary_csv(rows: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label",
            "mean_sasa_angstrom2",
            "median_sasa_angstrom2",
            "std_sasa_angstrom2",
            "min_sasa_angstrom2",
            "max_sasa_angstrom2",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    @staticmethod
    def summarize_records(records: Sequence[Mapping[str, object]], residue_labels: Sequence[str]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for label in list(residue_labels) + ["total_sasa_angstrom2"]:
            values = np.asarray([float(row[label]) for row in records], dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    "label": "TOTAL" if label == "total_sasa_angstrom2" else label,
                    "mean_sasa_angstrom2": float(np.mean(values)),
                    "median_sasa_angstrom2": float(np.median(values)),
                    "std_sasa_angstrom2": float(np.std(values)),
                    "min_sasa_angstrom2": float(np.min(values)),
                    "max_sasa_angstrom2": float(np.max(values)),
                }
            )
        return rows

    def calculate(
        self,
        tail_segid: str,
        phospho_resids: Sequence[int],
        step: int = 10,
        skip_first_n_frames: int = 0,
        chunk_size: int = 200,
    ) -> dict[str, object]:
        if not isinstance(self.dcd, str):
            raise ValueError("PhosphoSASAAnalyzer expects a single DCD file")
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        iterator = md.iterload(
            self.dcd,
            top=self.psf,
            chunk=int(chunk_size),
            stride=int(step),
            skip=int(skip_first_n_frames),
        )
        residue_indices: list[int] | None = None
        residue_labels: list[str] | None = None
        atom_indices: list[int] | None = None
        records: list[dict[str, object]] = []
        current_frame = int(skip_first_n_frames)
        for chunk in iterator:
            if residue_indices is None or residue_labels is None or atom_indices is None:
                residue_indices, residue_labels, atom_indices = self._target_residue_entries(
                    chunk.topology,
                    tail_segid=tail_segid,
                    phospho_resids=phospho_resids,
                )
            sasa = md.shrake_rupley(chunk, mode="residue", atom_indices=atom_indices) * 100.0
            selected = sasa[:, residue_indices]
            total = np.sum(selected, axis=1)
            if hasattr(chunk, "time") and chunk.time is not None:
                time_values = np.asarray(chunk.time, dtype=float)
            else:
                time_values = np.arange(chunk.n_frames, dtype=float)
            frame_values = np.arange(
                current_frame,
                current_frame + int(step) * chunk.n_frames,
                int(step),
                dtype=int,
            )
            current_frame = int(frame_values[-1]) + int(step) if frame_values.size else current_frame
            for idx in range(chunk.n_frames):
                row: dict[str, object] = {
                    "frame": int(frame_values[idx]),
                    "time_ps": float(time_values[idx]),
                }
                for label, value in zip(residue_labels, selected[idx], strict=False):
                    row[label] = float(value)
                row["total_sasa_angstrom2"] = float(total[idx])
                records.append(row)
        if residue_labels is None:
            raise ValueError("No SASA frames were processed")
        return {
            "records": records,
            "residue_labels": residue_labels,
            "tail_segid": str(tail_segid),
            "phospho_resids": [int(resid) for resid in phospho_resids],
            "n_frames_analyzed": int(len(records)),
        }


class PhosphoTailContactAnalyzer(_TrajectoryBase):
    @staticmethod
    def write_contacts_csv(records: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "contact_id",
            "tail_segid",
            "tail_resname",
            "tail_resid",
            "arrestin_segid",
            "arrestin_resname",
            "arrestin_resid",
            "tail_residue_label",
            "arrestin_residue_label",
            "contact_residue_label",
            "frames_observed",
            "lifetime_ps",
            "occupancy_fraction",
            "occupancy_percent",
            "max_continuous_frames",
            "max_continuous_ps",
            "segments_count",
            "first_observed_frame",
            "last_observed_frame",
            "mean_distance_angstrom",
            "n_frames_analyzed",
            "frame_dt_ps",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    @staticmethod
    def write_count_series_csv(rows: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "frame",
            "time_ps",
            "combined_frame_index",
            "n_contact_pairs",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    def calculate(
        self,
        tail_segid: str,
        arrestin_segid: str,
        tail_resids: Sequence[int],
        n_domain_range: tuple[int, int] = (1, 180),
        heavy_atom_cutoff: float = 4.0,
        step: int = 10,
        skip_first_n_frames: int = 0,
    ) -> dict[str, object]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        universe = self._build_universe()
        analyzed_frames = [int(ts.frame) for ts in universe.trajectory[skip_first_n_frames::step]]
        if not analyzed_frames:
            raise ValueError("No frames selected after skipping")
        frame_index = {frame: idx for idx, frame in enumerate(analyzed_frames)}
        frame_dt_ps = self._frame_dt_ps(universe.trajectory, step)
        tail_resid_text = " ".join(str(int(resid)) for resid in tail_resids)
        n_start, n_end = int(n_domain_range[0]), int(n_domain_range[1])
        tail_atoms = universe.select_atoms(
            f"segid {tail_segid} and resid {tail_resid_text} and not name H*"
        )
        arrestin_atoms = universe.select_atoms(
            f"segid {arrestin_segid} and resid {n_start}:{n_end} and protein and not name H*"
        )
        if tail_atoms.n_atoms == 0:
            raise ValueError(f"No tail heavy atoms found for segid {tail_segid}")
        if arrestin_atoms.n_atoms == 0:
            raise ValueError(f"No arrestin N-domain heavy atoms found for segid {arrestin_segid}")
        tail_info = [
            (
                self._safe_segid(atom),
                str(atom.resname).strip(),
                int(atom.resid),
            )
            for atom in tail_atoms
        ]
        arrestin_info = [
            (
                self._safe_segid(atom),
                str(atom.resname).strip(),
                int(atom.resid),
            )
            for atom in arrestin_atoms
        ]
        contacts: dict[tuple[str, str, int, str, str, int], dict[str, object]] = {}
        count_series: list[dict[str, object]] = []
        for ts in universe.trajectory[skip_first_n_frames::step]:
            pairs, distances = capped_distance(
                tail_atoms.positions,
                arrestin_atoms.positions,
                max_cutoff=float(heavy_atom_cutoff),
                box=ts.dimensions,
                return_distances=True,
            )
            frame_contacts: dict[tuple[str, str, int, str, str, int], float] = {}
            if len(pairs):
                for (tail_idx, arrestin_idx), distance in zip(pairs, distances, strict=False):
                    tail_key = tail_info[int(tail_idx)]
                    arrestin_key = arrestin_info[int(arrestin_idx)]
                    key = (
                        tail_key[0],
                        tail_key[1],
                        tail_key[2],
                        arrestin_key[0],
                        arrestin_key[1],
                        arrestin_key[2],
                    )
                    if key not in frame_contacts or float(distance) < frame_contacts[key]:
                        frame_contacts[key] = float(distance)
            count_series.append(
                {
                    "frame": int(ts.frame),
                    "time_ps": float(getattr(ts, "time", frame_index[int(ts.frame)] * frame_dt_ps)),
                    "n_contact_pairs": int(len(frame_contacts)),
                }
            )
            for key, min_distance in frame_contacts.items():
                entry = contacts.get(key)
                if entry is None:
                    tail_label = self._residue_label(key[0], key[1], key[2])
                    arrestin_label = self._residue_label(key[3], key[4], key[5])
                    entry = {
                        "tail_segid": key[0],
                        "tail_resname": key[1],
                        "tail_resid": key[2],
                        "arrestin_segid": key[3],
                        "arrestin_resname": key[4],
                        "arrestin_resid": key[5],
                        "tail_residue_label": tail_label,
                        "arrestin_residue_label": arrestin_label,
                        "contact_residue_label": f"{tail_label} - {arrestin_label}",
                        "frames": set(),
                        "distance_sum": 0.0,
                        "samples": 0,
                    }
                    contacts[key] = entry
                entry["frames"].add(int(ts.frame))
                entry["distance_sum"] = float(entry["distance_sum"]) + float(min_distance)
                entry["samples"] = int(entry["samples"]) + 1
        records: list[dict[str, object]] = []
        analyzed_total = len(analyzed_frames)
        for key, entry in contacts.items():
            observed_frames = sorted(int(frame) for frame in entry["frames"])
            observed_indices = sorted(frame_index[frame] for frame in observed_frames if frame in frame_index)
            segment_lengths = self._segment_lengths(observed_indices)
            observed_count = len(observed_indices)
            records.append(
                {
                    "contact_id": f"{key[0]}:{key[2]}-{key[3]}:{key[5]}",
                    "tail_segid": entry["tail_segid"],
                    "tail_resname": entry["tail_resname"],
                    "tail_resid": int(entry["tail_resid"]),
                    "arrestin_segid": entry["arrestin_segid"],
                    "arrestin_resname": entry["arrestin_resname"],
                    "arrestin_resid": int(entry["arrestin_resid"]),
                    "tail_residue_label": entry["tail_residue_label"],
                    "arrestin_residue_label": entry["arrestin_residue_label"],
                    "contact_residue_label": entry["contact_residue_label"],
                    "frames_observed": int(observed_count),
                    "lifetime_ps": float(observed_count * frame_dt_ps),
                    "occupancy_fraction": float(observed_count / float(analyzed_total)),
                    "occupancy_percent": float(observed_count / float(analyzed_total) * 100.0),
                    "max_continuous_frames": int(max(segment_lengths) if segment_lengths else 0),
                    "max_continuous_ps": float((max(segment_lengths) if segment_lengths else 0) * frame_dt_ps),
                    "segments_count": int(len(segment_lengths)),
                    "first_observed_frame": int(observed_frames[0]) if observed_frames else -1,
                    "last_observed_frame": int(observed_frames[-1]) if observed_frames else -1,
                    "mean_distance_angstrom": float(entry["distance_sum"]) / float(entry["samples"]),
                    "n_frames_analyzed": int(analyzed_total),
                    "frame_dt_ps": float(frame_dt_ps),
                }
            )
        records.sort(key=lambda row: float(row["lifetime_ps"]), reverse=True)
        return {
            "contacts": records,
            "count_series": count_series,
            "n_frames_analyzed": int(analyzed_total),
            "frame_dt_ps": float(frame_dt_ps),
            "skip_first_n_frames": int(skip_first_n_frames),
            "step": int(step),
            "tail_segid": str(tail_segid),
            "arrestin_segid": str(arrestin_segid),
        }


class PhosphoTailSaltBridgeAnalyzer(_TrajectoryBase):
    @staticmethod
    def _infer_donor_n_atom(resname: str, hydrogen_name: str) -> str:
        if resname == "LYS":
            return "NZ"
        if hydrogen_name == "HE":
            return "NE"
        if hydrogen_name.startswith("HH1"):
            return "NH1"
        if hydrogen_name.startswith("HH2"):
            return "NH2"
        return "NA"

    @staticmethod
    def write_salt_bridge_csv(records: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "salt_bridge_id",
            "donor_segid",
            "donor_resname",
            "donor_resid",
            "donor_n_atom",
            "donor_h_atom",
            "acceptor_segid",
            "acceptor_resname",
            "acceptor_resid",
            "acceptor_atom",
            "donor_residue_label",
            "acceptor_residue_label",
            "salt_bridge_residue_label",
            "salt_bridge_atom_label",
            "frames_observed",
            "lifetime_ps",
            "occupancy_fraction",
            "occupancy_percent",
            "max_continuous_frames",
            "max_continuous_ps",
            "segments_count",
            "first_observed_frame",
            "last_observed_frame",
            "mean_distance_angstrom",
            "n_frames_analyzed",
            "frame_dt_ps",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    @staticmethod
    def write_candidate_csv(rows: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "donor_segid",
            "donor_resname",
            "donor_resid",
            "donor_residue_label",
            "frames_observed",
            "lifetime_ps",
            "max_continuous_ps",
            "segments_count",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    @staticmethod
    def write_count_series_csv(rows: Sequence[Mapping[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "frame",
            "time_ps",
            "combined_frame_index",
            "n_salt_bridge_atom_pairs",
            "n_salt_bridge_residue_pairs",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    @staticmethod
    def _candidate_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        for row in records:
            key = str(row["donor_residue_label"])
            if key not in merged:
                merged[key] = {
                    "donor_segid": row["donor_segid"],
                    "donor_resname": row["donor_resname"],
                    "donor_resid": row["donor_resid"],
                    "donor_residue_label": row["donor_residue_label"],
                    "frames_observed": int(row["frames_observed"]),
                    "lifetime_ps": float(row["lifetime_ps"]),
                    "max_continuous_ps": float(row["max_continuous_ps"]),
                    "segments_count": int(row["segments_count"]),
                }
            else:
                acc = merged[key]
                acc["frames_observed"] = int(acc["frames_observed"]) + int(row["frames_observed"])
                acc["lifetime_ps"] = float(acc["lifetime_ps"]) + float(row["lifetime_ps"])
                acc["max_continuous_ps"] = max(float(acc["max_continuous_ps"]), float(row["max_continuous_ps"]))
                acc["segments_count"] = int(acc["segments_count"]) + int(row["segments_count"])
        rows = list(merged.values())
        rows.sort(key=lambda row: float(row["lifetime_ps"]), reverse=True)
        return rows

    def calculate(
        self,
        tail_segid: str,
        arrestin_segid: str,
        phospho_resids: Sequence[int],
        n_domain_range: tuple[int, int] = (1, 180),
        hydrogen_oxygen_cutoff: float = 2.5,
        step: int = 10,
        skip_first_n_frames: int = 0,
    ) -> dict[str, object]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        universe = self._build_universe()
        analyzed_frames = [int(ts.frame) for ts in universe.trajectory[skip_first_n_frames::step]]
        if not analyzed_frames:
            raise ValueError("No frames selected after skipping")
        frame_index = {frame: idx for idx, frame in enumerate(analyzed_frames)}
        frame_dt_ps = self._frame_dt_ps(universe.trajectory, step)
        phospho_text = " ".join(str(int(resid)) for resid in phospho_resids)
        n_start, n_end = int(n_domain_range[0]), int(n_domain_range[1])
        donor_h_atoms = universe.select_atoms(
            (
                f"segid {arrestin_segid} and resid {n_start}:{n_end} and resname LYS ARG and "
                "name HZ1 HZ2 HZ3 HE HH11 HH12 HH21 HH22"
            )
        )
        acceptor_o_atoms = universe.select_atoms(
            f"segid {tail_segid} and resid {phospho_text} and name O1P O2P OT"
        )
        if donor_h_atoms.n_atoms == 0:
            raise ValueError(f"No Lys/Arg sidechain hydrogens found for segid {arrestin_segid}")
        if acceptor_o_atoms.n_atoms == 0:
            raise ValueError(f"No phosphate oxygens found for segid {tail_segid}")
        donor_info = [
            (
                self._safe_segid(atom),
                str(atom.resname).strip(),
                int(atom.resid),
                self._infer_donor_n_atom(str(atom.resname).strip(), str(atom.name).strip()),
                str(atom.name).strip(),
            )
            for atom in donor_h_atoms
        ]
        acceptor_info = [
            (
                self._safe_segid(atom),
                str(atom.resname).strip(),
                int(atom.resid),
                str(atom.name).strip(),
            )
            for atom in acceptor_o_atoms
        ]
        atom_level: dict[tuple[str, str, int, str, str, str, str, int, str], dict[str, object]] = {}
        residue_level: dict[tuple[str, str, int, str, str, int], dict[str, object]] = {}
        count_series: list[dict[str, object]] = []
        for ts in universe.trajectory[skip_first_n_frames::step]:
            pairs, distances = capped_distance(
                donor_h_atoms.positions,
                acceptor_o_atoms.positions,
                max_cutoff=float(hydrogen_oxygen_cutoff),
                box=ts.dimensions,
                return_distances=True,
            )
            frame_atom_pairs: dict[tuple[str, str, int, str, str, str, str, int, str], float] = {}
            frame_residue_pairs: dict[tuple[str, str, int, str, str, int], float] = {}
            if len(pairs):
                for (donor_idx, acceptor_idx), distance in zip(pairs, distances, strict=False):
                    donor_key = donor_info[int(donor_idx)]
                    acceptor_key = acceptor_info[int(acceptor_idx)]
                    atom_key = (
                        donor_key[0],
                        donor_key[1],
                        donor_key[2],
                        donor_key[3],
                        donor_key[4],
                        acceptor_key[0],
                        acceptor_key[1],
                        acceptor_key[2],
                        acceptor_key[3],
                    )
                    residue_key = (
                        donor_key[0],
                        donor_key[1],
                        donor_key[2],
                        acceptor_key[0],
                        acceptor_key[1],
                        acceptor_key[2],
                    )
                    if atom_key not in frame_atom_pairs or float(distance) < frame_atom_pairs[atom_key]:
                        frame_atom_pairs[atom_key] = float(distance)
                    if residue_key not in frame_residue_pairs or float(distance) < frame_residue_pairs[residue_key]:
                        frame_residue_pairs[residue_key] = float(distance)
            count_series.append(
                {
                    "frame": int(ts.frame),
                    "time_ps": float(getattr(ts, "time", frame_index[int(ts.frame)] * frame_dt_ps)),
                    "n_salt_bridge_atom_pairs": int(len(frame_atom_pairs)),
                    "n_salt_bridge_residue_pairs": int(len(frame_residue_pairs)),
                }
            )
            for key, min_distance in frame_atom_pairs.items():
                entry = atom_level.get(key)
                if entry is None:
                    donor_label = self._residue_label(key[0], key[1], key[2])
                    acceptor_label = self._residue_label(key[5], key[6], key[7])
                    entry = {
                        "donor_segid": key[0],
                        "donor_resname": key[1],
                        "donor_resid": key[2],
                        "donor_n_atom": key[3],
                        "donor_h_atom": key[4],
                        "acceptor_segid": key[5],
                        "acceptor_resname": key[6],
                        "acceptor_resid": key[7],
                        "acceptor_atom": key[8],
                        "donor_residue_label": donor_label,
                        "acceptor_residue_label": acceptor_label,
                        "salt_bridge_residue_label": f"{donor_label} - {acceptor_label}",
                        "salt_bridge_atom_label": f"{donor_label}:{key[4]} -> {acceptor_label}:{key[8]}",
                        "frames": set(),
                        "distance_sum": 0.0,
                        "samples": 0,
                    }
                    atom_level[key] = entry
                entry["frames"].add(int(ts.frame))
                entry["distance_sum"] = float(entry["distance_sum"]) + float(min_distance)
                entry["samples"] = int(entry["samples"]) + 1
            for key, min_distance in frame_residue_pairs.items():
                entry = residue_level.get(key)
                if entry is None:
                    donor_label = self._residue_label(key[0], key[1], key[2])
                    acceptor_label = self._residue_label(key[3], key[4], key[5])
                    entry = {
                        "donor_segid": key[0],
                        "donor_resname": key[1],
                        "donor_resid": key[2],
                        "donor_n_atom": "MULTI",
                        "donor_h_atom": "MULTI",
                        "acceptor_segid": key[3],
                        "acceptor_resname": key[4],
                        "acceptor_resid": key[5],
                        "acceptor_atom": "MULTI",
                        "donor_residue_label": donor_label,
                        "acceptor_residue_label": acceptor_label,
                        "salt_bridge_residue_label": f"{donor_label} - {acceptor_label}",
                        "salt_bridge_atom_label": f"{donor_label} -> {acceptor_label}",
                        "frames": set(),
                        "distance_sum": 0.0,
                        "samples": 0,
                    }
                    residue_level[key] = entry
                entry["frames"].add(int(ts.frame))
                entry["distance_sum"] = float(entry["distance_sum"]) + float(min_distance)
                entry["samples"] = int(entry["samples"]) + 1
        analyzed_total = len(analyzed_frames)

        def finalize(entries: Mapping[object, Mapping[str, object]], id_builder) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for key, entry in entries.items():
                observed_frames = sorted(int(frame) for frame in entry["frames"])
                observed_indices = sorted(frame_index[frame] for frame in observed_frames if frame in frame_index)
                segment_lengths = self._segment_lengths(observed_indices)
                observed_count = len(observed_indices)
                rows.append(
                    {
                        "salt_bridge_id": id_builder(key),
                        "donor_segid": entry["donor_segid"],
                        "donor_resname": entry["donor_resname"],
                        "donor_resid": int(entry["donor_resid"]),
                        "donor_n_atom": entry["donor_n_atom"],
                        "donor_h_atom": entry["donor_h_atom"],
                        "acceptor_segid": entry["acceptor_segid"],
                        "acceptor_resname": entry["acceptor_resname"],
                        "acceptor_resid": int(entry["acceptor_resid"]),
                        "acceptor_atom": entry["acceptor_atom"],
                        "donor_residue_label": entry["donor_residue_label"],
                        "acceptor_residue_label": entry["acceptor_residue_label"],
                        "salt_bridge_residue_label": entry["salt_bridge_residue_label"],
                        "salt_bridge_atom_label": entry["salt_bridge_atom_label"],
                        "frames_observed": int(observed_count),
                        "lifetime_ps": float(observed_count * frame_dt_ps),
                        "occupancy_fraction": float(observed_count / float(analyzed_total)),
                        "occupancy_percent": float(observed_count / float(analyzed_total) * 100.0),
                        "max_continuous_frames": int(max(segment_lengths) if segment_lengths else 0),
                        "max_continuous_ps": float((max(segment_lengths) if segment_lengths else 0) * frame_dt_ps),
                        "segments_count": int(len(segment_lengths)),
                        "first_observed_frame": int(observed_frames[0]) if observed_frames else -1,
                        "last_observed_frame": int(observed_frames[-1]) if observed_frames else -1,
                        "mean_distance_angstrom": float(entry["distance_sum"]) / float(entry["samples"]),
                        "n_frames_analyzed": int(analyzed_total),
                        "frame_dt_ps": float(frame_dt_ps),
                    }
                )
            rows.sort(key=lambda row: float(row["lifetime_ps"]), reverse=True)
            return rows

        atom_records = finalize(
            atom_level,
            lambda key: f"{key[0]}:{key[2]}:{key[4]}-{key[5]}:{key[7]}:{key[8]}",
        )
        residue_records = finalize(
            residue_level,
            lambda key: f"{key[0]}:{key[2]}-{key[3]}:{key[5]}",
        )
        candidate_rows = self._candidate_rows(atom_records)
        return {
            "salt_bridges_atom": atom_records,
            "salt_bridges_residue": residue_records,
            "candidate_residues": candidate_rows,
            "count_series": count_series,
            "n_frames_analyzed": int(analyzed_total),
            "frame_dt_ps": float(frame_dt_ps),
            "skip_first_n_frames": int(skip_first_n_frames),
            "step": int(step),
            "tail_segid": str(tail_segid),
            "arrestin_segid": str(arrestin_segid),
        }


class PhosphoCouplingPlotter:
    def __init__(self, figsize: tuple[float, float] = (12.0, 6.0), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi

    def _save(self, fig: plt.Figure, save_path: str | None) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

    def plot_sasa_distribution(
        self,
        records: Sequence[Mapping[str, object]],
        residue_labels: Sequence[str],
        title: str,
        save_path: str | None = None,
        bins: int = 50,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ["#003049", "#d62828", "#f77f00", "#6a4c93", "#2a9d8f", "#7f5539"]
        for idx, label in enumerate(residue_labels):
            values = np.asarray([float(row[label]) for row in records], dtype=float)
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[idx % len(colors)],
                label=label,
            )
        ax.set_xlabel("SASA, Å²")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(title="Phospho Residue", fontsize=8, title_fontsize=9)
        self._save(fig, save_path)
        return fig, ax

    def plot_total_sasa_distribution(
        self,
        records: Sequence[Mapping[str, object]],
        title: str,
        save_path: str | None = None,
        bins: int = 50,
    ) -> tuple[plt.Figure, plt.Axes]:
        values = np.asarray([float(row["total_sasa_angstrom2"]) for row in records], dtype=float)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(values, bins=bins, density=True, color="#1d3557", alpha=0.85)
        ax.set_xlabel("Total Phospho SASA, Å²")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        self._save(fig, save_path)
        return fig, ax

    def plot_top_lifetime(
        self,
        records: Sequence[Mapping[str, object]],
        label_field: str,
        title: str,
        save_path: str | None = None,
        top_n: int = 15,
    ) -> tuple[list[dict[str, object]], plt.Figure, plt.Axes]:
        rows = [dict(row) for row in records]
        rows.sort(key=lambda row: float(row["lifetime_ps"]), reverse=True)
        top_rows = rows[: int(top_n)]
        plot_rows = list(reversed(top_rows))
        labels = [str(row[label_field]) for row in plot_rows]
        values = [float(row["lifetime_ps"]) for row in plot_rows]
        height = max(self.figsize[1], 0.45 * max(1, len(plot_rows)) + 1.5)
        fig, ax = plt.subplots(figsize=(self.figsize[0], height))
        ax.barh(np.arange(len(plot_rows)), values, color="#264653", alpha=0.9)
        ax.set_yticks(np.arange(len(plot_rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Lifetime (ps)")
        ax.set_ylabel("Interaction")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        self._save(fig, save_path)
        return top_rows, fig, ax

    def plot_count_timeseries(
        self,
        rows: Sequence[Mapping[str, object]],
        count_fields: Sequence[str],
        title: str,
        y_label: str,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ["#1d3557", "#d62828", "#2a9d8f", "#6a4c93"]
        x = np.arange(len(rows), dtype=int)
        for idx, field in enumerate(count_fields):
            values = np.asarray([float(row[field]) for row in rows], dtype=float)
            ax.plot(x, values, linewidth=1.3, color=colors[idx % len(colors)], label=field)
        boundaries: list[int] = []
        previous = None
        for idx, row in enumerate(rows):
            current = row.get("run_id")
            if previous is None:
                previous = current
                continue
            if current != previous:
                boundaries.append(idx)
                previous = current
        for boundary in boundaries:
            ax.axvline(boundary, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Analyzed Frame Index")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(count_fields) > 1:
            ax.legend(title="Series")
        self._save(fig, save_path)
        return fig, ax

    def plot_delta_sasa(
        self,
        rows: Sequence[Mapping[str, object]],
        title: str,
        save_path: str | None = None,
        value_field: str = "delta_mean_sasa_angstrom2",
    ) -> tuple[plt.Figure, plt.Axes]:
        plot_rows = [dict(row) for row in rows]
        labels = [str(row["label"]) for row in plot_rows]
        values = [float(row[value_field]) for row in plot_rows]
        height = max(self.figsize[1], 0.55 * max(1, len(plot_rows)) + 1.5)
        fig, ax = plt.subplots(figsize=(self.figsize[0], height))
        y = np.arange(len(plot_rows), dtype=int)
        colors = ["#1d3557" if value >= 0.0 else "#c1121f" for value in values]
        bars = ax.barh(y, values, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Δ mean SASA, Å²")
        ax.set_ylabel("Phospho Residue / Total")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        max_value = max((abs(value) for value in values), default=0.0)
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_width() + (max(max_value * 0.01, 0.5) if value >= 0.0 else -max(max_value * 0.01, 0.5)),
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0.0 else "right",
                fontsize=8,
            )
        self._save(fig, save_path)
        return fig, ax
