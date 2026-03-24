from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


class ProteinHydrogenBondAnalyzer:
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
    def _protein_like_segids(
        universe: mda.Universe,
        selection: str,
        min_residues_per_subunit: int,
    ) -> set[str]:
        if min_residues_per_subunit < 1:
            raise ValueError("min_residues_per_subunit must be >= 1")
        atoms = universe.select_atoms(selection)
        segids: set[str] = set()
        for segment in atoms.segments:
            if len(segment.residues) >= min_residues_per_subunit:
                segid = str(segment.segid).strip()
                if segid:
                    segids.add(segid)
        return segids

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

    @staticmethod
    def _contact_label(
        donor_segid: str,
        donor_resname: str,
        donor_resid: int,
        donor_atom: str,
        acceptor_segid: str,
        acceptor_resname: str,
        acceptor_resid: int,
        acceptor_atom: str,
    ) -> tuple[str, str]:
        residue_label = (
            f"{donor_segid}:{donor_resname}{donor_resid} - "
            f"{acceptor_segid}:{acceptor_resname}{acceptor_resid}"
        )
        atom_label = (
            f"{donor_segid}:{donor_resname}{donor_resid}:{donor_atom} -> "
            f"{acceptor_segid}:{acceptor_resname}{acceptor_resid}:{acceptor_atom}"
        )
        return residue_label, atom_label

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

    def calculate(
        self,
        donor_selection: str = "protein",
        acceptor_selection: str = "protein",
        d_a_cutoff: float = 3.0,
        d_h_a_angle_cutoff: float = 150.0,
        update_selections: bool = True,
        inter_subunit_only: bool = True,
        allowed_segids: Sequence[str] | None = None,
        min_residues_per_subunit: int = 20,
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> dict[str, object]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        universe = self._build_universe()
        analyzed_frames = [int(ts.frame) for ts in universe.trajectory[skip_first_n_frames::step]]
        if not analyzed_frames:
            raise ValueError("No frames selected after skipping")
        dt_raw = float(getattr(universe.trajectory, "dt", 1.0))
        if not np.isfinite(dt_raw) or dt_raw <= 0.0:
            dt_raw = 1.0
        frame_dt_ps = dt_raw * float(step)
        frame_index = {frame: idx for idx, frame in enumerate(analyzed_frames)}
        if allowed_segids is None:
            allowed_segids_set = self._protein_like_segids(
                universe=universe,
                selection="protein",
                min_residues_per_subunit=min_residues_per_subunit,
            )
        else:
            allowed_segids_set = {str(segid).strip() for segid in allowed_segids if str(segid).strip()}
        if not allowed_segids_set:
            raise ValueError("No allowed protein subunits were found")
        analyzer = HydrogenBondAnalysis(
            universe=universe,
            donors_sel=donor_selection,
            acceptors_sel=acceptor_selection,
            d_a_cutoff=d_a_cutoff,
            d_h_a_angle_cutoff=d_h_a_angle_cutoff,
            update_selections=update_selections,
        )
        analyzer.run(start=skip_first_n_frames, step=step)
        hbonds = np.asarray(analyzer.results.hbonds, dtype=float)
        contacts: dict[tuple[int, int], dict[str, object]] = {}
        total_rows = int(hbonds.shape[0]) if hbonds.ndim == 2 else 0
        for row in hbonds:
            frame = int(row[0])
            donor_idx = int(row[1])
            acceptor_idx = int(row[3])
            donor_atom = universe.atoms[donor_idx]
            acceptor_atom = universe.atoms[acceptor_idx]
            donor_segid = self._safe_segid(donor_atom)
            acceptor_segid = self._safe_segid(acceptor_atom)
            if donor_segid not in allowed_segids_set or acceptor_segid not in allowed_segids_set:
                continue
            if inter_subunit_only and donor_segid == acceptor_segid:
                continue
            key = (donor_idx, acceptor_idx)
            entry = contacts.get(key)
            if entry is None:
                donor_resname = str(donor_atom.resname).strip()
                acceptor_resname = str(acceptor_atom.resname).strip()
                donor_resid = int(donor_atom.resid)
                acceptor_resid = int(acceptor_atom.resid)
                donor_atom_name = str(donor_atom.name).strip()
                acceptor_atom_name = str(acceptor_atom.name).strip()
                residue_contact_label, atom_contact_label = self._contact_label(
                    donor_segid=donor_segid,
                    donor_resname=donor_resname,
                    donor_resid=donor_resid,
                    donor_atom=donor_atom_name,
                    acceptor_segid=acceptor_segid,
                    acceptor_resname=acceptor_resname,
                    acceptor_resid=acceptor_resid,
                    acceptor_atom=acceptor_atom_name,
                )
                entry = {
                    "donor_segid": donor_segid,
                    "donor_resname": donor_resname,
                    "donor_resid": donor_resid,
                    "donor_atom": donor_atom_name,
                    "acceptor_segid": acceptor_segid,
                    "acceptor_resname": acceptor_resname,
                    "acceptor_resid": acceptor_resid,
                    "acceptor_atom": acceptor_atom_name,
                    "donor_residue_label": self._residue_label(donor_segid, donor_resname, donor_resid),
                    "acceptor_residue_label": self._residue_label(acceptor_segid, acceptor_resname, acceptor_resid),
                    "contact_residue_label": residue_contact_label,
                    "contact_atom_label": atom_contact_label,
                    "frames": set(),
                    "distance_sum": 0.0,
                    "angle_sum": 0.0,
                    "samples": 0,
                }
                contacts[key] = entry
            entry["frames"].add(frame)
            entry["distance_sum"] = float(entry["distance_sum"]) + float(row[4])
            entry["angle_sum"] = float(entry["angle_sum"]) + float(row[5])
            entry["samples"] = int(entry["samples"]) + 1
        records: list[dict[str, object]] = []
        analyzed_total = len(analyzed_frames)
        for key, entry in contacts.items():
            observed_frames = sorted(int(frame) for frame in entry["frames"])
            observed_indices = sorted(frame_index[frame] for frame in observed_frames if frame in frame_index)
            segment_lengths = self._segment_lengths(observed_indices)
            observed_count = len(observed_indices)
            lifetime_ps = observed_count * frame_dt_ps
            occupancy_fraction = observed_count / float(analyzed_total)
            max_segment_frames = max(segment_lengths) if segment_lengths else 0
            record = {
                "contact_id": f"{key[0]}-{key[1]}",
                "donor_segid": entry["donor_segid"],
                "donor_resname": entry["donor_resname"],
                "donor_resid": int(entry["donor_resid"]),
                "donor_atom": entry["donor_atom"],
                "acceptor_segid": entry["acceptor_segid"],
                "acceptor_resname": entry["acceptor_resname"],
                "acceptor_resid": int(entry["acceptor_resid"]),
                "acceptor_atom": entry["acceptor_atom"],
                "donor_residue_label": entry["donor_residue_label"],
                "acceptor_residue_label": entry["acceptor_residue_label"],
                "contact_residue_label": entry["contact_residue_label"],
                "contact_atom_label": entry["contact_atom_label"],
                "frames_observed": observed_count,
                "lifetime_ps": float(lifetime_ps),
                "occupancy_fraction": float(occupancy_fraction),
                "occupancy_percent": float(occupancy_fraction * 100.0),
                "max_continuous_frames": int(max_segment_frames),
                "max_continuous_ps": float(max_segment_frames * frame_dt_ps),
                "segments_count": int(len(segment_lengths)),
                "first_observed_frame": int(observed_frames[0]) if observed_frames else -1,
                "last_observed_frame": int(observed_frames[-1]) if observed_frames else -1,
                "mean_distance_angstrom": float(entry["distance_sum"]) / float(entry["samples"]),
                "mean_angle_degree": float(entry["angle_sum"]) / float(entry["samples"]),
                "n_frames_analyzed": int(analyzed_total),
                "frame_dt_ps": float(frame_dt_ps),
            }
            records.append(record)
        records.sort(
            key=lambda row: (
                float(row["lifetime_ps"]),
                int(row["frames_observed"]),
                float(row["mean_distance_angstrom"]),
            ),
            reverse=True,
        )
        return {
            "contacts": records,
            "n_contacts": int(len(records)),
            "n_frames_analyzed": int(analyzed_total),
            "skip_first_n_frames": int(skip_first_n_frames),
            "step": int(step),
            "frame_dt_ps": float(frame_dt_ps),
            "raw_hbond_rows": int(total_rows),
            "allowed_segids": sorted(allowed_segids_set),
        }

    @staticmethod
    def write_csv(contacts: Sequence[dict[str, object]], output_csv: str) -> Path:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "contact_id",
            "donor_segid",
            "donor_resname",
            "donor_resid",
            "donor_atom",
            "acceptor_segid",
            "acceptor_resname",
            "acceptor_resid",
            "acceptor_atom",
            "donor_residue_label",
            "acceptor_residue_label",
            "contact_residue_label",
            "contact_atom_label",
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
            "mean_angle_degree",
            "n_frames_analyzed",
            "frame_dt_ps",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in contacts:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path

    def run(
        self,
        output_csv: str,
        donor_selection: str = "protein",
        acceptor_selection: str = "protein",
        d_a_cutoff: float = 3.0,
        d_h_a_angle_cutoff: float = 150.0,
        update_selections: bool = True,
        inter_subunit_only: bool = True,
        allowed_segids: Sequence[str] | None = None,
        min_residues_per_subunit: int = 20,
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> dict[str, object]:
        result = self.calculate(
            donor_selection=donor_selection,
            acceptor_selection=acceptor_selection,
            d_a_cutoff=d_a_cutoff,
            d_h_a_angle_cutoff=d_h_a_angle_cutoff,
            update_selections=update_selections,
            inter_subunit_only=inter_subunit_only,
            allowed_segids=allowed_segids,
            min_residues_per_subunit=min_residues_per_subunit,
            step=step,
            skip_first_n_frames=skip_first_n_frames,
        )
        output_path = self.write_csv(result["contacts"], output_csv=output_csv)
        return {
            "output_csv": str(output_path),
            "n_contacts": int(result["n_contacts"]),
            "n_frames_analyzed": int(result["n_frames_analyzed"]),
            "step": int(result["step"]),
            "skip_first_n_frames": int(result["skip_first_n_frames"]),
            "frame_dt_ps": float(result["frame_dt_ps"]),
            "raw_hbond_rows": int(result["raw_hbond_rows"]),
            "allowed_segids": list(result["allowed_segids"]),
        }
