from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms


class GeometryBase:
    def __init__(self, psf: str, dcd: str | Sequence[str]):
        self.psf = psf
        self.dcd = dcd

    def _build_universe(self) -> mda.Universe:
        if isinstance(self.dcd, str):
            return mda.Universe(self.psf, self.dcd)
        return mda.Universe(self.psf, list(self.dcd))

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


class RMSDAnalyzer(GeometryBase):
    def calculate(
        self,
        selection: str = "protein and backbone",
        ref_frame: int = 0,
        groupselections: Sequence[str] | None = None,
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> dict[str, np.ndarray]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        universe = self._build_universe()
        analyzer = rms.RMSD(
            universe,
            universe,
            select=selection,
            ref_frame=ref_frame,
            groupselections=list(groupselections) if groupselections else None,
        )
        analyzer.run(start=skip_first_n_frames, step=step)
        values = analyzer.results.rmsd.copy()
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("Unexpected RMSD output format")
        result: dict[str, np.ndarray] = {
            "md_step": values[:, 0].astype(float),
            "time_ps": values[:, 1].astype(float),
            "rmsd": values[:, 2].astype(float),
        }
        if groupselections:
            for idx, label in enumerate(groupselections):
                col = 3 + idx
                if col < values.shape[1]:
                    result[f"rmsd_group_{label}"] = values[:, col].astype(float)
        return result


class RMSFAnalyzer(GeometryBase):
    def calculate(
        self,
        target_selection: str = "name CA",
        align_selection: str = "protein and backbone",
        ref_frame: int = 0,
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        universe = self._build_universe()
        align.AlignTraj(
            universe,
            universe,
            select=align_selection,
            ref_frame=ref_frame,
            in_memory=True,
        ).run(start=skip_first_n_frames, step=step)
        atoms = universe.select_atoms(target_selection)
        analyzer = rms.RMSF(atoms)
        analyzer.run(start=skip_first_n_frames, step=step)
        return atoms.resids.copy(), analyzer.results.rmsf.copy()

    def calculate_detailed(
        self,
        target_selection: str = "name CA",
        align_selection: str = "protein and backbone",
        ref_frame: int = 0,
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> dict[str, np.ndarray]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        universe = self._build_universe()
        align.AlignTraj(
            universe,
            universe,
            select=align_selection,
            ref_frame=ref_frame,
            in_memory=True,
        ).run(start=skip_first_n_frames, step=step)
        atoms = universe.select_atoms(target_selection)
        analyzer = rms.RMSF(atoms)
        analyzer.run(start=skip_first_n_frames, step=step)
        return {
            "resids": atoms.resids.copy().astype(int),
            "segids": atoms.segids.astype(str),
            "rmsf": analyzer.results.rmsf.copy().astype(float),
        }


class HelicityAnalyzer(GeometryBase):
    def _is_helical(self, residue) -> int:
        try:
            phi = residue.phi_selection().dihedral.value()
            psi = residue.psi_selection().dihedral.value()
        except Exception:
            return 0
        if (-100 < phi < -25) and (-80 < psi < -10):
            return 1
        return 0

    def calculate(
        self,
        amino_acids: Iterable[int | str] | None = None,
        segids: Sequence[str] = ("B", "R"),
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> dict[str, np.ndarray] | str:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        if amino_acids is None:
            return "Please provide an amino_acids list for helicity analysis."
        residues = [str(item) for item in amino_acids]
        if not residues:
            return "Please provide an amino_acids list for helicity analysis."
        universe = self._build_universe()
        residue_selection = " ".join(residues)
        selected = {
            segid: universe.select_atoms(f"segid {segid} and resid {residue_selection}").residues
            for segid in segids
        }
        result: dict[str, list[float]] = {"frame": []}
        for segid in segids:
            result[f"helicity_{segid}"] = []
        total = len(residues)
        for ts in universe.trajectory[skip_first_n_frames::step]:
            result["frame"].append(float(ts.frame))
            for segid in segids:
                residues_in_seg = selected[segid]
                helical = sum(self._is_helical(residue) for residue in residues_in_seg)
                result[f"helicity_{segid}"].append((helical / total) * 100.0)
        return {key: np.asarray(value, dtype=float) for key, value in result.items()}
