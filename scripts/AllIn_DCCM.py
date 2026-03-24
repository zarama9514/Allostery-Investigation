from __future__ import annotations

from typing import Sequence

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align


class DCCMAnalyzer:
    def __init__(
        self,
        psf1: str,
        dcd1: str | Sequence[str],
        psf2: str | None = None,
        dcd2: str | Sequence[str] | None = None,
    ):
        self.psf1 = psf1
        self.dcd1 = dcd1
        self.psf2 = psf2
        self.dcd2 = dcd2

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

    @staticmethod
    def _build_universe(psf: str, dcd: str | Sequence[str]) -> mda.Universe:
        if isinstance(dcd, str):
            return mda.Universe(psf, dcd)
        return mda.Universe(psf, list(dcd))

    @staticmethod
    def _chain_ranges(segids: np.ndarray, resids: np.ndarray) -> list[dict[str, int | str]]:
        if segids.size == 0:
            return []
        ranges: list[dict[str, int | str]] = []
        start = 0
        current = str(segids[0])
        for idx in range(1, segids.size):
            if str(segids[idx]) != current:
                ranges.append(
                    {
                        "segid": current,
                        "start_idx": int(start),
                        "end_idx": int(idx - 1),
                        "start_resid": int(resids[start]),
                        "end_resid": int(resids[idx - 1]),
                    }
                )
                start = idx
                current = str(segids[idx])
        ranges.append(
            {
                "segid": current,
                "start_idx": int(start),
                "end_idx": int(segids.size - 1),
                "start_resid": int(resids[start]),
                "end_resid": int(resids[segids.size - 1]),
            }
        )
        return ranges

    def _extract_coords(
        self,
        universe: mda.Universe,
        selection: str,
        skip_first_n_frames: int,
        step: int,
        align_selection: str | None,
        align_ref_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if align_selection:
            align.AlignTraj(
                universe,
                universe,
                select=align_selection,
                ref_frame=align_ref_frame,
                in_memory=True,
            ).run(start=skip_first_n_frames, step=step)
        atoms = universe.select_atoms(selection)
        if atoms.n_atoms == 0:
            raise ValueError(f"Selection has no atoms: {selection}")
        coords = [atoms.positions.copy() for _ in universe.trajectory[skip_first_n_frames::step]]
        if not coords:
            raise ValueError("No frames selected after skipping")
        return np.asarray(coords, dtype=float), atoms.resids.copy(), atoms.segids.copy()

    @staticmethod
    def _dccm(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        n_frames = min(coords1.shape[0], coords2.shape[0])
        c1 = coords1[:n_frames]
        c2 = coords2[:n_frames]
        d1 = c1 - c1.mean(axis=0)
        d2 = c2 - c2.mean(axis=0)
        cov = np.einsum("t i d, t j d -> i j", d1, d2) / float(n_frames)
        var1 = np.einsum("t i d, t i d -> i", d1, d1) / float(n_frames)
        var2 = np.einsum("t j d, t j d -> j", d2, d2) / float(n_frames)
        den = np.sqrt(np.outer(var1, var2))
        out = np.zeros_like(cov, dtype=float)
        np.divide(cov, den, out=out, where=den > 0.0)
        return np.clip(out, -1.0, 1.0)

    def calculate(
        self,
        selection1: str = "protein and name CA",
        selection2: str | None = None,
        align_selection1: str = "protein and backbone",
        align_selection2: str | None = None,
        align_ref_frame: int = 0,
        step: int = 1,
        skip_first_n_frames: int = 0,
    ) -> dict[str, np.ndarray | list[dict[str, int | str]] | int]:
        step = self._validate_step(step)
        skip_first_n_frames = self._validate_skip(skip_first_n_frames)
        if (self.psf2 is None) != (self.dcd2 is None):
            raise ValueError("psf2 and dcd2 must be provided together")
        psf2 = self.psf1 if self.psf2 is None else self.psf2
        dcd2 = self.dcd1 if self.dcd2 is None else self.dcd2
        selection2 = selection1 if selection2 is None else selection2
        align_selection2 = align_selection1 if align_selection2 is None else align_selection2
        u1 = self._build_universe(self.psf1, self.dcd1)
        u2 = self._build_universe(psf2, dcd2)
        coords1, resids1, segids1 = self._extract_coords(
            universe=u1,
            selection=selection1,
            skip_first_n_frames=skip_first_n_frames,
            step=step,
            align_selection=align_selection1,
            align_ref_frame=align_ref_frame,
        )
        coords2, resids2, segids2 = self._extract_coords(
            universe=u2,
            selection=selection2,
            skip_first_n_frames=skip_first_n_frames,
            step=step,
            align_selection=align_selection2,
            align_ref_frame=align_ref_frame,
        )
        dccm = self._dccm(coords1, coords2)
        return {
            "dccm": dccm,
            "x_resids": resids1.astype(int),
            "y_resids": resids2.astype(int),
            "x_segids": segids1.astype(str),
            "y_segids": segids2.astype(str),
            "x_chain_ranges": self._chain_ranges(segids1.astype(str), resids1.astype(int)),
            "y_chain_ranges": self._chain_ranges(segids2.astype(str), resids2.astype(int)),
            "n_frames_used": int(min(coords1.shape[0], coords2.shape[0])),
        }
