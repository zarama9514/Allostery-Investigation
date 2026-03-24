from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class PlotBase:
    def __init__(self, figsize: tuple[float, float] = (10.0, 5.0), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi

    @staticmethod
    def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} is empty")
        return arr

    @staticmethod
    def _validate_same_length(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
        if a.size != b.size:
            raise ValueError(f"{a_name} and {b_name} must have the same length")

    def _save(self, fig: plt.Figure, save_path: str | None) -> None:
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")


class RMSProfilePlotter(PlotBase):
    def _plot_profile(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        values: Sequence[float] | np.ndarray,
        title: str,
        y_label: str,
        color: str,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        x = self._as_1d(amino_acid_numbers, "amino_acid_numbers")
        y = self._as_1d(values, "values")
        self._validate_same_length(x, y, "amino_acid_numbers", "values")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, y, color=color, linewidth=1.4)
        ax.set_xlabel("Amino Acid Number")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        self._save(fig, save_path)
        return fig, ax

    def plot_rmsd(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        rmsd_values: Sequence[float] | np.ndarray,
        title: str = "RMSD Profile",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        return self._plot_profile(
            amino_acid_numbers=amino_acid_numbers,
            values=rmsd_values,
            title=title,
            y_label="RMSD, Å.",
            color="steelblue",
            save_path=save_path,
        )

    def plot_rmsf(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        rmsf_values: Sequence[float] | np.ndarray,
        title: str = "RMSF Profile",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        return self._plot_profile(
            amino_acid_numbers=amino_acid_numbers,
            values=rmsf_values,
            title=title,
            y_label="RMSF, Å.",
            color="indianred",
            save_path=save_path,
        )


class RMSDDifferencePlotter(PlotBase):
    def calculate_absolute_difference(
        self,
        rmsd_values_a: Sequence[float] | np.ndarray,
        rmsd_values_b: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        a = self._as_1d(rmsd_values_a, "rmsd_values_a")
        b = self._as_1d(rmsd_values_b, "rmsd_values_b")
        self._validate_same_length(a, b, "rmsd_values_a", "rmsd_values_b")
        return np.abs(a - b)

    def plot_difference(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        rmsd_values_a: Sequence[float] | np.ndarray,
        rmsd_values_b: Sequence[float] | np.ndarray,
        title: str = "Absolute RMSD Difference Profile",
        save_path: str | None = None,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        diff = self.calculate_absolute_difference(rmsd_values_a, rmsd_values_b)
        x = self._as_1d(amino_acid_numbers, "amino_acid_numbers")
        self._validate_same_length(x, diff, "amino_acid_numbers", "rmsd difference")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, diff, color="darkorchid", linewidth=1.4, label="|Delta RMSD|")
        ax.set_xlabel("Amino Acid Number")
        ax.set_ylabel("|Delta RMSD|, Å.")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, save_path)
        return diff, fig, ax


class RMSFDifferencePlotter(PlotBase):
    def calculate_absolute_difference(
        self,
        rmsf_values_a: Sequence[float] | np.ndarray,
        rmsf_values_b: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        a = self._as_1d(rmsf_values_a, "rmsf_values_a")
        b = self._as_1d(rmsf_values_b, "rmsf_values_b")
        self._validate_same_length(a, b, "rmsf_values_a", "rmsf_values_b")
        return np.abs(a - b)

    def plot_difference(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        rmsf_values_a: Sequence[float] | np.ndarray,
        rmsf_values_b: Sequence[float] | np.ndarray,
        title: str = "Absolute RMSF Difference Profile",
        save_path: str | None = None,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        diff = self.calculate_absolute_difference(rmsf_values_a, rmsf_values_b)
        x = self._as_1d(amino_acid_numbers, "amino_acid_numbers")
        self._validate_same_length(x, diff, "amino_acid_numbers", "rmsf difference")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, diff, color="teal", linewidth=1.4, label="|Delta RMSF|")
        ax.set_xlabel("Amino Acid Number")
        ax.set_ylabel("|Delta RMSF|, Å.")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, save_path)
        return diff, fig, ax


class HelicityPlotter(PlotBase):
    def plot(
        self,
        md_steps: Sequence[float] | np.ndarray,
        helicity_series: Mapping[str, Sequence[float] | np.ndarray],
        title: str = "Helicity Profile",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        x = self._as_1d(md_steps, "md_steps")
        if not helicity_series:
            raise ValueError("helicity_series is empty")
        fig, ax = plt.subplots(figsize=self.figsize)
        for label, values in helicity_series.items():
            y = self._as_1d(values, f"helicity_series[{label}]")
            self._validate_same_length(x, y, "md_steps", f"helicity_series[{label}]")
            mean_value = float(np.mean(y))
            ax.plot(x, y, linewidth=1.3, label=f"{label} | mean helicity (%): {mean_value:.2f}")
        ax.set_xlabel("Molecular Dynamics Step")
        ax.set_ylabel("Helicity (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, save_path)
        return fig, ax

    def plot_from_geometry_output(
        self,
        geometry_output: Mapping[str, Sequence[float] | np.ndarray],
        title: str = "Helicity Profile",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        if "frame" not in geometry_output:
            raise ValueError("geometry_output must contain 'frame'")
        series = {
            key.replace("helicity_", "", 1): value
            for key, value in geometry_output.items()
            if key.startswith("helicity_")
        }
        if not series:
            raise ValueError("geometry_output must contain at least one key starting with 'helicity_'")
        return self.plot(
            md_steps=geometry_output["frame"],
            helicity_series=series,
            title=title,
            save_path=save_path,
        )
