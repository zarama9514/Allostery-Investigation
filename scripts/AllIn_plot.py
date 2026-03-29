from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def build_residue_annotation_blocks(
    segids: Sequence[str] | np.ndarray,
    resids: Sequence[int] | np.ndarray,
    block_definitions: Sequence[Mapping[str, object]],
    include_unmatched: bool = False,
) -> list[dict[str, object]]:
    segid_arr = np.asarray(segids, dtype=str).reshape(-1)
    resid_arr = np.asarray(resids, dtype=int).reshape(-1)
    if segid_arr.size != resid_arr.size:
        raise ValueError("segids and resids must have the same length")
    if segid_arr.size == 0:
        return []
    assignments: list[dict[str, object] | None] = []
    for segid, resid in zip(segid_arr, resid_arr):
        matched: dict[str, object] | None = None
        for raw_def in block_definitions:
            def_segid = str(raw_def.get("segid", "")).strip()
            if def_segid != segid:
                continue
            start_resid = int(raw_def.get("start_resid", int(resid_arr.min())))
            end_resid = int(raw_def.get("end_resid", int(resid_arr.max())))
            if start_resid <= int(resid) <= end_resid:
                matched = {
                    "segid": segid,
                    "label": str(raw_def.get("label", segid)),
                    "color": str(raw_def.get("color", "#d9d9d9")),
                    "alpha": float(raw_def.get("alpha", 0.12)),
                }
                break
        if matched is None and include_unmatched:
            matched = {
                "segid": segid,
                "label": segid,
                "color": "#d9d9d9",
                "alpha": 0.08,
            }
        assignments.append(matched)
    blocks: list[dict[str, object]] = []
    current = assignments[0]
    start_idx = 0
    for idx in range(1, len(assignments) + 1):
        candidate = assignments[idx] if idx < len(assignments) else None
        if candidate != current:
            if current is not None:
                blocks.append(
                    {
                        "segid": current["segid"],
                        "label": current["label"],
                        "color": current["color"],
                        "alpha": current["alpha"],
                        "start_idx": int(start_idx),
                        "end_idx": int(idx - 1),
                        "start_resid": int(resid_arr[start_idx]),
                        "end_resid": int(resid_arr[idx - 1]),
                    }
                )
            current = candidate
            start_idx = idx
    return blocks


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

    @staticmethod
    def _build_block_ticks(
        residue_numbers: np.ndarray,
        residue_annotation_blocks: Sequence[Mapping[str, object]],
        max_ticks_total: int = 12,
    ) -> tuple[np.ndarray, list[str]]:
        if not residue_annotation_blocks:
            return residue_numbers.astype(float), [str(int(x)) for x in residue_numbers]
        blocks = list(residue_annotation_blocks)
        if len(blocks) > max_ticks_total:
            selected_idx = np.linspace(0, len(blocks) - 1, num=max_ticks_total, dtype=int)
            selected_idx = np.unique(selected_idx)
            ticks = np.asarray(
                [
                    int((int(blocks[idx]["start_idx"]) + int(blocks[idx]["end_idx"])) // 2)
                    for idx in selected_idx
                ],
                dtype=float,
            )
            labels = [str(int(residue_numbers[int(tick)])) for tick in ticks.astype(int)]
            return ticks, labels
        ticks: list[int] = []
        labels: list[str] = []
        per_block = max(1, max_ticks_total // max(1, len(blocks)))
        for block in blocks:
            start_idx = int(block["start_idx"])
            end_idx = int(block["end_idx"])
            span = end_idx - start_idx + 1
            if span <= per_block:
                idx = np.arange(start_idx, end_idx + 1, dtype=int)
            elif per_block == 1:
                idx = np.asarray([(start_idx + end_idx) // 2], dtype=int)
            else:
                idx = np.linspace(start_idx, end_idx, num=per_block, dtype=int)
                idx = np.unique(idx)
            for item in idx:
                ticks.append(int(item))
                labels.append(str(int(residue_numbers[item])))
        unique_ticks: list[int] = []
        unique_labels: list[str] = []
        seen: set[int] = set()
        for tick, label in zip(ticks, labels):
            if tick in seen:
                continue
            seen.add(tick)
            unique_ticks.append(tick)
            unique_labels.append(label)
        return np.asarray(unique_ticks, dtype=float), unique_labels

    @staticmethod
    def _annotation_legend_entries(
        residue_annotation_blocks: Sequence[Mapping[str, object]],
        prefix: str = "",
    ) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for block in residue_annotation_blocks:
            label = (
                f"{prefix}{str(block['label'])} "
                f"({int(block['start_resid'])}-{int(block['end_resid'])})"
            )
            color = str(block.get("color", "#d9d9d9"))
            key = (label, color)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "label": label,
                    "color": color,
                    "alpha": float(block.get("alpha", 0.12)),
                }
            )
        return entries

    def _apply_annotation_legend(
        self,
        ax: plt.Axes,
        entries: Sequence[Mapping[str, object]],
        legend_title: str = "Annotated Regions",
        max_cols: int = 4,
        y_offset: float = -0.18,
    ) -> int:
        if not entries:
            return 0
        cols = max(1, min(int(max_cols), len(entries)))
        handles = [
            Patch(
                facecolor=str(entry["color"]),
                edgecolor="black",
                linewidth=0.6,
                alpha=max(0.85, float(entry.get("alpha", 0.12))),
                label=str(entry["label"]),
            )
            for entry in entries
        ]
        existing_legend = ax.get_legend()
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset),
            ncol=cols,
            frameon=False,
            fontsize=8,
            title=legend_title,
            title_fontsize=9,
            borderaxespad=0.0,
        )
        if existing_legend is not None:
            ax.add_artist(existing_legend)
        return int(math.ceil(len(entries) / float(cols)))

    def _apply_x_annotation_blocks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        residue_annotation_blocks: Sequence[Mapping[str, object]] | None,
    ) -> None:
        if not residue_annotation_blocks:
            return
        for block in residue_annotation_blocks:
            start_idx = float(block["start_idx"]) - 0.5
            end_idx = float(block["end_idx"]) + 0.5
            ax.axvspan(
                start_idx,
                end_idx,
                color=str(block.get("color", "#d9d9d9")),
                alpha=float(block.get("alpha", 0.12)),
                linewidth=0,
            )
            ax.axvline(end_idx, color="black", linewidth=0.4, alpha=0.25)

    def _apply_xy_annotation_blocks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        x_annotation_blocks: Sequence[Mapping[str, object]] | None,
        y_annotation_blocks: Sequence[Mapping[str, object]] | None,
    ) -> None:
        if x_annotation_blocks:
            for block in x_annotation_blocks:
                start_idx = float(block["start_idx"]) - 0.5
                end_idx = float(block["end_idx"]) + 0.5
                ax.axvspan(
                    start_idx,
                    end_idx,
                    color=str(block.get("color", "#d9d9d9")),
                    alpha=float(block.get("alpha", 0.07)),
                    linewidth=0,
                )
                ax.axvline(end_idx, color="black", linewidth=0.4, alpha=0.25)
        if y_annotation_blocks:
            for block in y_annotation_blocks:
                start_idx = float(block["start_idx"]) - 0.5
                end_idx = float(block["end_idx"]) + 0.5
                ax.axhspan(
                    start_idx,
                    end_idx,
                    color=str(block.get("color", "#d9d9d9")),
                    alpha=float(block.get("alpha", 0.07)),
                    linewidth=0,
                )
                ax.axhline(end_idx, color="black", linewidth=0.4, alpha=0.25)


class RMSProfilePlotter(PlotBase):
    def _apply_rmsf_axis_blocks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        residue_numbers: np.ndarray,
        residue_annotation_blocks: Sequence[Mapping[str, object]] | None,
    ) -> None:
        if not residue_annotation_blocks:
            return
        ticks, labels = self._build_block_ticks(residue_numbers, residue_annotation_blocks)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        self._apply_x_annotation_blocks(fig, ax, residue_annotation_blocks)
        legend_rows = self._apply_annotation_legend(
            ax,
            self._annotation_legend_entries(residue_annotation_blocks),
            y_offset=-0.19,
        )
        fig.subplots_adjust(bottom=min(0.42, 0.16 + 0.05 * legend_rows), top=0.88)

    def _plot_profile(
        self,
        x_values: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray,
        x_label: str,
        title: str,
        y_label: str,
        color: str,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        x = self._as_1d(x_values, "x_values")
        y = self._as_1d(values, "values")
        self._validate_same_length(x, y, "x_values", "values")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, y, color=color, linewidth=1.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, pad=12)
        ax.grid(alpha=0.3)
        self._save(fig, save_path)
        return fig, ax

    def plot_rmsd(
        self,
        md_steps: Sequence[float] | np.ndarray,
        rmsd_values: Sequence[float] | np.ndarray,
        title: str = "RMSD vs Molecular Dynamics Step",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        return self._plot_profile(
            x_values=md_steps,
            values=rmsd_values,
            x_label="Molecular Dynamics Step",
            title=title,
            y_label="RMSD, Å.",
            color="steelblue",
            save_path=save_path,
        )

    def plot_rmsd_from_geometry_output(
        self,
        geometry_output: Mapping[str, Sequence[float] | np.ndarray],
        title: str = "RMSD vs Molecular Dynamics Step",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        if "md_step" not in geometry_output or "rmsd" not in geometry_output:
            raise ValueError("geometry_output must contain 'md_step' and 'rmsd'")
        return self.plot_rmsd(
            md_steps=geometry_output["md_step"],
            rmsd_values=geometry_output["rmsd"],
            title=title,
            save_path=save_path,
        )

    def plot_rmsf(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        rmsf_values: Sequence[float] | np.ndarray,
        title: str = "RMSF Profile",
        residue_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        x = self._as_1d(amino_acid_numbers, "amino_acid_numbers").astype(int)
        y = self._as_1d(rmsf_values, "rmsf_values")
        self._validate_same_length(x, y, "amino_acid_numbers", "rmsf_values")
        fig, ax = plt.subplots(figsize=self.figsize)
        x_plot = np.arange(x.size, dtype=float) if residue_annotation_blocks else x.astype(float)
        ax.plot(x_plot, y, color="indianred", linewidth=1.4)
        ax.set_xlabel("Amino Acid Number", labelpad=4)
        ax.set_ylabel("RMSF, Å.")
        ax.set_title(title, pad=12)
        ax.grid(alpha=0.3)
        self._apply_rmsf_axis_blocks(fig, ax, x, residue_annotation_blocks)
        self._save(fig, save_path)
        return fig, ax

    def plot_rmsf_overlay(
        self,
        amino_acid_numbers: Sequence[int] | np.ndarray,
        rmsf_values_a: Sequence[float] | np.ndarray,
        rmsf_values_b: Sequence[float] | np.ndarray,
        label_a: str = "A",
        label_b: str = "B",
        color_a: str = "blue",
        color_b: str = "red",
        title: str = "RMSF Comparison",
        residue_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        x = self._as_1d(amino_acid_numbers, "amino_acid_numbers").astype(int)
        y_a = self._as_1d(rmsf_values_a, "rmsf_values_a")
        y_b = self._as_1d(rmsf_values_b, "rmsf_values_b")
        self._validate_same_length(x, y_a, "amino_acid_numbers", "rmsf_values_a")
        self._validate_same_length(x, y_b, "amino_acid_numbers", "rmsf_values_b")
        fig, ax = plt.subplots(figsize=self.figsize)
        x_plot = np.arange(x.size, dtype=float) if residue_annotation_blocks else x.astype(float)
        ax.plot(x_plot, y_a, color=color_a, linewidth=1.6, label=label_a)
        ax.plot(x_plot, y_b, color=color_b, linewidth=1.6, label=label_b)
        ax.set_xlabel("Amino Acid Number", labelpad=4)
        ax.set_ylabel("RMSF, Å.")
        ax.set_title(title, pad=12)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", title="System")
        self._apply_rmsf_axis_blocks(fig, ax, x, residue_annotation_blocks)
        self._save(fig, save_path)
        return fig, ax


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
        md_steps: Sequence[float] | np.ndarray,
        rmsd_values_a: Sequence[float] | np.ndarray,
        rmsd_values_b: Sequence[float] | np.ndarray,
        title: str = "Absolute RMSD Difference vs Molecular Dynamics Step",
        save_path: str | None = None,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        diff = self.calculate_absolute_difference(rmsd_values_a, rmsd_values_b)
        x = self._as_1d(md_steps, "md_steps")
        self._validate_same_length(x, diff, "md_steps", "rmsd difference")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, diff, color="darkorchid", linewidth=1.4, label="|Delta RMSD|")
        ax.set_xlabel("Molecular Dynamics Step")
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
        residue_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        save_path: str | None = None,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        diff = self.calculate_absolute_difference(rmsf_values_a, rmsf_values_b)
        x = self._as_1d(amino_acid_numbers, "amino_acid_numbers")
        self._validate_same_length(x, diff, "amino_acid_numbers", "rmsf difference")
        fig, ax = plt.subplots(figsize=self.figsize)
        x_plot = np.arange(x.size, dtype=float) if residue_annotation_blocks else x
        ax.plot(x_plot, diff, color="teal", linewidth=1.4, label="|Delta RMSF|")
        ax.set_xlabel("Amino Acid Number", labelpad=4)
        ax.set_ylabel("|Delta RMSF|, Å.")
        ax.set_title(title, pad=12)
        ax.legend()
        ax.grid(alpha=0.3)
        if residue_annotation_blocks:
            ticks, labels = self._build_block_ticks(x.astype(int), residue_annotation_blocks)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            self._apply_x_annotation_blocks(fig, ax, residue_annotation_blocks)
            legend_rows = self._apply_annotation_legend(
                ax,
                self._annotation_legend_entries(residue_annotation_blocks),
                y_offset=-0.19,
            )
            fig.subplots_adjust(bottom=min(0.42, 0.18 + 0.05 * legend_rows), top=0.88)
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


class DCCMPlotter(PlotBase):
    @staticmethod
    def _validate_matrix(dccm: np.ndarray) -> np.ndarray:
        arr = np.asarray(dccm, dtype=float)
        if arr.ndim != 2:
            raise ValueError("dccm must be a 2D matrix")
        if arr.size == 0:
            raise ValueError("dccm is empty")
        return arr

    @staticmethod
    def _build_residue_ticks(resids: np.ndarray, max_ticks: int = 12) -> tuple[np.ndarray, list[str]]:
        if resids.size <= max_ticks:
            idx = np.arange(resids.size)
        else:
            step = max(1, resids.size // max_ticks)
            idx = np.arange(0, resids.size, step)
            if idx[-1] != resids.size - 1:
                idx = np.append(idx, resids.size - 1)
        labels = [str(int(resids[i])) for i in idx]
        return idx, labels

    @staticmethod
    def _chain_map_text(chain_ranges: Sequence[Mapping[str, int | str]], axis_name: str) -> str:
        if not chain_ranges:
            return f"{axis_name}: no chain metadata"
        parts: list[str] = []
        for row in chain_ranges:
            segid = str(row["segid"])
            start_resid = int(row["start_resid"])
            end_resid = int(row["end_resid"])
            parts.append(f"{segid}:{start_resid}-{end_resid}")
        return f"{axis_name}: " + "; ".join(parts)

    def plot_dccm(
        self,
        dccm: Sequence[Sequence[float]] | np.ndarray,
        x_resids: Sequence[int] | np.ndarray,
        y_resids: Sequence[int] | np.ndarray,
        x_chain_ranges: Sequence[Mapping[str, int | str]] | None = None,
        y_chain_ranges: Sequence[Mapping[str, int | str]] | None = None,
        x_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        y_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        title: str = "DCCM Heatmap",
        cmap: str = "RdBu_r",
        vmin: float = -1.0,
        vmax: float = 1.0,
        colorbar_label: str = "Correlation",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        matrix = self._validate_matrix(np.asarray(dccm, dtype=float))
        x = self._as_1d(x_resids, "x_resids").astype(int)
        y = self._as_1d(y_resids, "y_resids").astype(int)
        if matrix.shape[1] != x.size:
            raise ValueError("dccm columns must match x_resids length")
        if matrix.shape[0] != y.size:
            raise ValueError("dccm rows must match y_resids length")
        fig, ax = plt.subplots(figsize=self.figsize)
        image = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label)
        x_tick_idx, x_tick_labels = self._build_residue_ticks(x)
        y_tick_idx, y_tick_labels = self._build_residue_ticks(y)
        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        ax.set_yticks(y_tick_idx)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Residue Number (X)")
        ax.set_ylabel("Residue Number (Y)")
        ax.set_title(title, pad=12)
        if x_annotation_blocks is None and x_chain_ranges:
            for row in x_chain_ranges[:-1]:
                ax.axvline(int(row["end_idx"]) + 0.5, color="black", linewidth=0.5, alpha=0.6)
            x_centers = [
                (int(row["start_idx"]) + int(row["end_idx"])) / 2.0
                for row in x_chain_ranges
            ]
            x_labels = [str(row["segid"]) for row in x_chain_ranges]
            top_ax = ax.twiny()
            top_ax.set_xlim(ax.get_xlim())
            top_ax.set_xticks(x_centers)
            top_ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
            top_ax.set_xlabel("Chain (X)")
        if y_annotation_blocks is None and y_chain_ranges:
            for row in y_chain_ranges[:-1]:
                ax.axhline(int(row["end_idx"]) + 0.5, color="black", linewidth=0.5, alpha=0.6)
            y_centers = [
                (int(row["start_idx"]) + int(row["end_idx"])) / 2.0
                for row in y_chain_ranges
            ]
            y_labels = [str(row["segid"]) for row in y_chain_ranges]
            right_ax = ax.twinx()
            right_ax.set_ylim(ax.get_ylim())
            right_ax.set_yticks(y_centers)
            right_ax.set_yticklabels(y_labels, fontsize=8)
            right_ax.set_ylabel("Chain (Y)")
        self._apply_xy_annotation_blocks(fig, ax, x_annotation_blocks, y_annotation_blocks)
        if x_annotation_blocks or y_annotation_blocks:
            legend_entries = self._annotation_legend_entries(x_annotation_blocks or [], prefix="X: ")
            legend_entries.extend(self._annotation_legend_entries(y_annotation_blocks or [], prefix="Y: "))
            legend_rows = self._apply_annotation_legend(
                ax,
                legend_entries,
                legend_title="Annotated Regions",
                max_cols=3,
                y_offset=-0.16,
            )
            fig.subplots_adjust(bottom=min(0.46, 0.16 + 0.06 * legend_rows), top=0.90)
        if (x_chain_ranges or y_chain_ranges) and not (x_annotation_blocks or y_annotation_blocks):
            x_map = self._chain_map_text(x_chain_ranges or [], "X")
            y_map = self._chain_map_text(y_chain_ranges or [], "Y")
            fig.text(0.5, -0.02, f"{x_map}\n{y_map}", ha="center", va="top", fontsize=8)
        self._save(fig, save_path)
        return fig, ax

    def calculate_normalized_difference(
        self,
        dccm_a: Sequence[Sequence[float]] | np.ndarray,
        dccm_b: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        a = self._validate_matrix(np.asarray(dccm_a, dtype=float))
        b = self._validate_matrix(np.asarray(dccm_b, dtype=float))
        if a.shape != b.shape:
            raise ValueError("Both DCCM heatmaps must have the same shape")
        return np.clip((a - b) / 2.0, -1.0, 1.0)

    def plot_dccm_difference(
        self,
        dccm_a: Sequence[Sequence[float]] | np.ndarray,
        dccm_b: Sequence[Sequence[float]] | np.ndarray,
        x_resids: Sequence[int] | np.ndarray,
        y_resids: Sequence[int] | np.ndarray,
        x_chain_ranges: Sequence[Mapping[str, int | str]] | None = None,
        y_chain_ranges: Sequence[Mapping[str, int | str]] | None = None,
        x_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        y_annotation_blocks: Sequence[Mapping[str, object]] | None = None,
        title: str = "Normalized DCCM Difference Heatmap",
        save_path: str | None = None,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        diff = self.calculate_normalized_difference(dccm_a, dccm_b)
        fig, ax = self.plot_dccm(
            dccm=diff,
            x_resids=x_resids,
            y_resids=y_resids,
            x_chain_ranges=x_chain_ranges,
            y_chain_ranges=y_chain_ranges,
            x_annotation_blocks=x_annotation_blocks,
            y_annotation_blocks=y_annotation_blocks,
            title=title,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            colorbar_label="(H1 - H2) / 2",
            save_path=save_path,
        )
        return diff, fig, ax

    def plot_from_community_output(
        self,
        community_output: Mapping[str, object],
        title: str = "DCCM Heatmap",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        required = ("dccm", "x_resids", "y_resids")
        for key in required:
            if key not in community_output:
                raise ValueError(f"community_output must contain '{key}'")
        return self.plot_dccm(
            dccm=np.asarray(community_output["dccm"], dtype=float),
            x_resids=np.asarray(community_output["x_resids"], dtype=int),
            y_resids=np.asarray(community_output["y_resids"], dtype=int),
            x_chain_ranges=community_output.get("x_chain_ranges"),
            y_chain_ranges=community_output.get("y_chain_ranges"),
            x_annotation_blocks=community_output.get("x_annotation_blocks"),
            y_annotation_blocks=community_output.get("y_annotation_blocks"),
            title=title,
            save_path=save_path,
        )

    def plot_difference_from_community_outputs(
        self,
        community_output_1: Mapping[str, object],
        community_output_2: Mapping[str, object],
        title: str = "Normalized DCCM Difference Heatmap",
        save_path: str | None = None,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        required = ("dccm", "x_resids", "y_resids")
        for key in required:
            if key not in community_output_1 or key not in community_output_2:
                raise ValueError(f"Both outputs must contain '{key}'")
        x1 = np.asarray(community_output_1["x_resids"], dtype=int)
        y1 = np.asarray(community_output_1["y_resids"], dtype=int)
        x2 = np.asarray(community_output_2["x_resids"], dtype=int)
        y2 = np.asarray(community_output_2["y_resids"], dtype=int)
        if x1.shape != x2.shape or y1.shape != y2.shape:
            raise ValueError("Both outputs must have matching residue axes")
        if not np.array_equal(x1, x2) or not np.array_equal(y1, y2):
            raise ValueError("Residue numbering must match between both outputs")
        return self.plot_dccm_difference(
            dccm_a=np.asarray(community_output_1["dccm"], dtype=float),
            dccm_b=np.asarray(community_output_2["dccm"], dtype=float),
            x_resids=x1,
            y_resids=y1,
            x_chain_ranges=community_output_1.get("x_chain_ranges"),
            y_chain_ranges=community_output_1.get("y_chain_ranges"),
            x_annotation_blocks=community_output_1.get("x_annotation_blocks"),
            y_annotation_blocks=community_output_1.get("y_annotation_blocks"),
            title=title,
            save_path=save_path,
        )


class HydrogenBondPlotter(PlotBase):
    @staticmethod
    def _read_contacts_csv(csv_path: str) -> list[dict[str, object]]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        rows: list[dict[str, object]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def _to_float(value: object, field_name: str) -> float:
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"Cannot parse float field '{field_name}': {value}") from exc

    def plot_top_contacts(
        self,
        contacts: Sequence[Mapping[str, object]] | None = None,
        csv_path: str | None = None,
        top_n: int = 15,
        lifetime_field: str = "lifetime_ps",
        label_field: str = "contact_residue_label",
        title: str = "Top Protein-Protein H-Bond Contacts by Lifetime",
        save_path: str | None = None,
    ) -> tuple[list[dict[str, object]], plt.Figure, plt.Axes]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if contacts is None:
            if not csv_path:
                raise ValueError("Provide contacts or csv_path")
            parsed = self._read_contacts_csv(csv_path)
        else:
            parsed = [dict(row) for row in contacts]
        if not parsed:
            raise ValueError("No contact rows available for plotting")
        for row in parsed:
            row[lifetime_field] = self._to_float(row.get(lifetime_field), lifetime_field)
            row[label_field] = str(row.get(label_field, "NA"))
        sorted_rows = sorted(parsed, key=lambda row: float(row[lifetime_field]), reverse=True)
        top_rows = sorted_rows[:top_n]
        top_rows_plot = list(reversed(top_rows))
        labels = [str(row[label_field]) for row in top_rows_plot]
        lifetimes = [float(row[lifetime_field]) for row in top_rows_plot]
        height = max(self.figsize[1], 0.45 * len(top_rows_plot) + 1.5)
        fig, ax = plt.subplots(figsize=(self.figsize[0], height))
        positions = np.arange(len(top_rows_plot))
        ax.barh(positions, lifetimes, color="darkcyan", alpha=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Lifetime (ps)")
        ax.set_ylabel("Residue Pair (Subunit)")
        if title:
            ax.set_title(title if top_n == 15 else f"Top {top_n} Protein-Protein H-Bond Contacts by Lifetime")
        ax.grid(axis="x", alpha=0.3)
        self._save(fig, save_path)
        return top_rows, fig, ax
