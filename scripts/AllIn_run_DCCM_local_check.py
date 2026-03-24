from __future__ import annotations

from pathlib import Path

from AllIn_DCCM import DCCMAnalyzer
from AllIn_plot import DCCMPlotter


def main() -> None:
    psf_path = r"C:\Users\Daniil\IT_projects\Mika_project\results\A\step5_input.psf"
    dcd_path = r"C:\Users\Daniil\IT_projects\Mika_project\results\A\step7_productionRUN5.nowat.dcd"
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "dccm_heatmap_A_RUN5.png"

    analyzer = DCCMAnalyzer(psf1=psf_path, dcd1=dcd_path)
    result = analyzer.calculate(
        selection1="(segid B or segid R) and name CA",
        align_selection1="(segid B or segid R) and backbone",
        step=100,
        skip_first_n_frames=100,
    )

    plotter = DCCMPlotter(figsize=(10, 9), dpi=300)
    plotter.plot_from_community_output(
        result,
        title="DCCM Heatmap (System A, RUN5)",
        save_path=str(out_plot),
    )

    print(f"plot={out_plot}")
    print(f"matrix_shape={result['dccm'].shape}")
    print(f"frames_used={result['n_frames_used']}")


if __name__ == "__main__":
    main()
