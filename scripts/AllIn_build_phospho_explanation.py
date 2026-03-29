from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PairSpec:
    system: str
    pair: str
    label: str
    sasa_prefix: str


PAIR_SPECS = (
    PairSpec("A", "T_to_A", "A:T_to_A", "T"),
    PairSpec("B", "T_to_A", "B:T_to_A", "T"),
    PairSpec("B", "L_to_C", "B:L_to_C", "L"),
    PairSpec("F670G", "T_to_A", "F670G:T_to_A", "T"),
    PairSpec("I669G", "T_to_A", "I669G:T_to_A", "T"),
    PairSpec("R668G", "T_to_A", "R668G:T_to_A", "T"),
)

KEY_RESIDS = (857, 859, 860)
KEY_LABELS = ("pS857", "pS859", "pT860")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def select_sasa_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        label = str(row["label"])
        if label == "TOTAL":
            out["TOTAL"] = row
        elif "SEP857" in label:
            out["pS857"] = row
        elif "SEP859" in label:
            out["pS859"] = row
        elif "TPO860" in label:
            out["pT860"] = row
    return out


def best_rows_by_resid(rows: list[dict[str, str]], resid_field: str) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for resid in KEY_RESIDS:
        subset = [row for row in rows if int(row[resid_field]) == resid]
        if subset:
            out[resid] = max(subset, key=lambda row: float(row["occupancy_percent"]))
    return out


def build_metrics(results_root: Path) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for spec in PAIR_SPECS:
        pair_root = results_root / spec.system / spec.pair
        sasa_rows = read_csv_rows(pair_root / "01_sasa" / f"{spec.sasa_prefix}_phospho_sasa_summary.csv")
        contact_rows = read_csv_rows(pair_root / "02_tail_n_domain_contacts" / "tail_n_domain_contacts.csv")
        salt_rows = read_csv_rows(pair_root / "03_salt_bridges" / "salt_bridges_residue_level.csv")
        sasa_map = select_sasa_rows(sasa_rows)
        best_contact = best_rows_by_resid(contact_rows, resid_field="tail_resid")
        best_salt = best_rows_by_resid(salt_rows, resid_field="acceptor_resid")
        row: dict[str, object] = {"system": spec.system, "pair": spec.pair, "label": spec.label}
        for site in ("pS857", "pS859", "pT860", "TOTAL"):
            sasa_row = sasa_map[site]
            row[f"{site}_mean_sasa"] = float(sasa_row["mean_sasa_angstrom2"])
            row[f"{site}_median_sasa"] = float(sasa_row["median_sasa_angstrom2"])
        for resid, site in zip(KEY_RESIDS, KEY_LABELS, strict=False):
            contact = best_contact.get(resid)
            salt = best_salt.get(resid)
            row[f"{site}_contact_partner"] = "" if contact is None else str(contact["arrestin_residue_label"])
            row[f"{site}_contact_occupancy"] = 0.0 if contact is None else float(contact["occupancy_percent"])
            row[f"{site}_contact_lifetime_ps"] = 0.0 if contact is None else float(contact["lifetime_ps"])
            row[f"{site}_salt_partner"] = "" if salt is None else str(salt["donor_residue_label"])
            row[f"{site}_salt_occupancy"] = 0.0 if salt is None else float(salt["occupancy_percent"])
            row[f"{site}_salt_lifetime_ps"] = 0.0 if salt is None else float(salt["lifetime_ps"])
        metrics.append(row)
    return metrics


def write_metrics_csv(metrics: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "pair",
        "label",
        "pS857_mean_sasa",
        "pS859_mean_sasa",
        "pT860_mean_sasa",
        "TOTAL_mean_sasa",
        "pS857_median_sasa",
        "pS859_median_sasa",
        "pT860_median_sasa",
        "TOTAL_median_sasa",
        "pS857_contact_partner",
        "pS857_contact_occupancy",
        "pS857_contact_lifetime_ps",
        "pS859_contact_partner",
        "pS859_contact_occupancy",
        "pS859_contact_lifetime_ps",
        "pT860_contact_partner",
        "pT860_contact_occupancy",
        "pT860_contact_lifetime_ps",
        "pS857_salt_partner",
        "pS857_salt_occupancy",
        "pS857_salt_lifetime_ps",
        "pS859_salt_partner",
        "pS859_salt_occupancy",
        "pS859_salt_lifetime_ps",
        "pT860_salt_partner",
        "pT860_salt_occupancy",
        "pT860_salt_lifetime_ps",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_heatmap(ax, data: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, cmap: str) -> None:
    image = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if np.isfinite(value) and value > np.nanmax(data) * 0.55 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_summary_panels(metrics: list[dict[str, object]], output_path: Path) -> None:
    row_labels = [str(row["label"]) for row in metrics]
    sasa_cols = ["pS857_mean_sasa", "pS859_mean_sasa", "pT860_mean_sasa", "TOTAL_mean_sasa"]
    contact_cols = ["pS857_contact_occupancy", "pS859_contact_occupancy", "pT860_contact_occupancy"]
    salt_cols = ["pS857_salt_occupancy", "pS859_salt_occupancy", "pT860_salt_occupancy"]
    sasa = np.asarray([[float(row[col]) for col in sasa_cols] for row in metrics], dtype=float)
    contact = np.asarray([[float(row[col]) for col in contact_cols] for row in metrics], dtype=float)
    salt = np.asarray([[float(row[col]) for col in salt_cols] for row in metrics], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), constrained_layout=True)
    render_heatmap(axes[0], sasa, row_labels, ["pS857", "pS859", "pT860", "TOTAL"], "Mean SASA (Å²)", "YlGnBu")
    render_heatmap(axes[1], contact, row_labels, ["pS857", "pS859", "pT860"], "Best Tail-Contact Occupancy (%)", "OrRd")
    render_heatmap(axes[2], salt, row_labels, ["pS857", "pS859", "pT860"], "Best Salt-Bridge Occupancy (%)", "PuRd")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_a_minus_mutant_sasa(metrics: list[dict[str, object]], output_path: Path) -> None:
    metric_map = {str(row["system"]): row for row in metrics if str(row["pair"]) == "T_to_A"}
    reference = metric_map["A"]
    mutants = ["F670G", "I669G", "R668G"]
    cols = ["pS857_mean_sasa", "pS859_mean_sasa", "pT860_mean_sasa", "TOTAL_mean_sasa"]
    labels = ["pS857", "pS859", "pT860", "TOTAL"]
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"F670G": "#1d3557", "I669G": "#c1121f", "R668G": "#2a9d8f"}
    for idx, mutant in enumerate(mutants):
        target = metric_map[mutant]
        delta = [float(reference[col]) - float(target[col]) for col in cols]
        ax.bar(x + (idx - 1) * width, delta, width=width, label=f"A - {mutant}", color=colors[mutant], alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ mean SASA, Å²")
    ax.set_title("Signed Mean SASA Differences: A - Mutant")
    ax.legend(title="Comparison")
    ax.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(metrics: list[dict[str, object]]) -> str:
    row = {str(item["label"]): item for item in metrics}

    def fmt(value: float) -> str:
        return f"{value:.1f}"

    a = row["A:T_to_A"]
    bt = row["B:T_to_A"]
    bl = row["B:L_to_C"]
    f670g = row["F670G:T_to_A"]
    i669g = row["I669G:T_to_A"]
    r668g = row["R668G:T_to_A"]

    lines: list[str] = []
    lines.append("This folder contains an interpretation of the phospho-tail coupling results relative to the mGlu3-betaarr1 paper.")
    lines.append("")
    lines.append("Paper anchor")
    lines.append("- The article reports that pS857, pS859, and pT860 on the mGlu3 C-tail engage the betaarr1 N-domain.")
    lines.append("- In the dynamic data, the strongest support for the paper is the combination of low SASA, high tail-contact occupancy, and high salt-bridge occupancy for these same three sites.")
    lines.append("")
    lines.append("What to read first")
    lines.append("- `phospho_key_site_summary.png`: one-page heatmap view of SASA, best contact occupancy, and best salt-bridge occupancy.")
    lines.append("- `A_minus_mutants_signed_delta_sasa.png`: signed `A - mutant` SASA shifts for pS857, pS859, pT860, and TOTAL.")
    lines.append("- `key_site_metrics.csv`: exact values used in this explanation.")
    lines.append("")
    lines.append("Interpretation")
    lines.append("")
    lines.append("A vs B")
    lines.append(f"- `A:T_to_A` and `B:T_to_A` are both strongly consistent with the paper for pS857/pS859/pT860. In `A`, best contact occupancies are {fmt(a['pS857_contact_occupancy'])}/{fmt(a['pS859_contact_occupancy'])}/{fmt(a['pT860_contact_occupancy'])}% and best salt occupancies are {fmt(a['pS857_salt_occupancy'])}/{fmt(a['pS859_salt_occupancy'])}/{fmt(a['pT860_salt_occupancy'])}%. In `B:T_to_A`, the same values are {fmt(bt['pS857_contact_occupancy'])}/{fmt(bt['pS859_contact_occupancy'])}/{fmt(bt['pT860_contact_occupancy'])}% and {fmt(bt['pS857_salt_occupancy'])}/{fmt(bt['pS859_salt_occupancy'])}/{fmt(bt['pT860_salt_occupancy'])}%.")
    lines.append(f"- `B:T_to_A` is slightly more buried than `A` for pS857 and pS859 by SASA ({fmt(a['pS857_mean_sasa'])} -> {fmt(bt['pS857_mean_sasa'])} A^2 and {fmt(a['pS859_mean_sasa'])} -> {fmt(bt['pS859_mean_sasa'])} A^2), which fits a maintained phospho-tail engagement.")
    lines.append(f"- `B:L_to_C` does not behave like an equally strong second copy. Its SASA is much higher for pS857/pS859/pT860 ({fmt(bl['pS857_mean_sasa'])}/{fmt(bl['pS859_mean_sasa'])}/{fmt(bl['pT860_mean_sasa'])} A^2), and its best contact occupancies collapse to {fmt(bl['pS857_contact_occupancy'])}/{fmt(bl['pS859_contact_occupancy'])}/{fmt(bl['pT860_contact_occupancy'])}%. This does not cleanly support a fully symmetric, equally locked 2:2 arrestin engagement throughout the trajectory.")
    lines.append("")
    lines.append("A vs F670G")
    lines.append(f"- `F670G` is only a mild perturbation overall: TOTAL mean SASA changes from {fmt(a['TOTAL_mean_sasa'])} to {fmt(f670g['TOTAL_mean_sasa'])} A^2.")
    lines.append(f"- The mutation weakens the article-like pattern mainly at pS859 and pT860. pS859 SASA rises from {fmt(a['pS859_mean_sasa'])} to {fmt(f670g['pS859_mean_sasa'])} A^2, pT860 rises from {fmt(a['pT860_mean_sasa'])} to {fmt(f670g['pT860_mean_sasa'])} A^2, and best contact occupancies drop from {fmt(a['pS859_contact_occupancy'])}/{fmt(a['pT860_contact_occupancy'])}% to {fmt(f670g['pS859_contact_occupancy'])}/{fmt(f670g['pT860_contact_occupancy'])}%.")
    lines.append(f"- pS857 stays close to `A` in SASA and still keeps a strong salt bridge ({fmt(f670g['pS857_salt_occupancy'])}%), so `F670G` looks partially consistent with the paper but less stable than `A` in the 859/860 region.")
    lines.append("")
    lines.append("A vs I669G")
    lines.append(f"- `I669G` is the least consistent with the article-like engaged state. TOTAL mean SASA increases from {fmt(a['TOTAL_mean_sasa'])} to {fmt(i669g['TOTAL_mean_sasa'])} A^2.")
    lines.append(f"- The clearest signal is pS857: SASA jumps from {fmt(a['pS857_mean_sasa'])} to {fmt(i669g['pS857_mean_sasa'])} A^2, while the best pS857 contact lifetime drops from {fmt(a['pS857_contact_lifetime_ps'])} to {fmt(i669g['pS857_contact_lifetime_ps'])} ps and the best pS857 salt lifetime drops from {fmt(a['pS857_salt_lifetime_ps'])} to {fmt(i669g['pS857_salt_lifetime_ps'])} ps.")
    lines.append(f"- pS859 and pT860 also remain more exposed or less stably wired than in `A`, so `I669G` is the strongest dynamic argument for weakened phospho-tail engagement with arrestin N-domain.")
    lines.append("")
    lines.append("A vs R668G")
    lines.append(f"- `R668G` shows the strongest burial for pS857 and pS859: SASA falls from {fmt(a['pS857_mean_sasa'])}/{fmt(a['pS859_mean_sasa'])} to {fmt(r668g['pS857_mean_sasa'])}/{fmt(r668g['pS859_mean_sasa'])} A^2, and best contact occupancies reach {fmt(r668g['pS857_contact_occupancy'])}/{fmt(r668g['pS859_contact_occupancy'])}% with salt occupancies {fmt(r668g['pS857_salt_occupancy'])}/{fmt(r668g['pS859_salt_occupancy'])}%.")
    lines.append(f"- That is more consistent with a locked article-like pS857/pS859 engagement than `A` itself. The caveat is pT860: its SASA is slightly higher than in `A` ({fmt(a['pT860_mean_sasa'])} -> {fmt(r668g['pT860_mean_sasa'])} A^2), even though its contact occupancy remains maximal ({fmt(r668g['pT860_contact_occupancy'])}%).")
    lines.append(f"- So `R668G` supports the paper strongly for pS857/pS859 and remains broadly supportive for pT860, but the 860 site is not buried more deeply than in `A`.")
    lines.append("")
    lines.append("Bottom line")
    lines.append("- Best agreement with the paper: `A:T_to_A`, `B:T_to_A`, and `R668G:T_to_A`.")
    lines.append("- Partial agreement with mild weakening: `F670G:T_to_A`.")
    lines.append("- Clear weakening relative to the paper-like engaged state: `I669G:T_to_A`.")
    lines.append("- Internal tension with a perfectly symmetric 2:2 interpretation: `B:L_to_C`, because the second tail is much more exposed and much less persistently engaged than `B:T_to_A`.")
    return "\n".join(lines) + "\n"


def run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results_2" / "phospho_tail_coupling"
    explanation_root = results_root / "explanation"
    explanation_root.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics(results_root)
    write_metrics_csv(metrics, explanation_root / "key_site_metrics.csv")
    plot_summary_panels(metrics, explanation_root / "phospho_key_site_summary.png")
    plot_a_minus_mutant_sasa(metrics, explanation_root / "A_minus_mutants_signed_delta_sasa.png")
    explanation = build_interpretation(metrics)
    (explanation_root / "README.md").write_text(explanation, encoding="utf-8")


if __name__ == "__main__":
    run()
