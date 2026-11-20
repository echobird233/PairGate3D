"""Generate the V4 HCEG ablation radar from paired m122 evaluation data.

The numerical authorities are the formal ablation CSV files. The script validates
the exact equivalence of full HCEG and distance-only gating before plotting, then
exports a vector PDF, a high-resolution PNG, a companion statistics CSV, and a
plain-language QA report.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DETECTORS = ("casa", "point-rcnn", "pvrcnn", "second-iou", "virconv")
DISPLAY_DETECTORS = ("CasA", "PointRCNN", "PV-RCNN", "SECOND-IoU", "VirConv")
RADAR_LABELS = ("CasA", "Point\nRCNN", "PV-RCNN", "SECOND-\nIoU", "Vir\nConv")
PLOTTED_VARIANTS = ("full_hceg", "overlap_only", "no_maturity")
STYLE = {
    "full_hceg": {
        "label": "Full HCEG = distance-only",
        "legend_label": "Mature joint",
        "color": "#6F4DA0",
        "linestyle": "-",
        "marker": "o",
        "alpha": 0.11,
    },
    "overlap_only": {
        "label": "Overlap-only",
        "legend_label": "Overlap-only",
        "color": "#2F6FA3",
        "linestyle": (0, (6, 2.5)),
        "marker": "o",
        "alpha": 0.09,
    },
    "no_maturity": {
        "label": "Joint gate without maturity trigger",
        "legend_label": "Joint, no maturity",
        "color": "#3A8D6D",
        "linestyle": (0, (2, 2)),
        "marker": "o",
        "alpha": 0.08,
    },
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "axes.titlesize": 6.4,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.2,
        "legend.fontsize": 5.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }
)


def _read_inputs(ablation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_sequence = pd.read_csv(ablation_dir / "per_sequence_metrics.csv")
    mechanism = pd.read_csv(ablation_dir / "mechanism_summary.csv")
    required_sequence = {"detector", "variant", "sequence", "HOTA"}
    required_mechanism = {
        "detector",
        "variant",
        "candidate_edges",
        "gated_edges",
        "gated_fraction_of_candidates",
    }
    if not required_sequence.issubset(per_sequence.columns):
        raise ValueError("per_sequence_metrics.csv lacks required columns")
    if not required_mechanism.issubset(mechanism.columns):
        raise ValueError("mechanism_summary.csv lacks required columns")
    return per_sequence, mechanism


def _validate_coverage(per_sequence: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    expected_variants = {"no_hceg", "distance_only", "overlap_only", "no_maturity", "full_hceg"}
    for detector in DETECTORS:
        seq_block = per_sequence.loc[per_sequence["detector"] == detector]
        mech_block = mechanism.loc[mechanism["detector"] == detector]
        if set(seq_block["variant"]) != expected_variants:
            raise ValueError(f"{detector}: incomplete sequence-level variant coverage")
        if set(mech_block["variant"]) != expected_variants:
            raise ValueError(f"{detector}: incomplete mechanism variant coverage")
        counts = seq_block.groupby("variant")["sequence"].nunique()
        if not (counts == 11).all():
            raise ValueError(f"{detector}: every variant must contain 11 sequences")


def _validate_full_equals_distance(
    per_sequence: pd.DataFrame, mechanism: pd.DataFrame
) -> None:
    for detector in DETECTORS:
        block = per_sequence.loc[per_sequence["detector"] == detector]
        full = (
            block.loc[block["variant"] == "full_hceg"]
            .sort_values("sequence")
            .reset_index(drop=True)
        )
        distance = (
            block.loc[block["variant"] == "distance_only"]
            .sort_values("sequence")
            .reset_index(drop=True)
        )
        if not np.array_equal(full["sequence"].to_numpy(), distance["sequence"].to_numpy()):
            raise ValueError(f"{detector}: sequence ordering mismatch")
        if not np.allclose(full["HOTA"], distance["HOTA"], rtol=0.0, atol=0.0):
            raise ValueError(f"{detector}: full HCEG and distance-only HOTA differ")

        mech = mechanism.loc[mechanism["detector"] == detector].set_index("variant")
        fields = ["candidate_edges", "gated_edges", "gated_fraction_of_candidates"]
        if not np.allclose(
            mech.loc["full_hceg", fields].to_numpy(dtype=float),
            mech.loc["distance_only", fields].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"{detector}: full and distance-only mechanism counts differ")


def compute_statistics(ablation_dir: Path) -> pd.DataFrame:
    per_sequence, mechanism = _read_inputs(ablation_dir)
    _validate_coverage(per_sequence, mechanism)
    _validate_full_equals_distance(per_sequence, mechanism)

    rows: list[dict[str, float | int | str]] = []
    tolerance = 1e-12
    for detector, display in zip(DETECTORS, DISPLAY_DETECTORS, strict=True):
        seq_block = per_sequence.loc[per_sequence["detector"] == detector]
        baseline = (
            seq_block.loc[seq_block["variant"] == "no_hceg"]
            .set_index("sequence")["HOTA"]
            .sort_index()
        )
        mech_block = mechanism.loc[mechanism["detector"] == detector].set_index("variant")
        for variant in PLOTTED_VARIANTS:
            variant_hota = (
                seq_block.loc[seq_block["variant"] == variant]
                .set_index("sequence")["HOTA"]
                .sort_index()
            )
            delta = variant_hota - baseline
            wins = int((delta > tolerance).sum())
            losses = int((delta < -tolerance).sum())
            ties = int(len(delta) - wins - losses)
            candidate_edges = int(mech_block.loc[variant, "candidate_edges"])
            gated_edges = int(mech_block.loc[variant, "gated_edges"])
            veto_rate = 100.0 * gated_edges / candidate_edges
            rows.append(
                {
                    "detector": detector,
                    "detector_display": display,
                    "variant": variant,
                    "variant_display": STYLE[variant]["label"],
                    "n_sequences": int(len(delta)),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "hota_win_rate_percent": 100.0 * wins / len(delta),
                    "candidate_edges": candidate_edges,
                    "gated_edges": gated_edges,
                    "candidate_edge_veto_rate_percent": veto_rate,
                }
            )
    statistics = pd.DataFrame.from_records(rows)
    if len(statistics) != len(DETECTORS) * len(PLOTTED_VARIANTS):
        raise ValueError("Expected exactly 15 detector-by-variant rows")
    return statistics


def _profile(statistics: pd.DataFrame, variant: str, field: str) -> np.ndarray:
    subset = statistics.loc[statistics["variant"] == variant].set_index("detector")
    return subset.loc[list(DETECTORS), field].to_numpy(dtype=float)


def _draw_panel(
    ax: mpl.axes.Axes,
    statistics: pd.DataFrame,
    *,
    field: str,
    title: str,
    radial_max: float,
    radial_ticks: tuple[float, ...],
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, len(DETECTORS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(RADAR_LABELS)
    # Pull the category labels toward the frame so the top label clears the title.
    ax.tick_params(axis="x", pad=-1)
    ax.set_ylim(0.0, radial_max)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in radial_ticks], color="#666666")
    ax.set_rlabel_position(20)
    ax.tick_params(axis="y", pad=1)
    ax.xaxis.grid(True, color="#C8C8C8", linewidth=0.55)
    ax.yaxis.grid(True, color="#D0D0D0", linewidth=0.55)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_linewidth(0.50)
    ax.set_title(title, pad=18, fontweight="bold")

    for variant in PLOTTED_VARIANTS:
        values = _profile(statistics, variant, field)
        closed_values = np.concatenate([values, values[:1]])
        style = STYLE[variant]
        ax.plot(
            closed_angles,
            closed_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.85,
            marker=style["marker"],
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.65,
            zorder=4,
        )
        ax.fill(
            closed_angles,
            closed_values,
            color=style["color"],
            alpha=style["alpha"],
            linewidth=0,
            zorder=2,
        )


def create_figure(statistics: pd.DataFrame) -> mpl.figure.Figure:
    # Single-column IEEE figure: two compact panels within an 88 mm canvas.
    width_inches = 88.0 / 25.4
    height_inches = 55.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width_inches, height_inches),
        subplot_kw={"projection": "polar"},
    )
    _draw_panel(
        axes[0],
        statistics,
        field="hota_win_rate_percent",
        title="(a) Sequence HOTA wins (%)",
        radial_max=110.0,
        radial_ticks=(25.0, 50.0, 75.0, 100.0),
    )
    _draw_panel(
        axes[1],
        statistics,
        field="candidate_edge_veto_rate_percent",
        title="(b) Candidate-edge veto (%)",
        radial_max=72.0,
        radial_ticks=(20.0, 40.0, 60.0),
    )
    handles = []
    for variant in PLOTTED_VARIANTS:
        style = STYLE[variant]
        handle = mpl.lines.Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.65,
            linewidth=0.85,
            markersize=2.8,
            label=style["legend_label"],
        )
        handles.append(handle)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        columnspacing=0.75,
        handlelength=1.7,
    )
    fig.subplots_adjust(left=0.055, right=0.955, top=0.79, bottom=0.24, wspace=0.16)
    return fig


def _write_qa(statistics: pd.DataFrame, output_path: Path) -> None:
    total_candidates = int(
        statistics.loc[statistics["variant"] == "full_hceg", "candidate_edges"].sum()
    )
    total_gated = int(
        statistics.loc[statistics["variant"] == "full_hceg", "gated_edges"].sum()
    )
    output_path.write_text(
        "# HCEG V4 dual-radar QA report\n\n"
        "## Figure contract\n\n"
        "- Main result: the joint no-maturity gate has the largest aggregate HOTA gain on all five detector streams.\n"
        "- Panel (a): strict positive per-sequence HOTA changes over 11 paired sequences.\n"
        "- Panel (b): gated candidate edges divided by all candidate edges.\n"
        "- Archetype: single-column quantitative two-panel figure.\n\n"
        "## Integrity checks\n\n"
        "- Five detector streams, three plotted variants, and 11 sequences per detector were verified.\n"
        "- The mature joint and mature distance-only HOTA values match exactly for every detector-sequence pair.\n"
        "- The mature joint and mature distance-only candidate and gated edge counts match exactly.\n"
        "- No min-max normalization, clipping, imputation, or row exclusion is used.\n"
        f"- The selected joint no-maturity rule evaluates {total_candidates:,} candidate edges and vetoes {total_gated:,}.\n"
        "- The radar is descriptive; area is not interpreted as a scalar performance score.\n\n"
        "## Export contract\n\n"
        "- Vector PDF and SVG with editable text.\n"
        "- 600 dpi PNG for manuscript insertion and preview.\n"
        "- Companion CSV contains all displayed percentages and underlying counts.\n",
        encoding="utf-8",
    )


def generate_outputs(ablation_dir: Path, output_dir: Path, official_png_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    official_png_dir.mkdir(parents=True, exist_ok=True)
    statistics = compute_statistics(ablation_dir)
    statistics_path = output_dir / "HCEG_V4_Dual_Radar_statistics.csv"
    statistics.to_csv(
        statistics_path,
        index=False,
        float_format="%.10f",
        quoting=csv.QUOTE_MINIMAL,
    )
    figure = create_figure(statistics)
    pdf_path = output_dir / "HCEG_V4_Dual_Radar.pdf"
    svg_path = output_dir / "HCEG_V4_Dual_Radar.svg"
    png_path = output_dir / "HCEG_V4_Dual_Radar.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(png_path, official_png_dir / png_path.name)
    _write_qa(statistics, output_dir / "HCEG_V4_Dual_Radar_QA.md")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-png-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generate_outputs(args.ablation_dir, args.output_dir, args.official_png_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
