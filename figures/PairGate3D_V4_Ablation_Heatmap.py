import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox


SOURCE_CSV = "PairGate3D_V4_Ablation_Heatmap_aggregate_source.csv"
DETECTORS = ("casa", "point-rcnn", "pvrcnn", "second-iou", "virconv")
DISPLAY = ("CasA", "PointRCNN", "PV-RCNN", "SECOND-IoU", "VirConv")
VARIANTS = ("no_hceg", "distance_only", "overlap_only", "full_hceg", "no_maturity")
ROW_LABELS = (
    "Gate off",
    "Distance-only mature",
    "Overlap-only mature",
    "Joint mature",
    "Joint no maturity",
)
MM_PER_INCH = 25.4


def save_ieee_column_figure(fig, filename: Path, png_filename: Path) -> None:
    # Measure visible artwork, including labels, rather than empty axes boxes.
    # Crop the vector artboard consistently with PNG; retain a 0.5 mm margin.
    # This only changes export bounds, never the plotted values or geometry.
    original_dpi = fig.dpi
    fig.set_dpi(400)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = np.any(rgba[:, :, :3] < 254, axis=2) & (rgba[:, :, 3] > 0)
    yy, xx = np.nonzero(ink)
    if not len(xx):
        raise ValueError("Figure contains no visible artwork")
    height = rgba.shape[0]
    page = Bbox.from_extents(xx.min()/400, (height-yy.max()-1)/400,
                            (xx.max()+1)/400, (height-yy.min())/400)
    page = page.padded(0.5 / MM_PER_INCH)
    fig.set_dpi(original_dpi)
    fig.savefig(f"{filename}.pdf", bbox_inches=page, pad_inches=0, dpi=400)
    fig.savefig(f"{filename}.svg", bbox_inches=page, pad_inches=0, dpi=400)
    fig.savefig(f"{png_filename}.png", bbox_inches=page, pad_inches=0, dpi=400)


def read_and_validate(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    required = {"detector", "variant", "HOTA"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    if len(df) != 25 or df.duplicated(["detector", "variant"]).any():
        raise ValueError("Expected one aggregate row for each of 5 detectors × 5 variants")
    if set(df["detector"].unique()) != set(DETECTORS) or set(df["variant"].unique()) != set(VARIANTS):
        raise ValueError("Aggregate detector or variant names do not match the frozen ablation suite")
    if not np.isfinite(df["HOTA"].to_numpy(dtype=float)).all():
        raise ValueError("Aggregate HOTA contains non-finite values")
    return df


def make_effect_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = df.set_index(["variant", "detector"])["HOTA"]
    rows = []
    for variant, label in zip(VARIANTS, ROW_LABELS, strict=True):
        raw_values = []
        for detector, display in zip(DETECTORS, DISPLAY, strict=True):
            off_hota = float(indexed.loc[("no_hceg", detector)])
            variant_hota = float(indexed.loc[(variant, detector)])
            raw_values.append({
                "variant": label,
                "variant_key": variant,
                "detector": display,
                "gate_off_hota": off_hota,
                "variant_hota": variant_hota,
                "delta_hota_points": variant_hota - off_hota,
            })
        rows.extend(raw_values)
        mean_off = float(np.mean([r["gate_off_hota"] for r in raw_values]))
        mean_variant = float(np.mean([r["variant_hota"] for r in raw_values]))
        rows.append({
            "variant": label,
            "variant_key": variant,
            "detector": "Mean",
            "gate_off_hota": mean_off,
            "variant_hota": mean_variant,
            "delta_hota_points": mean_variant - mean_off,
        })
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="variant", columns="detector", values="delta_hota_points").reindex(index=ROW_LABELS, columns=(*DISPLAY, "Mean"))
    values = matrix.to_numpy(dtype=float)
    if values.shape != (5, 6) or not np.isfinite(values).all():
        raise ValueError("Effect matrix must be a complete 5 × 6 finite array")
    if (values < -1e-10).any():
        raise ValueError("Expected all nonnegative aggregate HOTA changes")
    if not np.allclose(
        df.loc[df["variant"].eq("distance_only")].sort_values("detector")["HOTA"].to_numpy(),
        df.loc[df["variant"].eq("full_hceg")].sort_values("detector")["HOTA"].to_numpy(),
        rtol=0.0, atol=0.0,
    ):
        raise ValueError("Distance-only mature and joint mature aggregate values should match exactly")
    expected_means = np.array([0.000, 2.049, 1.964, 2.049, 2.341])
    if not np.allclose(matrix["Mean"].to_numpy(), expected_means, atol=0.0005, rtol=0.0):
        raise ValueError("Detector-equal means do not match the reported ablation means")
    return long, matrix


def draw_heatmap(matrix: pd.DataFrame, output_stem: Path, png_stem: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pairgate3d_sequential", ["#F1F2F3", SEQUENTIAL[1], SEQUENTIAL[2]]
    )
    fig = plt.figure(figsize=(89 / MM_PER_INCH, 65 / MM_PER_INCH))
    ax = fig.add_axes([0.34, 0.13, 0.54, 0.66])
    cax = fig.add_axes([0.92, 0.13, 0.022, 0.66])
    n_rows, n_cols = values.shape
    im = ax.pcolormesh(
        np.arange(n_cols + 1) - 0.5,
        np.arange(n_rows + 1) - 0.5,
        values,
        cmap=cmap,
        vmin=0.0,
        vmax=6.0,
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
        rasterized=False,
    )
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels((*DISPLAY, "Mean"), rotation=45, ha="left", va="bottom", rotation_mode="anchor", fontsize=5.2)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="major", top=False, bottom=False, labeltop=True, labelbottom=False, pad=3, length=0)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(ROW_LABELS, fontsize=5.3)
    ax.tick_params(axis="y", which="major", left=False, right=False, length=0, pad=3)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for (row, col), val in np.ndenumerate(values):
        # Direct labeling shows every aggregate cell value; no legend is used.
        rgba = cmap(val / 6.0)
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        text_color = "white" if luminance < 0.47 else BLACK
        ax.text(col, row, f"{val:.3f}", ha="center", va="center", fontsize=5.0, color=text_color)

    selected_row = ROW_LABELS.index("Joint no maturity")
    ax.add_patch(Rectangle(
        (-0.5, selected_row - 0.5), n_cols, 1.0,
        fill=False, edgecolor=CATEGORICAL[2], linewidth=0.9, zorder=4, clip_on=False,
    ))
    for tick in ax.get_yticklabels():
        if tick.get_text() == "Joint no maturity":
            tick.set_fontweight("bold")
            tick.set_color(CATEGORICAL[2])

    colorbar = fig.colorbar(im, cax=cax, ticks=[0, 2, 4, 6])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    colorbar.ax.tick_params(labelsize=5.0, length=2, width=0.5, pad=2)
    colorbar.outline.set_visible(False)
    colorbar.ax.set_title(r"$\Delta$HOTA", fontsize=5.1, pad=3)

    save_ieee_column_figure(fig, output_stem, png_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the PairGate3D structural ablation heatmap.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    default_png_dir = Path(__file__).resolve().parent.parent / "PNG figure"
    if not default_png_dir.is_dir():
        default_png_dir = Path(__file__).resolve().parent
    parser.add_argument("--png-dir", type=Path, default=default_png_dir)
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"Output directory must already exist: {args.output_dir}")
    if not args.png_dir.is_dir():
        raise FileNotFoundError(f"PNG directory must already exist: {args.png_dir}")
    source = Path(__file__).resolve().parent / SOURCE_CSV
    source_data = read_and_validate(source)
    long, matrix = make_effect_matrix(source_data)
    long.to_csv(args.output_dir / "PairGate3D_V4_Ablation_Heatmap_statistics.csv", index=False, float_format="%.12g")
    draw_heatmap(
        matrix,
        args.output_dir / "PairGate3D_V4_Ablation_Heatmap",
        args.png_dir / "PairGate3D_V4_Ablation_Heatmap",
    )
    print(matrix.round(3).to_string())


if __name__ == "__main__":
    main()
