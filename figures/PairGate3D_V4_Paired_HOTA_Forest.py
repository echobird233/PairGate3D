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
from matplotlib.transforms import Bbox


DETECTORS = ("casa", "point-rcnn", "pvrcnn", "second-iou", "virconv")
DISPLAY = ("CasA", "PointRCNN", "PV-RCNN", "SECOND-IoU", "VirConv")
SOURCE_CSV = "PairGate3D_V4_Paired_HOTA_Forest_per_sequence_source.csv"
RESAMPLES = 10_000
SEED = 2027
MM_PER_INCH = 25.4
mpl.rcParams["errorbar.capsize"] = 2.0


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
    df = pd.read_csv(source, dtype={"sequence": str})
    required = {"detector", "variant", "sequence", "HOTA"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    if len(df) != 275:
        raise ValueError(f"Expected all 275 source rows, found {len(df)}")
    df["sequence"] = df["sequence"].str.zfill(4)
    sequences = tuple(sorted(df["sequence"].unique()))
    expected_sequences = ("0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019")
    if sequences != expected_sequences:
        raise ValueError(f"Held-out sequence mismatch: {sequences}")
    if set(df["detector"].unique()) != set(DETECTORS):
        raise ValueError("Detector set does not match the five frozen KITTI streams")
    if set(df["variant"].unique()) != {"no_hceg", "distance_only", "overlap_only", "no_maturity", "full_hceg"}:
        raise ValueError("Unexpected ablation variant names")
    counts = df.groupby(["detector", "variant"])["sequence"].nunique()
    if len(counts) != 25 or not (counts == 11).all():
        raise ValueError("Each detector × variant pair must contain the same 11 sequences")
    if df.duplicated(["detector", "variant", "sequence"]).any():
        raise ValueError("Duplicate detector × variant × sequence rows found")
    if not np.isfinite(df["HOTA"].to_numpy(dtype=float)).all():
        raise ValueError("HOTA contains non-finite values")
    return df


def paired_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered_sequences = tuple(sorted(df["sequence"].unique()))
    for detector, display in zip(DETECTORS, DISPLAY, strict=True):
        block = df.loc[df["detector"].eq(detector)]
        gate_on = block.loc[block["variant"].eq("no_maturity")].set_index("sequence")["HOTA"]
        gate_off = block.loc[block["variant"].eq("no_hceg")].set_index("sequence")["HOTA"]
        paired = (gate_on.reindex(ordered_sequences) - gate_off.reindex(ordered_sequences)).to_numpy(dtype=float)
        if len(paired) != 11 or not np.isfinite(paired).all():
            raise ValueError(f"Invalid paired differences for {detector}")
        # Legacy RandomState reproduces the manuscript's 10,000-resample intervals.
        rng = np.random.RandomState(SEED)
        sample_indices = rng.randint(0, len(paired), size=(RESAMPLES, len(paired)))
        bootstrap_means = paired[sample_indices].mean(axis=1)
        low, high = np.percentile(bootstrap_means, [2.5, 97.5], method="linear")
        rows.append({
            "detector": detector,
            "display_name": display,
            "n_sequences": int(len(paired)),
            "mean_delta_hota_points": float(paired.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "resamples": RESAMPLES,
            "seed": SEED,
            "ci_method": "paired sequence bootstrap percentile (2.5th, 97.5th)",
            "bootstrap_rng": "numpy.random.RandomState",
        })
    return pd.DataFrame(rows)


def draw_forest(stats: pd.DataFrame, output_stem: Path, png_stem: Path) -> None:
    width_in = 89 / MM_PER_INCH
    height_in = 44 / MM_PER_INCH
    fig = plt.figure(figsize=(width_in, height_in))
    grid = fig.add_gridspec(1, 3, width_ratios=(0.95, 2.55, 1.75), wspace=0.025)
    names_ax = fig.add_subplot(grid[0, 0])
    forest_ax = fig.add_subplot(grid[0, 1])
    values_ax = fig.add_subplot(grid[0, 2])

    y = np.arange(len(stats) - 1, -1, -1)
    means = stats["mean_delta_hota_points"].to_numpy()
    low = stats["ci95_low"].to_numpy()
    high = stats["ci95_high"].to_numpy()
    xerr = np.vstack((means - low, high - means))
    effect_color = CATEGORICAL[0]
    interval_color = "#4393C3"
    axis_min = min(-0.3, float(low.min()) - 0.25)
    axis_max = max(10.5, float(high.max()) + 0.5)

    forest_ax.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.65, zorder=0)
    forest_ax.errorbar(
        means, y, xerr=xerr, fmt="D", color=effect_color, ecolor=interval_color,
        ms=3.8, markerfacecolor=effect_color, markeredgecolor=effect_color,
        elinewidth=1.0, capthick=0.8, zorder=3,
    )
    forest_ax.set_xlim(axis_min, axis_max)
    forest_ax.set_ylim(-0.55, len(stats) - 0.22)
    forest_ax.set_yticks([])
    forest_ax.set_xticks([0, 2, 4, 6, 8, 10])
    forest_ax.tick_params(axis="x", labelsize=5.2, pad=2, length=2.0)
    forest_ax.set_xlabel(r"$\Delta$HOTA (Gate on $-$ Gate off; points)", fontsize=5.7, labelpad=2)
    forest_ax.spines["left"].set_visible(False)
    forest_ax.spines["bottom"].set_linewidth(0.6)
    forest_ax.spines["top"].set_visible(False)
    forest_ax.spines["right"].set_visible(False)
    forest_ax.set_axisbelow(True)

    for ax in (names_ax, values_ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.55, len(stats) - 0.22)
        ax.axis("off")
    for yi, row in zip(y, stats.itertuples(index=False), strict=True):
        # Direct labels carry each detector and its mean/interval.
        names_ax.text(1.0, yi, row.display_name, ha="right", va="center", fontsize=5.5, color=BLACK)
        values_ax.text(
            0.015, yi,
            f"{row.mean_delta_hota_points:+.3f}  [{row.ci95_low:.3f}, {row.ci95_high:.3f}]",
            ha="left", va="center", fontsize=5.0, color=BLACK,
        )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.27, top=0.90, wspace=0.025)

    save_ieee_column_figure(fig, output_stem, png_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the PairGate3D paired KITTI HOTA forest plot.")
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
    df = read_and_validate(source)
    stats = paired_effects(df)
    stats.to_csv(args.output_dir / "PairGate3D_V4_Paired_HOTA_Forest_statistics.csv", index=False, float_format="%.12g")
    draw_forest(
        stats,
        args.output_dir / "PairGate3D_V4_Paired_HOTA_Forest",
        args.png_dir / "PairGate3D_V4_Paired_HOTA_Forest",
    )
    print(stats[["display_name", "n_sequences", "mean_delta_hota_points", "ci95_low", "ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
