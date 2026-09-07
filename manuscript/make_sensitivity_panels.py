"""Build the six panels for the three sensitivity Results subsections.

Every number drawn here is read from a committed results file; nothing is
hardcoded, so re-running the upstream experiment and re-running this script
keeps the figures and the text in step.

  ext_a_separation.pdf   RDFace disease separation under stacked controls
  ext_b_resolution.pdf   measurement drift against image resolution
  ref_a_ancestry.pdf     where the ancestry-conditioned reference helps
  ref_b_age.pdf          the same reading for age
  anc_a_scale.pdf        22q interaction q, raw scale against log scale
  anc_b_ratio.pdf        22q case-to-control ratio per ancestry group

    python manuscript/make_sensitivity_panels.py [--chop-run DIR]

Colours are the Okabe-Ito blue / vermillion / bluish-green triple, validated for
colour-vision deficiency; every split is additionally carried by marker shape or
by a direct label, never by hue alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "image"

RDFACE = REPO / "experiments" / "external_rdface" / "results"
REFSENS = REPO / "experiments" / "reference_sensitivity" / "results"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
INK = "#1a1a1a"
MUTED = "#6E6E6E"
GRID = "#DDDDDD"

SEED = 20260907
N_BOOT = 2000

ANCESTRY_LABEL = {"white": "European", "black": "African", "asian": "Asian"}
ANCESTRY_COLOR = {"white": BLUE, "black": VERMILLION, "asian": GREEN}
ANCESTRY_MARKER = {"white": "o", "black": "s", "asian": "^"}

FEATURE_LABEL = {
    "upper_vermilion_height": "upper vermilion height",
    "upper_lip_eversion": "upper lip eversion",
    "vermilion_total": "vermilion total",
    "lower_vermilion_height": "lower vermilion height",
    "lower_lip_eversion": "lower lip eversion",
    "canthal_to_pupillary_ratio": "canthal / pupillary ratio",
    "cheek_area_mean": "cheek area",
    "malar_bulge_mean": "malar bulge",
    "face_width_uniformity": "face width uniformity",
    "nose_bridge_width": "nose bridge width",
}


def label(feature: str) -> str:
    return FEATURE_LABEL.get(feature, feature.replace("_", " "))


def save(fig: plt.Figure, stem: str) -> None:
    """Vector for the manuscript, raster alongside it for quick inspection."""
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {path}.pdf")


def style(ax: plt.Axes) -> None:
    """Recessive grid and axes; the data carries the ink."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.set_axisbelow(True)


# ==========================================================================
# Panel (a): RDFace disease separation survives every control
# ==========================================================================
SUBSET_LABEL = {
    "all frontal images": "all frontal images",
    "near-duplicates removed": "near-duplicates removed",
    "same-patient images removed": "same-patient images removed",
    "same-patient removed, short side >= 128 px": "and short side $\\geq$ 128 px",
    "same-patient removed, short side >= 160 px": "and short side $\\geq$ 160 px",
    "same-patient removed, short side >= 224 px": "and short side $\\geq$ 224 px",
}


def panel_separation() -> None:
    d = pd.read_csv(RDFACE / "disease_separation.csv")
    d = d.set_index("subset").loc[list(SUBSET_LABEL)].reset_index()
    d["sd_below"] = -d.z_vs_null
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    ax.hlines(y, 0, d.sd_below, color=GRID, lw=1.4, zorder=1)
    ax.scatter(d.sd_below, y, s=40, c=BLUE, zorder=3,
               edgecolor="white", linewidths=0.6)
    for yi, row in zip(y, d.itertuples()):
        ax.text(row.sd_below + 0.35, yi, f"{row.sd_below:.1f}",
                va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{SUBSET_LABEL[r.subset]}\n$n$ = {r.n_images}, "
         f"{r.n_diseases} diseases" for r in d.itertuples()], fontsize=7.0)
    ax.set_xlim(0, 12.6)
    ax.set_xticks([0, 5, 10])
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("SD below the permutation null", fontsize=8.5, color=INK)
    ax.axhline(2.5, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=0)
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "ext_a_separation")


# ==========================================================================
# Panel (b): how far the catalogue can be pushed down in resolution
# ==========================================================================
def panel_resolution() -> None:
    from matplotlib.ticker import NullFormatter, NullLocator

    d = pd.read_csv(REFSENS / "resolution_drift.csv")
    px = np.array([64, 96, 128, 160, 224])
    bias = np.abs(d[[f"bias_{p}" for p in px]].to_numpy())

    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ax.fill_between(px, np.percentile(bias, 25, axis=0),
                    np.percentile(bias, 75, axis=0),
                    color=BLUE, alpha=0.14, lw=0, zorder=1)
    ax.plot(px, np.median(bias, axis=0), color=BLUE, lw=2.2, zorder=4,
            label="median of the 120 (IQR shaded)")

    worst = d.reindex(d.bias_128.abs().sort_values(ascending=False).index).head(3)
    for colour, row in zip((VERMILLION, GREEN, MUTED), worst.itertuples()):
        vals = np.abs([getattr(row, f"bias_{p}") for p in px])
        ax.plot(px, vals, color=colour, lw=1.2, zorder=3, label=label(row.feature))

    ax.axhline(0.1, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.text(218, 0.115, "0.1 reference SD", fontsize=6.5, color=INK, ha="right")
    ax.axvline(128, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=2)
    ax.text(126, 0.015, "operating floor", fontsize=6.5, color=MUTED,
            rotation=90, va="bottom", ha="right")

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(px)
    ax.set_xticklabels(px)
    ax.set_xlim(62, 228)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Image short side (pixels)", fontsize=8.5, color=INK)
    ax.set_ylabel("$|$shift$|$ from the native measurement\n(reference SD)",
                  fontsize=8.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False, handletextpad=0.5,
              borderpad=0.2, labelspacing=0.35)
    style(ax)
    fig.tight_layout()
    save(fig, "ext_b_resolution")


# ==========================================================================
# Panels (c) and (d): does the conditioning help where it should?
# ==========================================================================
SENSITIVITY_COLUMN = {"ancestry": "race_eta2", "age": "age_eta2",
                      "resolution": "res_drift_224"}
ARM_TITLE = {"ancestry": "Ancestry-conditioned",
             "age": "Age-conditioned",
             "resolution": "Resolution-conditioned"}
ARM_XLABEL = {
    "ancestry": "Partial $\\eta^2$ of ancestry in the\nmapped measurement",
    "age": "Partial $\\eta^2$ of age band in the\nmapped measurement",
    "resolution": "Drift of the mapped measurement\nat 224 px (reference SD)",
}


def panel_mechanism(arm: str, stem: str) -> None:
    """Does the conditioning help where the measurement is most sensitive?

    One point per HPO term. The abscissa is how sensitive that term's mapped
    measurement is to the conditioned variable, the ordinate how much the term
    gains when the reference is conditioned on it.
    """
    from scipy.stats import spearmanr, mannwhitneyu

    col = SENSITIVITY_COLUMN[arm]
    comp = pd.read_csv(REFSENS / arm / "comparison.csv")
    sens = pd.read_csv(REFSENS / "feature_sensitivity.csv")[["feature", col]]
    d = comp.merge(sens, on="feature", how="inner").dropna(subset=[col, "gain"])

    rho, _ = spearmanr(d[col], d.gain)
    cut = d[col].median()
    hi, lo = d[d[col] > cut], d[d[col] <= cut]
    _, p_mw = mannwhitneyu(hi.gain, lo.gain)

    fig, ax = plt.subplots(figsize=(2.7, 2.9))
    ax.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax.axvline(cut, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.scatter(lo[col], lo.gain, s=13, facecolor="white", edgecolor=MUTED,
               linewidths=0.7, zorder=3)
    ax.scatter(hi[col], hi.gain, s=15, c=BLUE, alpha=0.75, linewidths=0, zorder=3)

    for sub, colour, side in ((lo, MUTED, "left"), (hi, BLUE, "right")):
        med = sub.gain.median()
        x0, x1 = (d[col].min(), cut) if side == "left" else (cut, d[col].max())
        ax.hlines(med, x0, x1, color=colour, lw=2.4, zorder=5)

    ax.set_title(ARM_TITLE[arm], fontsize=8, color=INK, pad=5)
    ax.set_xlabel(ARM_XLABEL[arm], fontsize=7.5, color=INK)
    ax.set_ylabel("Gain in signed effect", fontsize=7.5, color=INK)
    ax.set_ylim(-0.75, 0.95)
    ax.text(0.96, 0.035,
            f"$\\rho$ = {rho:+.2f}\n"
            f"median {hi.gain.median():+.3f} vs {lo.gain.median():+.3f}\n"
            f"MW $p$ = {p_mw:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            color=INK)
    handles = [Line2D([], [], ls="none", marker="o", markersize=3.8,
                      markerfacecolor="white", markeredgecolor=MUTED,
                      label=f"less sensitive ($n$ = {len(lo)})"),
               Line2D([], [], ls="none", marker="o", markersize=3.8,
                      markerfacecolor=BLUE, markeredgecolor="none",
                      label=f"more sensitive ($n$ = {len(hi)})")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, frameon=False,
              handletextpad=0.3, borderpad=0.15, labelspacing=0.25)
    style(ax)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, stem)
    print(f"  {arm}: rho={rho:+.3f}  MW p={p_mw:.4f}  "
          f"median gain hi={hi.gain.median():+.4f} lo={lo.gain.median():+.4f}")


# ==========================================================================
# Panel (e): the 22q interaction on the raw and on the log scale
# ==========================================================================
def panel_scale(chop: Path) -> None:
    d = pd.read_csv(chop / "interaction120" / "log_scale_check.csv")
    d = d[(d.q_raw < 0.05) | (d.q_log < 0.05)].sort_values("q_raw")
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    ax.axvspan(1e-3, 0.05, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.axvline(0.05, color=INK, lw=0.9, ls=(0, (3, 3)), zorder=2)
    for yi, row in zip(y, d.itertuples()):
        ax.annotate("", xy=(row.q_log, yi), xytext=(row.q_raw, yi),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                    shrinkA=3.5, shrinkB=3.5), zorder=2)
    ax.scatter(d.q_raw, y, s=34, c=BLUE, zorder=4, edgecolor="white",
               linewidths=0.6, label="raw scale")
    ax.scatter(d.q_log, y, s=38, marker="s", facecolor="white",
               edgecolor=VERMILLION, linewidths=1.3, zorder=4, label="log scale")

    ax.set_yticks(y)
    ax.set_yticklabels([label(f) for f in d.feature], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.4)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Interaction $q$ (BH over 97 measurements)",
                  fontsize=8.5, color=INK)
    ax.text(0.053, len(d) - 0.55, "$q$ = 0.05", fontsize=6.8, color=INK)
    ax.legend(loc="lower left", fontsize=7, frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.3)
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "anc_a_scale")


# ==========================================================================
# Panel (f): case-to-control ratio, one estimate per ancestry group
# ==========================================================================
def panel_ratio(chop: Path) -> None:
    rng = np.random.default_rng(SEED)
    q22 = pd.read_csv(chop / "features_22q_race120.csv")
    ff = pd.read_csv(chop / "features_fairface_race120.csv")
    ff = ff[(ff[["pose_yaw", "pose_pitch", "pose_roll"]].abs() < 8.0).all(axis=1)]

    feats = pd.read_csv(chop / "interaction120" / "log_scale_check.csv")
    feats = feats[(feats.q_raw < 0.05) | (feats.q_log < 0.05)]
    feats = feats.sort_values("q_raw").feature.tolist()

    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    ax.axvline(1.0, color=INK, lw=0.9, ls=(0, (3, 3)), zorder=2)
    offsets = {"white": +0.24, "black": 0.0, "asian": -0.24}
    y = {f: i for i, f in enumerate(feats[::-1])}

    for f in feats:
        for grp in ("white", "black", "asian"):
            a = q22.loc[q22.disease == grp, f].dropna().to_numpy()
            b = ff.loc[ff.disease == grp, f].dropna().to_numpy()
            ratio = a.mean() / b.mean()
            boot = np.array([
                rng.choice(a, a.size, replace=True).mean()
                / rng.choice(b, b.size, replace=True).mean()
                for _ in range(N_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            yy = y[f] + offsets[grp]
            ax.plot([lo, hi], [yy, yy], color=ANCESTRY_COLOR[grp], lw=1.1,
                    solid_capstyle="butt", zorder=3)
            ax.scatter(ratio, yy, s=26, marker=ANCESTRY_MARKER[grp],
                       c=ANCESTRY_COLOR[grp], zorder=4, edgecolor="white",
                       linewidths=0.5)

    ax.set_yticks(list(y.values()))
    ax.set_yticklabels([label(f) for f in y], fontsize=7.5)
    ax.set_ylim(-0.7, len(feats) - 0.3)
    ax.set_xlabel("22q11.2 mean $/$ control mean", fontsize=8.5, color=INK)
    handles = [Line2D([], [], ls="none", marker=ANCESTRY_MARKER[g], markersize=4.5,
                      color=ANCESTRY_COLOR[g], label=ANCESTRY_LABEL[g])
               for g in ("white", "black", "asian")]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.3)
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "anc_b_ratio")


def main() -> None:
    ap = argparse.ArgumentParser()
    default = sorted((REPO / "results" / "chop_22q").glob("*"))[-1]
    ap.add_argument("--chop-run", type=Path, default=default,
                    help="timestamped 22q run directory")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"22q run: {args.chop_run}")

    panel_separation()
    panel_resolution()
    panel_mechanism("ancestry", "ref_a_ancestry")
    panel_mechanism("age", "ref_b_age")
    panel_mechanism("resolution", "ref_c_resolution")
    panel_scale(args.chop_run)
    panel_ratio(args.chop_run)


if __name__ == "__main__":
    main()
