"""Build the Results 3.2 figure panels from the synthetic-vs-real analysis.

Reads the post-audit analysis outputs produced by facekit_analysis (the 120-feature
run; `analysis_pre_audit/` is the superseded 125-feature run and must not be used)
and writes three vector panels that carry the numbers reported in Results 3.2:

  (a) per-syndrome discriminative effect, synthetic against real, one point per
      syndrome-measurement pair, split by whether the two agree in sign
  (b) within-syndrome agreement between the synthetic and real distributions
  (c) variance ratio by facial region

    python manuscript/make_synthetic_panels.py [--analysis DIR]

Colours are the Okabe-Ito blue/vermillion pair (validated for colour-vision
deficiency); "agrees" is additionally the recessive mark, so the split is never
carried by hue alone.
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
DEFAULT_ANALYSIS = (
    HERE.parent.parent / "facekit_analysis" / "analysis"
)

BLUE = "#0072B2"     # direction agrees
VERMILLION = "#D55E00"  # direction disagrees
INK = "#1a1a1a"
MUTED = "#6E6E6E"
GRID = "#DDDDDD"

SYNDROME_LABEL = {
    "22q11.2_deletion_syndrome": "22q11.2 deletion",
    "angelman_syndrome": "Angelman",
    "cornelia_de_lange_syndrome": "Cornelia de Lange",
    "kabuki_syndrome": "Kabuki",
    "kbg_syndrome": "KBG",
    "nicolaides_baraitser_syndrome": "Nicolaides–Baraitser",
    "noonan_syndrome": "Noonan",
    "rubinstein_taybi_syndrome": "Rubinstein–Taybi",
    "smith_magenis_syndrome": "Smith–Magenis",
    "williams_beuren_syndrome": "Williams–Beuren",
}

REGION_LABEL = {
    "chin_jaw": "Chin & jaw",
    "eye": "Eye",
    "eyebrow": "Eyebrow",
    "mouth_lip": "Mouth & lips",
    "forehead_hairline": "Forehead",
    "face_global": "Whole face",
    "nose": "Nose",
    "philtrum": "Philtrum",
}


def save(fig: plt.Figure, stem: Path) -> None:
    """Vector for the manuscript, raster alongside it for quick inspection."""
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


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


def panel_a(sign_all: pd.DataFrame, out: Path) -> None:
    """Synthetic vs real discriminative effect size, 265 pairs."""
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    lim = 1.05 * max(
        sign_all.d_real.abs().max(), sign_all.d_syn.abs().max()
    )

    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], ls=(0, (4, 3)), lw=0.9, color=MUTED, zorder=1)

    ok = sign_all[~sign_all.sign_mismatch]
    bad = sign_all[sign_all.sign_mismatch]
    ax.scatter(ok.d_real, ok.d_syn, s=13, c=BLUE, alpha=0.45,
               linewidths=0, zorder=2, label=f"Direction agrees ({len(ok)})")
    ax.scatter(bad.d_real, bad.d_syn, s=22, facecolor=VERMILLION,
               edgecolor="white", linewidths=0.5, zorder=3,
               label=f"Direction disagrees ({len(bad)})")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Cohen's $d$, real images", fontsize=9, color=INK)
    ax.set_ylabel("Cohen's $d$, synthetic images", fontsize=9, color=INK)
    style(ax)

    rho = sign_all.groupby("disease").apply(
        lambda d: d.d_syn.corr(d.d_real, method="spearman"), include_groups=False
    ).median()
    ax.text(0.04, 0.95, f"median $\\rho$ = {rho:.3f}", transform=ax.transAxes,
            fontsize=8, color=INK, va="top")
    ax.legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    save(fig, out)


def panel_b(within: pd.DataFrame, out: Path) -> None:
    """Within-syndrome |Cohen's d| between the synthetic and real distributions."""
    order = (
        within.groupby("syndrome").abs_d.median().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(3.1, 3.2))

    data = [within.loc[within.syndrome == s, "abs_d"].to_numpy() for s in order]
    pos = np.arange(len(order))
    bp = ax.boxplot(data, positions=pos, vert=False, widths=0.6,
                    showfliers=False, patch_artist=True, zorder=2)
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor=BLUE, linewidth=1.0)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=BLUE, linewidth=0.9)
    for med in bp["medians"]:
        med.set(color=BLUE, linewidth=1.8)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        ax.scatter(vals, i + rng.uniform(-0.16, 0.16, len(vals)),
                   s=6, color=MUTED, alpha=0.35, linewidths=0, zorder=3)

    ax.axvline(0.2, color=VERMILLION, lw=1.0, ls=(0, (4, 3)), zorder=1)
    share = 100 * (within.abs_d < 0.2).mean()
    ax.legend(handles=[Line2D([], [], color=VERMILLION, lw=1.0, ls=(0, (4, 3)),
                              label=f"$|d| = 0.2$ ({share:.1f}% below)")],
              loc="lower right", fontsize=7, frameon=False, handletextpad=0.5,
              borderpad=0.2)

    ax.set_yticks(pos)
    ax.set_yticklabels([SYNDROME_LABEL[s] for s in order], fontsize=8)
    ax.set_xlabel("$|$Cohen's $d|$, synthetic vs real", fontsize=9, color=INK)
    ax.set_xlim(left=0)
    ax.set_ylim(-1.5, len(order) - 0.4)  # clear band at the bottom for the legend
    ax.grid(axis="x", color=GRID, lw=0.6)
    style(ax)

    fig.tight_layout(pad=0.4)
    save(fig, out)


def panel_c(var_feat: pd.DataFrame, out: Path) -> None:
    """Variance ratio, synthetic over real, by facial region."""
    order = (
        var_feat.groupby("region").var_ratio_syn_over_real.median()
        .sort_values().index
    )
    fig, ax = plt.subplots(figsize=(6.5, 2.3))

    data = [
        var_feat.loc[var_feat.region == r, "var_ratio_syn_over_real"].to_numpy()
        for r in order
    ]
    pos = np.arange(len(order))
    bp = ax.boxplot(data, positions=pos, widths=0.55, showfliers=False,
                    patch_artist=True, zorder=2)
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor=BLUE, linewidth=1.0)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=BLUE, linewidth=0.9)
    for med in bp["medians"]:
        med.set(color=BLUE, linewidth=1.8)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals,
                   s=6, color=MUTED, alpha=0.35, linewidths=0, zorder=3)

    ax.axhline(1.0, color=INK, lw=0.9, zorder=1)
    overall = var_feat.var_ratio_syn_over_real.median()
    ax.axhline(overall, color=VERMILLION, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.legend(handles=[
        Line2D([], [], color=INK, lw=0.9, label="equal variance"),
        Line2D([], [], color=VERMILLION, lw=1.0, ls=(0, (4, 3)),
               label=f"overall median {overall:.2f}"),
    ], loc="upper left", fontsize=7, frameon=False, ncol=2,
       handletextpad=0.5, columnspacing=1.4, borderpad=0.2)

    ax.set_xticks(pos)
    ax.set_xticklabels(
        [f"{REGION_LABEL[r]}\n($n$={int((var_feat.region == r).sum())})" for r in order],
        fontsize=7,
    )
    ax.set_ylabel("Variance ratio\nsynthetic / real", fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(order) - 0.1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    style(ax)

    fig.tight_layout(pad=0.4)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS,
                    help="facekit_analysis/analysis directory (post-audit run)")
    ap.add_argument("--out", type=Path, default=HERE / "image")
    args = ap.parse_args()

    fm = args.analysis / "failure_modes"
    sign_all = pd.read_csv(fm / "sign_all.csv")
    var_feat = pd.read_csv(fm / "variance_ratio_per_feature.csv")

    within = pd.concat(
        pd.read_csv(f).assign(syndrome=f.stem)
        for f in sorted((args.analysis / "per_disease").glob("*.csv"))
    )
    within["abs_d"] = within.cohen_d_syn_vs_real.abs()

    for name, n in (("sign_all", len(sign_all)), ("variance", len(var_feat)),
                    ("per_disease", len(within))):
        if n != 265:
            raise SystemExit(f"{name}: expected 265 syndrome-measurement pairs, got {n}")

    args.out.mkdir(parents=True, exist_ok=True)
    panel_a(sign_all, args.out / "syn_a_effect_scatter")
    panel_b(within, args.out / "syn_b_within_d")
    panel_c(var_feat, args.out / "syn_c_variance_ratio")
    print(f"wrote 3 panels to {args.out}")


if __name__ == "__main__":
    main()
