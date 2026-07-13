"""Feature-validity evaluation: does each geometric feature actually measure the
HPO term it is mapped to?

Two contrasts with opposite failure modes (see PROTOCOL.md):

  CROSS  gold-positive patients pooled across syndromes (capped per syndrome)
         vs patients whose syndrome has the term at low/absent clinical
         frequency.  Low label contamination, HIGH cohort confound
         -> can produce FALSE POSITIVES, so it is judged against a placebo
            null built from the ~110 features of unrelated facial regions.

  WITHIN gold-positive vs gold-silent patients *inside the same syndrome*
         (gold-silent = has >=1 other present HPO but not this one).
         Zero cohort confound, HIGH label contamination
         -> can only DILUTE, never fabricate, so a hit here is trustworthy.

Verdict combines both.  Two calibration controls (one known-good, one
known-bad HPO/feature pair) must land as expected or the run is void.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "labels"
PHENO = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/"
    "mm_fusion_top50/combined_data/phenotypes.csv"
)
OUT = Path(__file__).resolve().parent / "results"

SEED = 42
CAP_PER_SYNDROME = 30      # so pooling actually cancels disease-specific face shape
MIN_POS = 30               # patients, after pose filter + patient collapse
MIN_WITHIN_GROUP = 5       # per syndrome, per arm, before it can enter the WITHIN pool
MIN_ANN_FOR_NEG = 10       # annotated patients a disease needs before its 0% rate means anything

MIN_WITHIN_POS = 30        # below this the WITHIN contrast has no power to fail on

# Calibration. The strongest available positive control is the slant pair: the
# SAME feature (eye_fissure_slant_mean) with OPPOSITE expected directions. A
# cohort effect pushes a feature one way, so it cannot make both contrasts pass;
# only a feature that genuinely tracks the phenotype can.
CONTROL_GOOD = ["HP:0000494", "HP:0000582"]   # Down- / Up-slanted palpebral fissures
CONTROL_BAD = ["HP:0000348"]                  # High forehead -> forehead_height (MISS x3)


def region_of(col: str) -> str:
    """Anatomical region of a feature column. Drives placebo-null exclusion."""
    c = col.lower()
    if c.startswith("eb_"):
        return "eyebrow"
    if c.startswith("eye") or "canthal" in c or "iris" in c or "gaze" in c:
        return "eye"
    if c.startswith("nose") or "nostril" in c or "ala_" in c or "columella" in c or "nasolabial" in c:
        return "nose"
    if "philtrum" in c:
        return "philtrum"
    if "mouth" in c or "lip" in c or "vermilion" in c or "cupid" in c:
        return "mouth"
    if "chin" in c or "jaw" in c:
        return "chin_jaw"
    if "forehead" in c or "hairline" in c:
        return "forehead"
    if "cheek" in c or "malar" in c or "midface" in c:
        return "cheek_midface"
    return "face_global"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg), ties at 0.5. NaN-safe caller."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pheno = pd.read_csv(PHENO)
    gold = pd.read_csv(LABELS / "gold_hpo_facial.csv")
    vocab = pd.read_csv(LABELS / "facial_hpo_vocab.csv")
    freq = pd.read_csv(LABELS / "disease_hpo_freq.csv")
    omap = pd.read_csv(LABELS / "disease_omim_map.csv")

    df = pheno.merge(gold, on="image_id", how="inner", validate="one_to_one")
    df = df[df["frontal_ok"]].copy()

    def parse(cell) -> set[str]:
        if pd.isna(cell):
            return set()
        return {h.strip() for h in str(cell).split(";") if h.strip()}

    df["present"] = df["present_facial_hpo"].map(parse)
    df["absent"] = df["absent_facial_hpo"].map(parse)

    feat_cols = [
        c for c in pheno.columns
        if c not in ("disease", "image_id", "pose_yaw", "pose_pitch", "pose_roll", "frontal_ok")
    ]

    # --- collapse to patient level: median feature, union of HPO labels ---
    agg = df.groupby("patient_id").agg(
        disease=("disease", "first"),
        present=("present", lambda s: set().union(*s)),
        absent=("absent", lambda s: set().union(*s)),
        n_img=("image_id", "size"),
    )
    med = df.groupby("patient_id")[feat_cols].median()
    pat = agg.join(med).reset_index()
    pat["n_present"] = pat["present"].map(len)

    # HPOA is far too sparse to define negatives on its own (median 9 HPO per
    # disease; hypertelorism is listed for only 24/50 diseases yet is *observed*
    # in 22/39). Treating "not listed" as negative floods the negative arm with
    # true positives and destroys the positive control. So a disease counts as a
    # negative for term X only if BOTH:
    #   (a) HPOA does not list X for it at freq > 0.05, and
    #   (b) among its annotated patients, the observed gold-positive rate for X
    #       is < 5% (needs >= MIN_ANN_FOR_NEG annotated patients to be meaningful).
    freq = freq.merge(omap, on=["omim", "disease"], how="inner")
    hi = freq[freq["freq_value"] > 0.05]
    hpoa_hpo = hi.groupby("disease")["hpo_id"].agg(set).to_dict()

    ann = pat[pat["n_present"] >= 1]
    neg_ok: dict[str, set[str]] = {}
    for hpo in vocab["hpo_id"]:
        ok = set()
        for dis, sub in ann.groupby("disease"):
            if len(sub) < MIN_ANN_FOR_NEG:
                continue
            rate = sub["present"].map(lambda s: hpo in s).mean()
            if rate < 0.05 and hpo not in hpoa_hpo.get(dis, set()):
                ok.add(dis)
        neg_ok[hpo] = ok

    return pat, vocab, pd.DataFrame({"feat": feat_cols}), neg_ok


def cap(g: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Cap each syndrome's contribution so pooling is not dominated by one disease."""
    out = []
    for _, sub in g.groupby("disease"):
        if len(sub) > CAP_PER_SYNDROME:
            sub = sub.iloc[rng.choice(len(sub), CAP_PER_SYNDROME, replace=False)]
        out.append(sub)
    return pd.concat(out) if out else g.iloc[:0]


def contrast(pos: pd.DataFrame, neg: pd.DataFrame, target: str,
             direction: int, feat_cols: list[str]) -> dict:
    """Directed AUC of target + placebo null over unrelated-region features."""
    tgt_region = region_of(target)

    def directed(col: str, sign: int) -> float:
        p = pos[col].to_numpy(dtype=float) * sign
        n = neg[col].to_numpy(dtype=float) * sign
        p, n = p[np.isfinite(p)], n[np.isfinite(n)]
        return auc(p, n)

    a = directed(target, direction)
    if not np.isfinite(a):
        return dict(auc=np.nan, null_pct=np.nan, p_emp=np.nan, n_null=0)

    # placebo null: |AUC - 0.5| over features from OTHER anatomical regions
    null = []
    for c in feat_cols:
        if c == target or region_of(c) == tgt_region:
            continue
        v = directed(c, 1)
        if np.isfinite(v):
            null.append(abs(v - 0.5))
    null = np.asarray(null)
    dev = abs(a - 0.5)
    p_emp = float((null >= dev).mean()) if len(null) else np.nan
    null_pct = float((null < dev).mean() * 100) if len(null) else np.nan
    return dict(auc=a, null_pct=null_pct, p_emp=p_emp, n_null=len(null))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pat, vocab, fc, neg_ok = load()
    feat_cols = fc["feat"].tolist()

    print(f"patients (frontal_ok, collapsed): {len(pat)}")
    print(f"features: {len(feat_cols)} | vocab terms: {len(vocab)}")

    rows = []
    for _, v in vocab.iterrows():
        hpo, col, d = v["hpo_id"], v["csv_column"], int(v["expected_direction"])
        if col not in feat_cols:
            continue

        is_pos = pat["present"].map(lambda s: hpo in s)
        pos_all = pat[is_pos]

        # ---- CROSS: pooled positives vs verified-negative syndromes ----
        # negatives restricted to *annotated* patients of diseases where the term
        # is both absent from HPOA and observed at <5% in the gold labels.
        neg_mask = (~is_pos) & (pat["n_present"] >= 1) & pat["disease"].isin(neg_ok[hpo])
        pos_c = cap(pos_all, rng)
        neg_c = cap(pat[neg_mask], rng)

        r = dict(
            hpo_id=hpo, hpo_name=v["hpo_name"], feature=col,
            region=region_of(col), direction=d,
            n_pos=len(pos_all), n_pos_capped=len(pos_c), n_neg_capped=len(neg_c),
            n_syndromes_pos=pos_all["disease"].nunique(),
            n_syndromes_neg=len(neg_ok[hpo]),
        )
        if len(pos_all) < MIN_POS or len(neg_c) < MIN_POS:
            r["verdict"] = "UNDERPOWERED"
            rows.append(r)
            continue

        cx = contrast(pos_c, neg_c, col, d, feat_cols)
        r.update(auc_cross=cx["auc"], null_pct_cross=cx["null_pct"], p_emp_cross=cx["p_emp"])

        # attribution: does ANY feature of the same region capture the phenotype?
        # best >> mapped  -> the HPO->feature mapping / formula is wrong
        # best also ~0.5  -> the region's landmarks are broken
        best, best_auc = None, 0.5
        for c2 in feat_cols:
            if region_of(c2) != r["region"]:
                continue
            for sgn in (1, -1):
                p2 = pos_c[c2].to_numpy(dtype=float) * sgn
                n2 = neg_c[c2].to_numpy(dtype=float) * sgn
                p2, n2 = p2[np.isfinite(p2)], n2[np.isfinite(n2)]
                a2 = auc(p2, n2)
                if np.isfinite(a2) and a2 > best_auc:
                    best, best_auc = c2, a2
        r["best_in_region"] = best
        r["auc_best_in_region"] = best_auc

        # ---- WITHIN: same syndrome, gold-positive vs gold-silent ----
        # gold-silent = annotated with >=1 other facial HPO but not this one.
        # Per-syndrome AUCs are too small to beat a placebo null, so instead
        # z-score every feature *inside* each syndrome (which is what cancels the
        # cohort confound) and only then pool across syndromes. Power now scales
        # with the total, not with the smallest cohort.
        zp, zn, syns = [], [], 0
        for dis, sub in pat.groupby("disease"):
            m = sub["present"].map(lambda s: hpo in s)
            sp, sn = sub[m], sub[(~m) & (sub["n_present"] >= 1)]
            if len(sp) < MIN_WITHIN_GROUP or len(sn) < MIN_WITHIN_GROUP:
                continue
            ref = pd.concat([sp, sn])
            mu, sd = ref[feat_cols].mean(), ref[feat_cols].std().replace(0, np.nan)
            zp.append((sp[feat_cols] - mu) / sd)
            zn.append((sn[feat_cols] - mu) / sd)
            syns += 1
        if syns:
            zpos, zneg = pd.concat(zp), pd.concat(zn)
            c = contrast(zpos, zneg, col, d, feat_cols)
            r["auc_within"] = c["auc"]
            r["null_pct_within"] = c["null_pct"]
            r["p_emp_within"] = c["p_emp"]
            r["n_within_syn"] = syns
            r["n_within_pos"] = len(zpos)
            r["n_within_neg"] = len(zneg)
        else:
            r["auc_within"] = np.nan
            r["n_within_syn"] = 0
            r["n_within_pos"] = 0

        rows.append(r)

    res = pd.DataFrame(rows)

    # ---- verdicts ----
    def verdict(r) -> str:
        if r.get("verdict") == "UNDERPOWERED":
            return "UNDERPOWERED"
        a, np_ = r["auc_cross"], r["null_pct_cross"]
        if not np.isfinite(a):
            return "UNDERPOWERED"
        if a < 0.5 and np_ >= 95:
            return "INVERTED"
        cross_ok = (a > 0.65) and (np_ >= 95)
        w, wnp = r["auc_within"], r.get("null_pct_within", np.nan)
        # dilution by false negatives shrinks the WITHIN effect size but not its
        # specificity, so gate on the placebo null + correct direction, not on a
        # large absolute AUC.
        within_ok = np.isfinite(w) and (w > 0.5) and (wnp >= 95)
        # the WITHIN arm is diluted by false negatives; if it is also small it
        # simply has no power, and its silence must not be read as a refutation.
        within_powered = np.isfinite(w) and r.get("n_within_pos", 0) >= MIN_WITHIN_POS
        if within_ok:
            return "VALIDATED"          # zero-confound contrast passed -> trustworthy
        if cross_ok and within_powered:
            return "LIKELY_COHORT"      # clean contrast had power and refused
        if cross_ok:
            return "CROSS_ONLY"         # clean contrast had no power -> inconclusive
        return "NO_SIGNAL"

    res["verdict"] = res.apply(verdict, axis=1)
    res = res.sort_values(["verdict", "auc_cross"], ascending=[True, False])
    res.to_csv(OUT / "feature_validity.csv", index=False)

    # ---- calibration controls ----
    print("\n=== CALIBRATION CONTROLS ===")
    for hpo in CONTROL_GOOD + CONTROL_BAD:
        want = "should PASS" if hpo in CONTROL_GOOD else "should be NO_SIGNAL"
        m = res[res["hpo_id"] == hpo]
        if m.empty:
            print(f"  {hpo}: NOT IN VOCAB — cannot calibrate")
            continue
        r = m.iloc[0]
        print(f"  {r['hpo_name']:<30} dir={r['direction']:+d} {r['feature']:<24} "
              f"AUC={r.get('auc_cross', float('nan')):.3f} null%={r.get('null_pct_cross', float('nan')):5.1f} "
              f"-> {r['verdict']:<14} ({want})")

    print("\n=== VERDICT COUNTS ===")
    print(res["verdict"].value_counts().to_string())

    print("\n=== BY REGION (testable terms only) ===")
    t = res[res["verdict"] != "UNDERPOWERED"]
    if len(t):
        tab = t.groupby("region")["verdict"].value_counts().unstack(fill_value=0)
        tab["n"] = tab.sum(axis=1)
        tab["pass_rate"] = (tab.get("VALIDATED", 0) / tab["n"] * 100).round(0)
        print(tab.sort_values("pass_rate").to_string())

    print(f"\nwrote {OUT / 'feature_validity.csv'}")


if __name__ == "__main__":
    main()
