"""Step 0: feature filter -> the 83-HPO filtered direction-code vocab.

Drops direction codes with expected_direction == 0, confidence == 'LOW', or a
feature_group in config.DROP_FEATURE_GROUPS. Saves filtered_direction_codes.csv.
"""
import common
import config


def run(run_dir):
    dc = common.load_direction_codes()
    n0 = len(dc)

    drop_dir0 = dc["expected_direction"] == 0
    drop_low = dc["confidence"] == "LOW"
    drop_grp = dc["feature_group"].isin(config.DROP_FEATURE_GROUPS)
    keep = ~(drop_dir0 | drop_low | drop_grp)
    filtered = dc[keep].reset_index(drop=True)

    out = run_dir / "filtered_direction_codes.csv"
    filtered.to_csv(out, index=False)
    vocab = sorted(filtered["hpo_id"].unique().tolist())

    print(f"[step0] direction codes: {n0} -> {len(filtered)} (vocab size {len(vocab)})")
    print(f"[step0]   dropped expected_direction==0: {int(drop_dir0.sum())}")
    print(f"[step0]   dropped confidence==LOW:       {int(drop_low.sum())}")
    print(f"[step0]   dropped feature_group in {list(config.DROP_FEATURE_GROUPS)}: "
          f"{int(drop_grp.sum())}")
    print(f"[step0]   (union of the three drop masks: {int((~keep).sum())})")
    print(f"[step0] wrote {out}")
    return filtered, vocab
