"""Step 1: re-split gold-positive patients + z-score base.

Test = 100 gold-positive patients, stratified by disease (proportional, >=1 per
disease where possible). Tune = the remaining gold-positive patients. Z-score
mean/std fit on ALL non-test patients (skipna). Local seeded numpy Generator.
"""
import numpy as np
import pandas as pd

import fcrm_data
import config


def _allocate(n_per_disease: dict[str, int], n_test: int) -> dict[str, int]:
    """Largest-remainder proportional allocation with >=1 per disease, capped by supply."""
    diseases = list(n_per_disease)
    total = sum(n_per_disease.values())
    assert len(diseases) <= n_test, "more diseases than test slots; cannot give >=1 each"
    # start at 1 each (every gold-pos disease has >=1 patient), distribute the rest by quota.
    alloc = {d: 1 for d in diseases}
    remaining = n_test - len(diseases)
    quotas = {d: (n_per_disease[d] / total) * n_test for d in diseases}
    # rank by fractional remainder of the proportional quota, capacity = supply - current alloc
    order = sorted(diseases, key=lambda d: (quotas[d] - int(quotas[d])), reverse=True)
    i = 0
    while remaining > 0:
        d = order[i % len(order)]
        if alloc[d] < n_per_disease[d]:
            alloc[d] += 1
            remaining -= 1
        i += 1
        if i > 10 * n_test:  # safety; cannot happen given total >> n_test
            break
    assert sum(alloc.values()) == n_test
    return alloc


def run(run_dir, pooled: pd.DataFrame, feats: list[str], seed: int):
    rng = np.random.default_rng(seed)

    gold_pos = pooled[pooled["present"].apply(len) > 0].copy()
    n_per = gold_pos.groupby("disease")["patient_id"].nunique().to_dict()
    print(f"[step1] gold-positive patients: {len(gold_pos)} across {len(n_per)} diseases "
          f"(of {pooled['disease'].nunique()} total)")

    alloc = _allocate(n_per, config.N_TEST)

    test_ids: list[int] = []
    for disease in sorted(alloc):
        ids = gold_pos.loc[gold_pos["disease"] == disease, "patient_id"].to_numpy().copy()
        rng.shuffle(ids)  # local generator
        test_ids.extend(int(x) for x in ids[: alloc[disease]])
    test_set = set(test_ids)
    tune_ids = [int(p) for p in gold_pos["patient_id"] if int(p) not in test_set]

    split = pd.DataFrame(
        [(p, gold_pos.set_index("patient_id").at[p, "disease"],
          "test" if p in test_set else "tune")
         for p in gold_pos["patient_id"].astype(int)],
        columns=["patient_id", "disease", "split"],
    )
    split.to_csv(run_dir / "gold_split.csv", index=False)

    # z-score base = all non-test patients (gold-pos + gold-neg)
    mean, std = fcrm_data.zscore_base_stats(pooled, test_set, feats)
    pd.DataFrame({"mean": mean, "std": std}).to_csv(run_dir / "zscore_stats.csv")
    z = fcrm_data.compute_z(pooled, mean, std, feats)

    # verify
    assert not (test_set & set(tune_ids)), "patient in both test and tune"
    n_base = int((~pooled["patient_id"].isin(test_set)).sum())
    per_disease = (split[split["split"] == "test"].groupby("disease").size()
                   .sort_index())
    print(f"[step1] test={len(test_set)} tune={len(tune_ids)} "
          f"(sum={len(test_set) + len(tune_ids)} == gold-pos {len(gold_pos)})")
    print(f"[step1] z-score base (non-test) patients: {n_base}")
    print("[step1] per-disease test allocation:")
    print(per_disease.to_string())
    print(f"[step1] wrote gold_split.csv, zscore_stats.csv")
    return split, test_set, tune_ids, mean, std, z
