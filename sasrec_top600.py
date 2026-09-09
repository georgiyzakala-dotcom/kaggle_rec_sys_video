"""Metrics at depths up to 600 from frozen SASRec and ALS predictions."""

from __future__ import annotations

import polars as pl

from sasrec_evaluation import counts, fraction_summary, summarize_counts
from sasrec_selection import KEYS, SOURCES, validate_unseen
from sasrec_top300 import extend_shard, summarize_extension
from validation import validate_candidate_output

PRIMARY = "als600_sasrec600_1800"
DEEP_POLICIES = {
    "als600_only_1200": (600, 0),
    "sasrec600_only_1200": (0, 600),
    "als400_sasrec200_1200": (400, 200),
    "als200_sasrec400_1200": (200, 400),
    "als400_sasrec400_1400": (400, 400),
    "als600_sasrec200_1400": (600, 200),
    "als200_sasrec600_1400": (200, 600),
    "als600_sasrec300_1500": (600, 300),
    "als300_sasrec600_1500": (300, 600),
    PRIMARY: (600, 600),
}


def require_prefix(deeper, original, k, source):
    columns = [*KEYS, "rank"]
    if not (
        deeper.filter(pl.col("rank") <= k)
        .select(columns)
        .sort("user_id", "rank")
        .equals(original.select(columns).sort("user_id", "rank"))
    ):
        raise ValueError(f"{source} top{k} differs from frozen predictions")


def evaluate_deep_shard(
    users, truth, sources, sas200, als400, sas600, als600, store, old_stats
):
    users = users.sort("user_id")
    for source, frame, old, depth in (
        ("sasrec", sas600, sas200, 200),
        ("implicit_als", als600, als400, 400),
    ):
        validate_candidate_output(frame, k=600, source_name=source)
        validate_unseen(frame, users, store)
        require_prefix(frame, old, depth, source)
    stats = extend_shard(
        users,
        truth,
        sources,
        sas200,
        als400,
        sas600.filter(pl.col("rank") <= 300),
        store,
        old_stats,
    )
    other = pl.concat([sources[s].select(KEYS) for s in SOURCES[:-1]]).unique()
    views = {}
    for source, frame in (("sasrec", sas600), ("implicit_als", als600)):
        for k in (400, 600):
            views[f"{source}_at_{k}"] = frame.filter(pl.col("rank") <= k).select(KEYS)
    for policy, (a, s) in DEEP_POLICIES.items():
        views[policy] = pl.concat(
            [
                other,
                als600.filter(pl.col("rank") <= a).select(KEYS),
                sas600.filter(pl.col("rank") <= s).select(KEYS),
            ]
        ).unique()
    for name, pairs in views.items():
        stats = stats.with_columns(
            counts(users, pairs, f"m__{name}__count"),
            counts(users, pairs.join(truth, on=KEYS, how="semi"), f"m__{name}__hits"),
        )
    for k in (400, 600):
        common = views[f"sasrec_at_{k}"].join(
            views[f"implicit_als_at_{k}"], on=KEYS, how="semi"
        )
        stats = stats.with_columns(
            counts(users, common, f"o{k}_common"),
            counts(users, common.join(truth, on=KEYS, how="semi"), f"o{k}_shared_hits"),
        )
    return stats


def summarize_deep(stats, frozen_policy):
    result = summarize_extension(stats, frozen_policy)
    old_union = result["policies"]["add_source_1000"]
    baseline = result["policies"]["baseline_800"]
    for name, (a, s) in DEEP_POLICIES.items():
        m = summarize_counts(stats, name)
        result["policies"][name] = {
            **m,
            "als_cap": a,
            "sasrec_cap": s,
            "sum_source_caps": 600 + a + s,
            "delta_recall_vs_baseline800": m["candidate_recall"]
            - baseline["candidate_recall"],
            "delta_recall_vs_both200": m["candidate_recall"]
            - old_union["candidate_recall"],
            "delta_positive_hits_vs_both200": m["positive_hits"]
            - old_union["positive_hits"],
            "delta_mean_candidate_count_vs_both200": m["mean_candidate_count"]
            - old_union["mean_candidate_count"],
        }
    result["sources_depth_curve"] = {
        s: {
            str(k): summarize_counts(stats, f"{s}_at_{k}")
            for k in (50, 100, 150, 200, 300, 400, 600)
        }
        for s in ("sasrec", "implicit_als")
    }
    result["overlap_at_depth"] = {}
    for k in (200, 300, 400, 600):
        a, b = (
            stats[f"m__{s}_at_{k}__count"].cast(pl.Int64)
            for s in ("sasrec", "implicit_als")
        )
        common = stats["o__implicit_als__common" if k == 200 else f"o{k}_common"].cast(
            pl.Int64
        )
        shared = int(
            stats[
                "o__implicit_als__shared_hits" if k == 200 else f"o{k}_shared_hits"
            ].sum()
        )
        a_hits, b_hits = (
            result["sources_depth_curve"][s][str(k)]["positive_hits"]
            for s in ("sasrec", "implicit_als")
        )
        union = int((a + b - common).sum())
        result["overlap_at_depth"][str(k)] = {
            "intersection_pairs": int(common.sum()),
            "micro_jaccard": int(common.sum()) / union if union else None,
            "shared_positive_hits": shared,
            "sasrec_only_positive_hits": a_hits - shared,
            "als_only_positive_hits": b_hits - shared,
            "shared_share_of_sasrec_hits": shared / a_hits if a_hits else None,
            "shared_share_of_als_hits": shared / b_hits if b_hits else None,
            "per_user": {
                "jaccard": fraction_summary(common, a + b - common),
                "share_sasrec_in_als": fraction_summary(common, a),
                "share_als_in_sasrec": fraction_summary(common, b),
            },
        }
    policies = result["policies"]
    result.update(
        primary_policy=PRIMARY,
        candidate_recall=policies[PRIMARY]["candidate_recall"],
        coverage=policies[PRIMARY]["coverage"],
        als_inference_reused=False,
        frozen_model_weights_reused=True,
        equal_budget1200_delta_both300_minus_als600=(
            policies["als300_sasrec300_1200"]["candidate_recall"]
            - policies["als600_only_1200"]["candidate_recall"]
        ),
        budgets_warning="both600 uses 1800 source slots; both300 and ALS600-only use 1200; other sources stay at 200 each",
    )
    return result
