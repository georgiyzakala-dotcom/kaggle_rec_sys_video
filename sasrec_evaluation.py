"""Exact fold metrics from bounded user shards; no policy/checkpoint selection."""

from __future__ import annotations

import polars as pl

from sasrec_selection import KEYS, POLICIES, SOURCES, standalone_top20, validate_unseen
from validation import (
    validate_candidate_output,
    validate_final_recommendations,
    validate_no_nulls,
    validate_unique_keys,
)

DEPTHS = (50, 100, 150, 200)
POLICY_CAPS = {
    "baseline_800": (200, 0),
    **POLICIES,
    "add_source_1000": (200, 200),
    "als400_control_1000": (400, 0),
}


def counts(users, pairs, name):
    return users.join(
        pairs.group_by("user_id").len(name=name), on="user_id", how="left"
    ).select(pl.col(name).fill_null(0).cast(pl.UInt32))[name]


def evaluate_shard(users, truth, sources, native, als400, store):
    """Persist sufficient per-user counts, never average shard-level ratios."""
    users = users.sort("user_id")
    validate_no_nulls(users, name="evaluation users")
    validate_unique_keys(users, keys=("user_id",), name="evaluation users")
    validate_unique_keys(truth, keys=KEYS, name="evaluation truth")
    validate_unseen(truth, users, store)
    if set(sources) != set(SOURCES):
        raise ValueError("all four fixed sources are required")
    frames = {**sources, "sasrec": native, "als400": als400}
    for name, frame in frames.items():
        validate_candidate_output(
            frame,
            k=400 if name == "als400" else 200,
            source_name="implicit_als" if name == "als400" else name,
        )
        validate_unseen(frame, users, store)
    # The larger-budget control must use exactly the original fitted ALS.
    prefix = als400.filter(pl.col("rank") <= 200).select(*KEYS, "rank")
    if not prefix.sort("user_id", "rank").equals(
        sources["implicit_als"].select(*KEYS, "rank").sort("user_id", "rank")
    ):
        raise ValueError("ALS400 top200 differs from frozen baseline candidates")

    stats = users.with_columns(counts(users, truth, "relevant"))
    views = {}
    for name, frame in {**sources, "sasrec": native}.items():
        for depth in DEPTHS:
            views[f"{name}_at_{depth}"] = frame.filter(pl.col("rank") <= depth).select(
                KEYS
            )
    views["implicit_als_at_400"] = als400.select(KEYS)
    for depth in DEPTHS[:-1]:
        views[f"baseline_at_{depth}_per_source"] = pl.concat(
            [sources[s].filter(pl.col("rank") <= depth).select(KEYS) for s in SOURCES]
        ).unique()
    other = pl.concat([sources[s].select(KEYS) for s in SOURCES[:-1]]).unique()
    for name, (a_cap, s_cap) in POLICY_CAPS.items():
        views[name] = pl.concat(
            [
                other,
                als400.filter(pl.col("rank") <= a_cap).select(KEYS),
                native.filter(pl.col("rank") <= s_cap).select(KEYS),
            ]
        ).unique()
    for name, pairs in views.items():
        hits = pairs.join(truth, on=KEYS, how="semi")
        stats = stats.with_columns(
            counts(users, pairs, f"m__{name}__count"),
            counts(users, hits, f"m__{name}__hits"),
        )
    for name in SOURCES:
        common = native.select(KEYS).join(
            sources[name].select(KEYS), on=KEYS, how="semi"
        )
        stats = stats.with_columns(
            counts(users, common, f"o__{name}__common"),
            counts(
                users,
                common.join(truth, on=KEYS, how="semi"),
                f"o__{name}__shared_hits",
            ),
        )
    needed = (
        sources["implicit_als"]
        .select(KEYS)
        .join(views["replace_als"], on=KEYS, how="anti")
        .join(truth, on=KEYS, how="semi")
    )
    stats = stats.with_columns(counts(users, needed, "als_still_needed_hits"))
    recommendations = {}
    for name, frame in {**sources, "sasrec": native}.items():
        recs = standalone_top20(frame, sources["global_popularity"], users)
        validate_final_recommendations(recs, expected_k=20)
        pairs = recs.explode("item_ids", empty_as_null=False).rename(
            {"item_ids": "item_id"}
        )
        validate_unseen(pairs, users, store)
        stats = stats.with_columns(
            counts(users, pairs.join(truth, on=KEYS, how="semi"), f"p__{name}__hits")
        )
        recommendations[name] = recs
    return stats, recommendations


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarize_counts(stats, name):
    count, hits = stats[f"m__{name}__count"], stats[f"m__{name}__hits"]
    labeled = int((stats["relevant"] > 0).sum())
    oracle_hits = int(hits.clip(upper_bound=20).sum())
    return {
        "candidate_recall": ratio(int(hits.sum()), int(stats["relevant"].sum())),
        "candidate_recall_macro_labeled_users": (
            stats.filter(pl.col("relevant") > 0)
            .select((pl.col(f"m__{name}__hits") / pl.col("relevant")).mean())
            .item()
            or 0.0
        ),
        "positive_hits": int(hits.sum()),
        "candidate_pairs": int(count.sum()),
        "candidate_user_hit_rate": ratio(int((hits > 0).sum()), labeled),
        "candidate_oracle_p20_all_targets": ratio(oracle_hits, 20 * stats.height),
        "candidate_oracle_p20_labeled_users": ratio(oracle_hits, 20 * labeled),
        "coverage": float((count > 0).mean()),
        "mean_candidate_count": float(count.mean()),
        **{
            f"p{q}_candidate_count": float(
                count.quantile(q / 100, interpolation="nearest")
            )
            for q in (50, 90, 95, 99)
        },
    }


def fraction_summary(numerator, denominator, mask=None):
    valid = denominator > 0
    if mask is not None:
        valid = valid & mask
    values = numerator.filter(valid) / denominator.filter(valid)
    return {
        "mean": values.mean(),
        "p50": values.median(),
        "p90": values.quantile(0.9),
        "defined_users": len(values),
        "undefined_users": len(numerator) - len(values),
    }


def summarize(stats, frozen_policy):
    validate_unique_keys(stats, keys=("user_id",), name="fold per-user metrics")
    validate_no_nulls(stats, name="fold per-user metrics")
    if stats.is_empty() or frozen_policy not in POLICIES:
        raise ValueError("invalid evaluation universe or frozen policy")
    measures = {
        c.split("__")[1]: summarize_counts(stats, c.split("__")[1])
        for c in stats.columns
        if c.startswith("m__") and c.endswith("__count")
    }
    baseline = measures["baseline_800"]
    policies = {}
    for name, (a_cap, s_cap) in POLICY_CAPS.items():
        value = measures[name]
        policies[name] = {
            **value,
            "als_cap": a_cap,
            "sasrec_cap": s_cap,
            "sum_source_caps": 600 + a_cap + s_cap,
            "delta_recall": value["candidate_recall"] - baseline["candidate_recall"],
            "delta_positive_hits": value["positive_hits"] - baseline["positive_hits"],
            "delta_mean_candidate_count": value["mean_candidate_count"]
            - baseline["mean_candidate_count"],
        }
    overlaps = {}
    left = stats["m__sasrec_at_200__count"].cast(pl.Int64)
    a_hits = measures["sasrec_at_200"]["positive_hits"]
    for source in SOURCES:
        right = stats[f"m__{source}_at_200__count"].cast(pl.Int64)
        common = stats[f"o__{source}__common"].cast(pl.Int64)
        shared = int(stats[f"o__{source}__shared_hits"].sum())
        b_hits = measures[f"{source}_at_200"]["positive_hits"]
        union = left + right - common
        both = (left > 0) & (right > 0)
        denominators = {
            "jaccard": union,
            "share_left_in_right": left,
            "share_right_in_left": right,
        }
        overlaps[source] = {
            "left_pairs": int(left.sum()),
            "right_pairs": int(right.sum()),
            "intersection_pairs": int(common.sum()),
            "micro_jaccard": ratio(int(common.sum()), int(union.sum()))
            if union.sum()
            else None,
            "left_positive_hits": a_hits,
            "right_positive_hits": b_hits,
            "shared_positive_hits": shared,
            "left_only_positive_hits": a_hits - shared,
            "right_only_positive_hits": b_hits - shared,
            "shared_share_of_left_hits": shared / a_hits if a_hits else None,
            "shared_share_of_right_hits": shared / b_hits if b_hits else None,
            "per_user": {
                n: fraction_summary(common, d) for n, d in denominators.items()
            },
            "both_sources_nonempty_users": int(both.sum()),
            "per_user_both_sources_nonempty": {
                n: fraction_summary(common, d, both) for n, d in denominators.items()
            },
        }
    labeled = int((stats["relevant"] > 0).sum())
    top20 = {
        name: {
            "precision_at_20_all_targets": ratio(
                int(stats[f"p__{name}__hits"].sum()), 20 * stats.height
            ),
            "precision_at_20_labeled_users": ratio(
                int(stats[f"p__{name}__hits"].sum()), 20 * labeled
            ),
        }
        for name in (*SOURCES, "sasrec")
    }
    add, control = policies["add_source_1000"], policies["als400_control_1000"]
    return {
        "target_users": stats.height,
        "labeled_users": labeled,
        "positive_pairs": int(stats["relevant"].sum()),
        "sources": {
            name: {str(k): measures[f"{name}_at_{k}"] for k in DEPTHS}
            for name in (*SOURCES, "sasrec")
        },
        "baseline_depth_curve": {
            str(k): measures[f"baseline_at_{k}_per_source"] for k in DEPTHS[:-1]
        }
        | {"200": baseline},
        "als_at_400": measures["implicit_als_at_400"],
        "policies": policies,
        "frozen_policy": frozen_policy,
        "frozen_policy_delta_recall": policies[frozen_policy]["delta_recall"],
        "policy_selected_on_this_fold": False,
        "new_positive_hits_vs_all_sources": add["delta_positive_hits"],
        "als_still_needed_hits": int(stats["als_still_needed_hits"].sum()),
        "add_source_minus_als400_recall": add["candidate_recall"]
        - control["candidate_recall"],
        "overlap_sasrec_vs_sources": overlaps,
        "standalone_with_global_fallback": top20,
        "candidate_recall": measures["sasrec_at_200"]["candidate_recall"],
        "coverage": measures["sasrec_at_200"]["coverage"],
        **top20["sasrec"],
        "precision_semantics": "standalone source top20 with global fallback; not a trained ranker score",
    }
