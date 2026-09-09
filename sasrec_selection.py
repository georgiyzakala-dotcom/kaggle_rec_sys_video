"""Next-day SASRec selection metrics with fixed source budgets and user universe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from interfaces import FINAL_RECOMMENDATION_SCHEMA
from metrics import evaluate_candidate_metrics, evaluate_precision_at_20
from sasrec_data import SequenceStore
from validation import validate_candidate_output

SOURCES = ("global_popularity", "recency_popularity", "item2item", "implicit_als")
# All four alternatives retain the three other sources at 200 each.
POLICIES = {
    "replace_als": (0, 200),
    "blend_150_50": (150, 50),
    "blend_100_100": (100, 100),
    "blend_50_150": (50, 150),
}
KEYS = ["user_id", "item_id"]


def id_sample(targets: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    """Sampling is independent of history length, labels and candidate coverage."""
    return (
        targets.with_columns(pl.col("user_id").hash(seed=seed).alias("sample"))
        .sort("sample", "user_id")
        .head(count)
        .select("user_id")
        .sort("user_id")
    )


def validate_unseen(candidates, users, store: SequenceStore):
    if (
        candidates.select("user_id")
        .unique()
        .join(users, on="user_id", how="anti")
        .height
    ):
        raise ValueError("candidate user outside the fixed selection universe")
    ids = candidates["item_id"].to_numpy()
    item_indices = np.searchsorted(store.item_ids, ids)
    if (item_indices >= len(store.item_ids)).any() or not np.array_equal(
        store.item_ids[item_indices], ids
    ):
        raise ValueError("unknown history item in selection candidates")
    for key, frame in candidates.partition_by("user_id", as_dict=True).items():
        user = np.uint64(key[0])
        index = int(np.searchsorted(store.user_ids, user))
        if index < len(store.user_ids) and store.user_ids[index] == user:
            items = np.searchsorted(store.item_ids, frame["item_id"].to_numpy()) + 1
            if np.intersect1d(items, store.seen_for(index)).size:
                raise ValueError("selection candidates contain a history-seen pair")


def overlap(left, right, truth, users):
    a, b = left.select(KEYS).unique(), right.select(KEYS).unique()
    common = a.join(b, on=KEYS, how="semi")
    a_hits = a.join(truth, on=KEYS, how="semi")
    b_hits = b.join(truth, on=KEYS, how="semi")
    shared_hits = common.join(truth, on=KEYS, how="semi").height
    counts = users.join(
        a.group_by("user_id").len(name="left"), on="user_id", how="left"
    )
    counts = counts.join(
        b.group_by("user_id").len(name="right"), on="user_id", how="left"
    )
    counts = counts.join(
        common.group_by("user_id").len(name="common"), on="user_id", how="left"
    ).fill_null(0)
    counts = counts.with_columns(
        (pl.col("left") + pl.col("right") - pl.col("common")).alias("union")
    )
    fractions = {}
    for name, denominator in (
        ("jaccard", "union"),
        ("share_left_in_right", "left"),
        ("share_right_in_left", "right"),
    ):
        defined = counts.filter(pl.col(denominator) > 0)
        values = defined.select(
            (pl.col("common") / pl.col(denominator)).alias("fraction")
        )["fraction"]
        fractions[name] = {
            "mean": values.mean(),
            "p50": values.median(),
            "p90": values.quantile(0.9),
            "defined_users": defined.height,
            "undefined_users": users.height - defined.height,
        }
    union = a.height + b.height - common.height
    return {
        "left_pairs": a.height,
        "right_pairs": b.height,
        "intersection_pairs": common.height,
        "micro_jaccard": common.height / union if union else None,
        "left_positive_hits": a_hits.height,
        "right_positive_hits": b_hits.height,
        "shared_positive_hits": shared_hits,
        "left_only_positive_hits": a_hits.height - shared_hits,
        "right_only_positive_hits": b_hits.height - shared_hits,
        "shared_share_of_left_hits": shared_hits / a_hits.height
        if a_hits.height
        else None,
        "shared_share_of_right_hits": shared_hits / b_hits.height
        if b_hits.height
        else None,
        "per_user": fractions,
    }


def standalone_top20(native, fallback, users):
    # Native ordering first, independent global-popularity fallback second.
    rows = pl.concat(
        [
            native.select(*KEYS, "rank").with_columns(pl.lit(0).alias("priority")),
            fallback.select(*KEYS, "rank").with_columns(pl.lit(1).alias("priority")),
        ]
    ).sort("user_id", "priority", "rank", "item_id")
    rows = rows.unique(subset=KEYS, keep="first", maintain_order=True)
    recommendations = rows.group_by("user_id", maintain_order=True).agg(
        pl.col("item_id").head(20).alias("item_ids")
    )
    recommendations = recommendations.sort("user_id").cast(FINAL_RECOMMENDATION_SCHEMA)
    if (
        not recommendations.select("user_id").equals(users.sort("user_id"))
        or recommendations.filter(pl.col("item_ids").list.len() != 20).height
    ):
        raise ValueError("global fallback cannot fill exactly 20 for every target user")
    return recommendations


@dataclass
class SelectionEvaluator:
    users: pl.DataFrame
    truth: pl.DataFrame
    sources: dict[str, pl.DataFrame]
    store: SequenceStore

    def __post_init__(self):
        if set(self.sources) != set(SOURCES):
            raise ValueError("selection requires all four fixed candidate sources")
        if self.truth.is_empty():
            raise ValueError("selection universe has no next-day positive labels")
        validate_unseen(self.truth, self.users, self.store)
        self.other = pl.concat(
            [self.sources[s].select(KEYS) for s in SOURCES[:-1]]
        ).unique()
        self.baseline = pl.concat(
            [self.other, self.sources["implicit_als"].select(KEYS)]
        ).unique()
        self.baseline_metrics = self.measure(self.baseline)
        self.source_metrics = {}
        for name, frame in self.sources.items():
            validate_candidate_output(frame, k=200, source_name=name)
            validate_unseen(frame, self.users, self.store)
            self.source_metrics[name] = self.measure(frame)

    def measure(self, pairs):
        metrics = evaluate_candidate_metrics(pairs, self.truth, self.users)
        hits = pairs.select(KEYS).unique().join(self.truth, on=KEYS, how="semi")
        metrics["positive_hits"] = hits.height
        # Integer oracle hits avoid 1-ULP differences in parallel means changing
        # the policy tie-break when two unions have identical hit counts.
        oracle_hits = (
            hits.group_by("user_id")
            .len()
            .select(pl.col("len").clip(upper_bound=20).sum())
            .item()
        )
        metrics["candidate_oracle_p20_all_targets"] = oracle_hits / (
            20 * self.users.height
        )
        metrics["candidate_oracle_p20_labeled_users"] = oracle_hits / (
            20 * self.truth["user_id"].n_unique()
        )
        return metrics

    def evaluate(self, native):
        validate_candidate_output(native, k=200, source_name="sasrec")
        validate_unseen(native, self.users, self.store)
        native_metrics = self.measure(native)
        policies = {}
        als = self.sources["implicit_als"]
        for name, (als_cap, sasrec_cap) in POLICIES.items():
            pairs = pl.concat(
                [
                    self.other,
                    als.filter(pl.col("rank") <= als_cap).select(KEYS),
                    native.filter(pl.col("rank") <= sasrec_cap).select(KEYS),
                ]
            ).unique()
            policies[name] = {
                **self.measure(pairs),
                "sum_source_caps": 800,
                "als_cap": als_cap,
                "sasrec_cap": sasrec_cap,
            }
            policies[name]["delta_recall"] = (
                policies[name]["candidate_recall"]
                - self.baseline_metrics["candidate_recall"]
            )
        winner = max(
            POLICIES,
            key=lambda name: (
                policies[name]["candidate_recall"],
                policies[name]["candidate_oracle_p20_all_targets"],
                -list(POLICIES).index(name),
            ),
        )
        added = native.select(KEYS).join(self.baseline, on=KEYS, how="anti")
        new_hits = added.join(self.truth, on=KEYS, how="semi").height
        recs = standalone_top20(native, self.sources["global_popularity"], self.users)
        return {
            "objective": policies[winner]["delta_recall"],
            "objective_name": "best_equal_budget_union_delta_micro_recall",
            "selected_policy": winner,
            "policies": policies,
            "native_sasrec": native_metrics,
            "standalone_with_global_fallback": evaluate_precision_at_20(
                recs, self.truth, self.users
            ),
            "add_source_1000": {
                "new_positive_hits": new_hits,
                "delta_recall": new_hits / self.truth.height,
                "candidate_recall": (self.baseline_metrics["positive_hits"] + new_hits)
                / self.truth.height,
                "mean_added_unique_candidates": added.height / self.users.height,
                "eligible_for_selection": False,
            },
            "overlap_sasrec_vs_sources": {
                source: overlap(native, frame, self.truth, self.users)
                for source, frame in self.sources.items()
            },
            "target_users": self.users.height,
            "labeled_users": self.truth["user_id"].n_unique(),
            "positive_pairs": self.truth.height,
        }
