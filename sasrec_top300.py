"""Deeper retrieval and exact metrics using the existing frozen fold models."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import torch

from interfaces import CANDIDATE_SCHEMA
from sasrec_evaluation import (
    counts,
    evaluate_shard,
    fraction_summary,
    summarize,
    summarize_counts,
)
from sasrec_model import stable_topk
from sasrec_selection import KEYS, validate_unseen
from validation import validate_candidate_output

EXTENDED_POLICIES = {
    "als300_only_900": (300, 0),
    "als300_sasrec200_1100": (300, 200),
    "als200_sasrec300_1100": (200, 300),
    "als300_sasrec300_1200": (300, 300),
}
PRIMARY = "als300_sasrec300_1200"


@torch.inference_mode()
def predict_vectorized_seen(
    model,
    loader,
    *,
    k=300,
    batch_size=64,
    item_chunk_size=32768,
    callback=None,
    check_stop=None,
):
    """Same scores/top-k as SASRec.predict, one seen-mask assignment per chunk.

    Sort sparse seen coordinates on CPU once per batch. This avoids a separate
    CPU->GPU allocation for every (user, catalog chunk), without changing query
    encoding, score arithmetic, eligible items or deterministic tie resolution.
    """
    model._check_loader(loader)
    model.encoder.eval()
    outputs = []
    for batch_index, batch in enumerate(
        loader.iter_predict_batches(batch_size=batch_size)
    ):
        if check_stop:
            check_stop()
        active = torch.where((batch.inputs != 0).any(1))[0]
        if len(active):
            query = model.encoder.encode_users(
                batch.inputs[active].to(model.device, dtype=torch.int64)
            ).float()
            users = batch.user_ids[active.numpy()]
            seen = [batch.seen_items[i] for i in active.tolist()]
            flat = np.concatenate(seen).astype(np.int64, copy=False)
            rows = np.repeat(
                np.arange(len(active), dtype=np.int64), [len(s) for s in seen]
            )
            order = np.argsort(flat, kind="stable")
            flat = flat[order]
            seen_ids = torch.as_tensor(flat, device=model.device)
            seen_rows = torch.as_tensor(rows[order], device=model.device)
            best_values = torch.empty((len(active), 0), device=model.device)
            best_ids = torch.empty(
                (len(active), 0), device=model.device, dtype=torch.int64
            )
            for start in range(1, len(model.item_ids) + 1, item_chunk_size):
                if check_stop:
                    check_stop()
                end = min(start + item_chunk_size, len(model.item_ids) + 1)
                scores = (
                    query @ model.encoder.item_embedding.weight[start:end].float().T
                )
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("non-finite SASRec retrieval scores")
                left, right = np.searchsorted(flat, [start, end])
                scores[seen_rows[left:right], seen_ids[left:right] - start] = -torch.inf
                ids = torch.arange(start, end, device=model.device).expand(
                    len(active), -1
                )
                values, indices = stable_topk(scores, ids, k, ids_sorted=True)
                best_values, best_ids = stable_topk(
                    torch.cat((best_values, values), 1),
                    torch.cat((best_ids, indices), 1),
                    k,
                )
            values, indices = best_values.cpu().numpy(), best_ids.cpu().numpy()
            for row, user in enumerate(users):
                for rank, (value, index) in enumerate(
                    zip(values[row], indices[row], strict=True), 1
                ):
                    if math.isfinite(float(value)):
                        outputs.append(
                            (
                                int(user),
                                int(model.item_ids[index - 1]),
                                float(value),
                                rank,
                                "sasrec",
                            )
                        )
        if callback:
            callback(batch_index + 1)
    result = pl.DataFrame(outputs, schema=CANDIDATE_SCHEMA, orient="row")
    validate_candidate_output(result, k=k, source_name="sasrec")
    return result


def require_same_prefix(deeper, original, k):
    columns = [*KEYS, "rank"]
    if (
        not deeper.filter(pl.col("rank") <= k)
        .select(columns)
        .sort("user_id", "rank")
        .equals(original.select(columns).sort("user_id", "rank"))
    ):
        raise ValueError(
            "deeper SASRec top200 differs from the saved frozen predictions"
        )


def extend_shard(users, truth, sources, sas200, als400, sas300, store, old_stats=None):
    users = users.sort("user_id")
    validate_candidate_output(sas300, k=300, source_name="sasrec")
    validate_unseen(sas300, users, store)
    require_same_prefix(sas300, sas200, 200)
    stats, _ = evaluate_shard(users, truth, sources, sas200, als400, store)
    if old_stats is not None and not stats.equals(old_stats.select(stats.columns)):
        raise ValueError(
            "independent recomputation differs from saved per-user metrics"
        )
    other = pl.concat(
        [
            sources[s].select(KEYS)
            for s in ("global_popularity", "recency_popularity", "item2item")
        ]
    ).unique()
    views = {
        "sasrec_at_300": sas300.select(KEYS),
        "implicit_als_at_300": als400.filter(pl.col("rank") <= 300).select(KEYS),
    }
    for name, (a_cap, s_cap) in EXTENDED_POLICIES.items():
        views[name] = pl.concat(
            [
                other,
                als400.filter(pl.col("rank") <= a_cap).select(KEYS),
                sas300.filter(pl.col("rank") <= s_cap).select(KEYS),
            ]
        ).unique()
    for name, pairs in views.items():
        stats = stats.with_columns(
            counts(users, pairs, f"m__{name}__count"),
            counts(users, pairs.join(truth, on=KEYS, how="semi"), f"m__{name}__hits"),
        )
    common = views["sasrec_at_300"].join(
        views["implicit_als_at_300"], on=KEYS, how="semi"
    )
    stats = stats.with_columns(
        counts(users, common, "o300_common"),
        counts(users, common.join(truth, on=KEYS, how="semi"), "o300_shared_hits"),
    )
    return stats


def summarize_extension(stats, frozen_policy):
    base = summarize(stats, frozen_policy)
    sources300 = {
        s: summarize_counts(stats, f"{s}_at_300") for s in ("sasrec", "implicit_als")
    }
    baseline = base["policies"]["baseline_800"]
    old_union = base["policies"]["add_source_1000"]
    policies = dict(base["policies"])
    for name, (a_cap, s_cap) in EXTENDED_POLICIES.items():
        values = summarize_counts(stats, name)
        policies[name] = {
            **values,
            "als_cap": a_cap,
            "sasrec_cap": s_cap,
            "sum_source_caps": 600 + a_cap + s_cap,
            "delta_recall_vs_baseline800": values["candidate_recall"]
            - baseline["candidate_recall"],
            "delta_recall_vs_both200": values["candidate_recall"]
            - old_union["candidate_recall"],
            "delta_positive_hits_vs_both200": values["positive_hits"]
            - old_union["positive_hits"],
            "delta_mean_candidate_count_vs_both200": values["mean_candidate_count"]
            - old_union["mean_candidate_count"],
        }
    a = stats["m__sasrec_at_300__count"].cast(pl.Int64)
    b = stats["m__implicit_als_at_300__count"].cast(pl.Int64)
    common = stats["o300_common"].cast(pl.Int64)
    shared = int(stats["o300_shared_hits"].sum())
    a_hits, b_hits = (
        sources300["sasrec"]["positive_hits"],
        sources300["implicit_als"]["positive_hits"],
    )
    union = int((a + b - common).sum())
    return {
        "base_metrics_recomputed": base,
        "target_users": stats.height,
        "labeled_users": base["labeled_users"],
        "positive_pairs": base["positive_pairs"],
        "primary_policy": PRIMARY,
        "policies": policies,
        "sources_at_300": sources300,
        "overlap_sasrec300_als300": {
            "intersection_pairs": int(common.sum()),
            "micro_jaccard": int(common.sum()) / union if union else None,
            "left_positive_hits": a_hits,
            "right_positive_hits": b_hits,
            "shared_positive_hits": shared,
            "left_only_positive_hits": a_hits - shared,
            "right_only_positive_hits": b_hits - shared,
            "shared_share_of_left_hits": shared / a_hits if a_hits else None,
            "shared_share_of_right_hits": shared / b_hits if b_hits else None,
            "per_user": {
                "jaccard": fraction_summary(common, a + b - common),
                "share_sasrec_in_als": fraction_summary(common, a),
                "share_als_in_sasrec": fraction_summary(common, b),
            },
        },
        "candidate_recall": policies[PRIMARY]["candidate_recall"],
        "coverage": policies[PRIMARY]["coverage"],
        "precision_at_20_all_targets": None,
        "precision_at_20_labeled_users": None,
        "precision_semantics": "primary result is an unranked union; standalone top20 metrics are unchanged in base_metrics_recomputed",
        "model_fits": 0,
        "als_inference_reused": True,
        "standalone_top20_unchanged": True,
        "budgets_warning": "both300 costs 1200 source slots; both200 costs 1000; ALS600 control was not evaluated",
    }
