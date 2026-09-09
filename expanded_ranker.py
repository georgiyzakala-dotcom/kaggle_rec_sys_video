"""History-only features and bounded sampling for the five-source ranker."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import torch

from candidate_pipeline import (
    CandidateUnionConfig,
    attach_implicit_als_cross_scores,
    attach_item2item_cross_scores,
    attach_ranking_cross_scores,
    build_candidate_union,
    validate_union_features,
)
from features import (
    attach_als_factor_features,
    attach_history_features,
    attach_union_score_ranks,
    build_covisit_aggregate_features,
)
from full_history_profiles import add_profile_features
from item2item import Item2ItemDataLoader
from pipeline import SOURCE_ORDER, _item2item_seeds
from ranker_data import assign_binary_labels, deterministic_sampling_uniform
from sasrec_data import SASRecDataLoader
from validation import validate_feature_table

CAPS = {
    "global_popularity": 200,
    "recency_popularity": 200,
    "item2item": 200,
    "implicit_als": 600,
    "sasrec": 600,
}
EXTRA_FEATURES = (
    "generated_by_sasrec",
    "generator_score_sasrec",
    "generator_rank_sasrec",
    "generator_rank_norm_sasrec",
    "cross_score_sasrec",
    "cross_score_available_sasrec",
    "union_rank_cross_score_sasrec",
    "union_rank_norm_cross_score_sasrec",
    "sasrec_query_norm",
    "sasrec_item_norm",
    "sasrec_cosine",
)
PHASE_FOLDS = {
    "rolling_validation": ("rolling_1", "rolling_2"),
    "canonical_validation": ("rolling_1", "rolling_2", "rolling_3"),
    "final": ("rolling_1", "rolling_2", "rolling_3", "canonical"),
}
EVALUATION_FOLDS = {
    "rolling_validation": "rolling_3",
    "canonical_validation": "canonical",
}


def validate_phase_horizon(phase, fold_specs, prediction_start):
    """Training label windows must end before the evaluation/prediction horizon."""
    from datetime import timedelta

    limit = (
        datetime.fromisoformat(fold_specs[EVALUATION_FOLDS[phase]]["cutoff"])
        if phase in EVALUATION_FOLDS
        else prediction_start
    )
    cutoffs = [
        datetime.fromisoformat(fold_specs[f]["cutoff"]) for f in PHASE_FOLDS[phase]
    ]
    if cutoffs != sorted(set(cutoffs)) or any(
        x + timedelta(days=1) > limit for x in cutoffs
    ):
        raise ValueError("training labels cross the evaluation/prediction horizon")


def source_union(sources):
    return build_candidate_union(
        sources, CandidateUnionConfig.from_mapping(CAPS, total_cap=sum(CAPS.values()))
    )


def prepare_seeds(history, users, model, cutoff, seed):
    c = model.config
    loader = Item2ItemDataLoader(
        reference_time=cutoff,
        max_history_items=c.history_cap,
        max_seed_items=c.seed_k,
        seed=seed,
    )
    loader.load_predict_data(history=history, target_users=users).prepare_predict_data()
    return _item2item_seeds(loader)


@torch.inference_mode()
def attach_sasrec_scores(union, model, store, *, seed=42, batch_size=32768):
    """Score all union pairs, including pairs absent from SASRec's native top600."""
    users = union.select("user_id").unique().sort("user_id")["user_id"].to_numpy()
    loader = SASRecDataLoader(store, max_length=model.config.max_length, seed=seed)
    loader.load_predict_data(user_ids=users).prepare_predict_data()
    model._check_loader(loader)
    model.encoder.eval()
    queries = torch.zeros((len(users), model.config.embedding_dim), device=model.device)
    active_users = np.zeros(len(users), dtype=bool)
    for batch in loader.iter_predict_batches(batch_size=64):
        active = (batch.inputs != 0).any(1)
        positions = np.searchsorted(users, batch.user_ids)
        active_users[positions] = active.numpy()
        if active.any():
            query = model.encoder.encode_users(
                batch.inputs[active].to(model.device, dtype=torch.int64)
            ).float()
            queries[torch.as_tensor(positions[active.numpy()], device=model.device)] = (
                query
            )
    u = np.searchsorted(users, union["user_id"].to_numpy())
    items = union["item_id"].to_numpy()
    i = np.searchsorted(model.item_ids, items)
    known = i < len(model.item_ids)
    known[known] &= model.item_ids[i[known]] == items[known]
    available = active_users[u] & known
    output = np.zeros((union.height, 4), dtype=np.float32)
    valid = np.flatnonzero(available)
    for start in range(0, len(valid), batch_size):
        rows = valid[start : start + batch_size]
        query = queries[torch.as_tensor(u[rows], device=model.device)]
        embedding = model.encoder.item_embedding(
            torch.as_tensor(i[rows] + 1, device=model.device)
        )
        scores = (query * embedding).sum(-1)
        qnorm = query.norm(dim=-1)
        inorm = embedding.norm(dim=-1)
        cosine = scores / (qnorm * inorm).clamp_min(1e-12)
        output[rows] = (
            torch.stack((scores, qnorm, inorm, cosine.clamp(-1, 1)), 1).cpu().numpy()
        )
    if not np.isfinite(output).all():
        raise ValueError("non-finite SASRec cross features")
    return union.with_columns(
        pl.Series("cross_score_sasrec", output[:, 0], dtype=pl.Float64),
        pl.Series("cross_score_available_sasrec", available),
        pl.Series("sasrec_query_norm", output[:, 1]),
        pl.Series("sasrec_item_norm", output[:, 2]),
        pl.Series("sasrec_cosine", output[:, 3]),
    )


def cross_scored_union(sources, *, history, users, models, sasrec, store, cutoff, seed):
    union = source_union(sources)
    for name in SOURCE_ORDER[:2]:
        union = attach_ranking_cross_scores(
            union, source=name, item_ranking=models[name].item_ranking
        )
    seeds = prepare_seeds(history, users, models["item2item"], cutoff, seed)
    union = attach_item2item_cross_scores(
        union,
        seeds=seeds,
        neighbor_table=models["item2item"].neighbor_table,
        config=models["item2item"].config,
        reference_time=cutoff,
    )
    union = attach_implicit_als_cross_scores(union, model=models["implicit_als"])
    union = attach_sasrec_scores(union, sasrec, store, seed=seed)
    validate_union_features(
        union,
        config=CandidateUnionConfig.from_mapping(CAPS, total_cap=1800),
        require_cross_scores=tuple(CAPS),
    )
    # These ranks depend on the complete user query. Never recompute them after sampling.
    return attach_union_score_ranks(
        union, sources=("item2item", "implicit_als", "sasrec")
    ), seeds


def sample_union(union, truth, *, fold, seed, negative_probability):
    if not 0 < negative_probability <= 1:
        raise ValueError("invalid inclusion probability")
    labeled = assign_binary_labels(union, truth)
    uniform = deterministic_sampling_uniform(
        labeled, seed=seed, fold=f"expanded_ranker:{fold}"
    )
    selected = (
        labeled.with_columns(pl.Series("__sample", uniform))
        .filter((pl.col("label") == 1) | (pl.col("__sample") < negative_probability))
        .drop("__sample")
    )
    if selected["label"].sum() != labeled["label"].sum():
        raise ValueError("sampling dropped positives")
    return selected.with_columns(
        pl.when(pl.col("label") == 1)
        .then(1.0)
        .otherwise(1.0 / negative_probability)
        .cast(pl.Float32)
        .alias("sample_weight")
    )


def materialize_features(union, *, seeds, lookups, models, profiles, cutoff, features):
    """Expensive pair features only for retained rows; query ranks stay frozen."""
    result = attach_history_features(
        union,
        user_features=lookups["user_features"],
        item_features=lookups["item_features"],
    )
    result = build_covisit_aggregate_features(
        result,
        seeds=seeds,
        neighbor_table=models["item2item"].neighbor_table,
        config=models["item2item"].config,
        cutoff=cutoff,
    )
    if result.filter(
        (pl.col("covisit_available") != pl.col("cross_score_available_item2item"))
        | (
            pl.col("covisit_available")
            & (
                (
                    pl.col("covisit_contribution_max") - pl.col("cross_score_item2item")
                ).abs()
                > 1e-9
            )
        )
    ).height:
        raise ValueError("co-visitation features disagree with cross scores")
    result = attach_als_factor_features(
        result,
        user_norms=lookups["als_user_norms"],
        item_norms=lookups["als_item_norms"],
    )
    result = add_profile_features(result, profiles)
    metadata = [c for c in ("label", "sample_weight") if c in result.columns]
    result = result.select(
        "user_id", "item_id", *metadata, *(pl.col(f).cast(pl.Float32) for f in features)
    )
    validate_feature_table(result.select("user_id", "item_id", *features))
    return result
