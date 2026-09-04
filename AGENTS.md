# AGENTS.md

This file applies to the entire repository. Its purpose is to keep work on the
Kaggle recommender system reproducible across independent Codex windows.
Communicate with the user in Russian; write code, field names, and technical
identifiers in English.

## Objective

Build a ranked list of 20 videos for every user in
`data/target_user_ids.parquet`, predicting the videos with which the user is
most likely to have a relevant interaction during the following day.

A `(user_id, item_id)` pair is relevant if at least one of the following events
occurs during the target period:

- `event_type == "like"`;
- `event_type == "favorite"`;
- `event_type == "watch_time"` and `watch_time > 60` (strictly greater than
  60).

Multiple relevant events for the same pair produce one relevant item. Exclude
pairs already observed in the historical period from both local ground truth
and recommendations. Recommend only items present in the historical period.
The final submission must contain every target user and at most 20 unique items
per user; the working standard is exactly 20.

The primary metric is macro Precision@20:
`mean_u(|recommendations_u ∩ relevant_u| / 20)`. Because the competition
implementation may handle users with empty ground truth differently, always
name and store both diagnostic variants separately:

- `precision_at_20_all_targets`: evaluated over all target users; empty ground
  truth contributes 0;
- `precision_at_20_labeled_users`: evaluated only over users with non-empty
  local ground truth.

Never compare scores that use different denominators as if they were the same
metric.

## Data and Known State

Raw data is immutable:

- `data/train.parquet` — interaction events;
- `data/target_user_ids.parquet` — users for whom predictions are required.

`research/prepare_data.ipynb` is an executed exploratory preparation notebook.
It creates the following derived snapshots under `data/`:

- `data/prepared_train_data.parquet` — 40,409,221 daily user-item rows from
  calendar dates before `2024-12-03` (408,025 users; 1,860,887 items);
- `data/prepared_test_data.parquet` — 1,228,675 daily user-item rows from
  calendar date `2024-12-03`, covering only `00:00:00` through `08:56:28`
  (116,916 users; 434,910 items);
- `data/item_features_data.parquet` — 13 columns for 1,860,887 items,
  calculated from `data/prepared_train_data.parquet`;
- `data/user_features_data.parquet` — 13 columns for 408,025 users, calculated
  from `data/prepared_train_data.parquet`.

The prepared train/test schema is `user_id UInt64`, `item_id Int32`,
`date Date`, `dt Datetime[us]`, `views UInt32`, `watch_time Int64`,
`is_like Int32`, `is_favorite Int32`, and `is_positive Int32`. Raw events are
grouped by `(user_id, item_id, calendar date)`: `dt` is the first timestamp,
`views` is the event count, `watch_time` is the maximum, and the three binary
flags use the daily maximum. `is_positive` implements the relevance rule
above.

The feature snapshots first collapse prepared rows to one row per user-item,
using summed `views` and maxima for `watch_time` and binary flags. They then
aggregate by item or user into `views`, `avg_views`, `avg_watch_time`,
`p3_watch_time`, `p8_watch_time`, `median_watch_time`, `likes`, `favorites`,
`positives`, `like_rate`, `favorite_rate`, and `positive_rate`. The rates and
watch-time statistics are therefore over unique counterpart IDs, not raw
events or daily rows.

These four files are reproducible notebook outputs, not additional raw data.
Treat the existing snapshots as read-only and do not silently overwrite them.
New or revised derived data should normally go under `artifacts/<run_id>/` (or
another explicitly configured non-`data/` path), with its generating command
and split recorded.

Snapshot verified on 2026-08-31 (derive all values from the data in code; do
not hardcode them):

- train: 44,413,436 rows, 408,025 users, and 1,884,722 items;
- time range: `2024-11-27 08:56:29` through `2024-12-03 08:56:28`;
- schema: `user_id UInt64`, `item_id Int32`, `event_type Categorical`,
  `watch_time Int64`, and `date Datetime[us]`; there are no null values;
- event counts: 41,400,136 `watch_time`, 2,018,926 `favorite`, and 994,374
  `like`;
- target: 200,152 unique `user_id` values, all of which occur in train.

The time range touches seven calendar dates but contains only six complete
24-hour periods. Therefore, do not treat the final calendar date as a complete
holdout day.

In particular, the calendar split in `prepare_data.ipynb` is not the canonical
local-validation split. Its prepared train contains events from the canonical
validation window on `2024-12-02`, while its prepared test contains only the
last 8 hours, 56 minutes, and 29 seconds of the raw timeline. Daily aggregation
also makes an exact split at `2024-12-02 08:56:28` impossible to reconstruct
when a user-item has events on both sides of that timestamp. Consequently:

- do not use `prepared_train_data.parquet`, `prepared_test_data.parquet`, or
  either feature snapshot for the canonical holdout;
- build every canonical or rolling split from raw `train.parquet` before any
  daily aggregation, then aggregate the history and validation sides
  independently;
- if the prepared calendar split is used for exploration, label its metrics as
  non-canonical and never compare them directly with canonical 24-hour scores;
- for final fitting on the full history, create a new full-history daily
  aggregate from all of `train.parquet` and regenerate features from it; the
  current feature snapshots intentionally stop before calendar date
  `2024-12-03`.

### Shared Daily Interaction Contract

All models in a given temporal fold must use the same immutable daily
interaction snapshots. Raw events are read by the data-preparation pipeline,
not separately by every model. The required order is:

1. split raw events at the exact timestamp cutoff;
2. aggregate the raw history and validation sides independently;
3. materialize versioned fold artifacts under `artifacts/<run_id>/`;
4. let model-specific loaders read the prepared history artifact and produce
   their batches.

The shared aggregation has exactly one row per
`(user_id, item_id, calendar date)` within each side of a fold. Its schema and
semantics match the existing prepared snapshots:

- `user_id UInt64`, `item_id Int32`, `date Date`, `dt Datetime[us]`;
- `dt` is the minimum raw timestamp in the group;
- `views UInt32` is the number of raw event rows in the group (it is an event
  row count, not a guaranteed count of distinct physical plays);
- `watch_time Int64` is the maximum;
- `is_like Int32` and `is_favorite Int32` are daily maxima of the corresponding
  event indicators;
- `is_positive Int32` is one when `watch_time > 60`, `is_like == 1`, or
  `is_favorite == 1`.

The split-before-aggregation order is mandatory. A user-item pair may produce
one daily row on each side when the timestamp cutoff falls inside a calendar
day; those rows must never be merged across the cutoff. Fold-specific user/item
features must be derived only from that fold's prepared history. All candidate
models and rankers in the fold reuse the same history snapshot. Rolling folds
receive separate snapshots, and final fitting receives a separate aggregate of
the complete raw history.

## Canonical Local Validation Protocol

The primary split must simulate the hidden next-day period:

1. Set `cutoff = train.date.max() - 1 day`.
2. Split raw events into history `date < cutoff` and validation
   `date >= cutoff`.
3. Apply the shared daily interaction aggregation independently to both raw
   sides and validate uniqueness of `(user_id, item_id, date)` in each result.
4. Materialize and reuse the prepared fold snapshots; fit all features,
   statistics, candidates, and ranking logic using prepared history only.
5. Filter prepared validation rows using `is_positive`, then deduplicate by
   `(user_id, item_id)` over the complete 24-hour target period.
6. Remove ground-truth pairs already observed in prepared history.
7. Separately remove validation items absent from prepared history, and always
   log their count and share.
8. Use the intersection with `target_user_ids` as the primary user universe,
   and calculate both Precision@20 variants defined above.

For the current data, this split gives a cutoff of `2024-12-02 08:56:28`, with
40,213,747 raw history events and 4,199,689 raw validation events. After daily
aggregation, applying the relevance filter, and removing old pairs and cold
items, 1,740,770 pairs remain for 212,238 users. Target users account for
1,284,148 of those pairs across 147,036 users. Use these values as sanity
checks, not as algorithm constants.

When computationally feasible, also evaluate on several rolling 24-hour folds.
The primary report must still include the canonical final holdout so that runs
remain comparable. Do not use the validation period to build or tune
popularity, co-visitation, embeddings, negative samples, thresholds, or
fallback lists.

## Modeling Rules

- Use the immutable prepared history snapshot for the current fold as the
  common interaction input to every model. A model-specific loader may build
  additional mappings or features, but it must not redefine the daily
  aggregation or read validation events into fit data.
- Start by building and recording a strong deterministic baseline: global
  popularity/recency with per-user exclusion of previously seen items.
- Improve candidate generation and ranking independently. Useful candidate
  sources include recent/trending popularity, co-visitation or item-to-item
  statistics based on strong events, users with similar histories, and blends
  of these sources.
- For every candidate source, measure candidate recall, coverage, mean
  candidate count, and contribution to final hits in addition to Precision@20.
- Tune weights for `like`, `favorite`, long watches, time decay, and blends only
  on local temporal folds. Do not present heuristic weights as validated
  improvements.
- The current dataset has no item or user metadata. Do not assume that titles,
  categories, video duration, or other features exist unless such data is
  actually added.
- Always maintain a non-personalized fallback that can fill each list to 20
  after removing seen items and duplicates.
- Compare every new approach with the current best run using the same split and
  seed. Also report runtime and peak memory usage when they are material.

## Engineering Conventions

- Use the existing `.venv` and Python 3.12. For large tables, prefer Polars
  `scan_parquet`, lazy operations, predicate/projection pushdown, and
  aggregations. Do not load the full train dataset into pandas.
- Preserve the source ID dtypes. Never pass `UInt64 user_id` through a floating-
  point representation.
- Give every random operation an explicit seed; use `42` by default.
- Exploration may live in `research/`, but move reusable logic into importable
  modules or CLI scripts instead of leaving it only in notebook cells.
- As the project grows, use `src/` for library code, `scripts/` for CLIs,
  `configs/` for configurations, `tests/` for fast tests,
  `artifacts/<run_id>/` for metrics and models, and `research/` for notebooks.
- Paths must be relative to the repository root or supplied through CLI
  arguments. Do not hardcode machine-specific absolute paths.
- Never modify raw files under `data/`. The four existing prepared snapshots
  documented above are a read-only exception; do not add further derived
  artifacts there by default.
- Do not add large predictions, caches, or models to version control. If Git is
  configured, inspect the existing `.gitignore` and uncommitted changes first.
- Do not install dependencies unless necessary. If a dependency is added,
  record it in the project's chosen dependency file and explain its purpose.
- Do not submit to Kaggle or perform any other external action unless the user
  explicitly asks for it.

## Long-Running Experiments

- Codex must not start a full experiment, an overnight job, a large ablation,
  or training of many model/fold combinations inside its own tool session.
  Codex may run unit tests and explicitly limited smoke runs needed to validate
  the implementation. A long run may be started by Codex only when the user
  explicitly requests that exact run in the current conversation.
- For long work, Codex prepares production code, config files, tests, a CLI
  runner, and a small shell launcher that the user can start directly in a
  terminal. The handoff must include the exact launch command, output and log
  paths, expected phases, and commands for monitoring and stopping the run.
- The terminal runner must provide useful nested progress for the overall run,
  stage, config/fold, and model iterations or batches where applicable. It must
  show elapsed time and an ETA whenever the unit of work is measurable.
- Every long runner must also write a compact plain-text log without ANSI
  progress control sequences. Each important line must include a timestamp and
  enough structured context to identify `stage`, `config`, `fold`, and
  `operation`. Log phase/config/fold start, finish, failure, duration, current
  metric, and best metric/config. Avoid per-row or otherwise unbounded logging;
  use rotation or another explicit size limit.
- Long multi-model selection must persist atomic checkpoints at safe
  boundaries, normally after each completed `(stage, config, fold)`, and must
  atomically save the current best portable model. Where practical, the runner
  must resume completed work after interruption instead of repeating it.
  Checkpoints must never expose partial files as completed artifacts.
- Launchers must set explicit CPU/thread limits, prevent accidental concurrent
  duplicate runs, refuse to overwrite completed artifacts, and publish final
  artifacts atomically. A failed or interrupted run must leave a readable log
  and enough checkpoint metadata to determine exactly what completed.
- Before handing a long runner to the user, Codex must validate `--help`, shell
  syntax, unit tests, and at least one limited end-to-end smoke run. The smoke
  must use a distinct artifact/run ID and must not be recorded as a full model
  experiment.

## State Handoff Between Codex Windows

Before starting work:

1. Read this file in full.
2. Inspect the file tree and all existing changes.
3. Read `PROJECT_STATE.md` and the experiment log if they exist.
4. Identify the current best run and the exact command that reproduces it.
5. Do not repeat an existing experiment without a clear reason.

During the first substantive pipeline task, create and subsequently maintain:

- `PROJECT_STATE.md` — a concise, current handoff describing what works, the
  current best result, the exact run command, known issues, and the next 3–5
  steps;
- `experiments/results.csv` — an append-only log containing at least `run_id`,
  `timestamp`, `split`, `seed`, `candidate_config`, `ranker_config`,
  `p20_all_targets`, `p20_labeled_users`, `candidate_recall`, `coverage`,
  `runtime`, `artifact_path`, and `notes`.

For every meaningful run, store at least `config.json` and `metrics.json` under
`artifacts/<run_id>/`. Never overwrite another run's results. If an experiment
fails, record the cause when it could help a later window.

After a task, update `PROJECT_STATE.md` whenever the best score, interfaces,
pipeline structure, known limitations, or next step changes. A handoff must
contain concrete facts and commands, not a general narrative. During parallel
work, create separate modules and artifacts, and reread shared files immediately
before editing them to avoid overwriting another worker's changes.

## Checks Required Before Reporting Completion

- Add a small synthetic test covering relevance, seen-pair filtering,
  deduplication, and Precision@20.
- Run a smoke test on a limited sample, followed by a full validation run when
  it fits the available resources.
- Verify deterministic output when rerunning with the same seed.
- Validate recommendation output: exactly one row per target user; no missing,
  extra, or duplicate users; exactly 20 unique items per user; every item is
  known from train; no train-seen pair is present; and there are no null values.
- Before writing the final CSV, locate the official `sample_submission` or an
  exact specification for serializing `item_ids`. The current repository has no
  sample file, so do not guess whether IDs must be space-separated,
  comma-separated, or formatted as a list.
- In the final report, list changed files, verification commands, both local
  metric variants, and all remaining assumptions or risks.

## Immediate Priorities

1. Implement the canonical timestamp-level temporal split directly from raw
   data and add metric tests; do not reuse the prepared calendar split.
2. Produce a popularity-plus-recency baseline and valid top-20 output for every
   user.
3. Add co-visitation candidates and measure candidate recall by source.
4. Build a simple ranker or blend and run ablations on the same folds.
5. After validating a local improvement, fit on the full train dataset and
   create a strictly validated submission file.
