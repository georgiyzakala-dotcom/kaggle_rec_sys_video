# PROJECT_STATE

Обновлено: 2026-09-03. Задача 10 завершена full LTR selection и ровно одной
canonical оценкой frozen winner. Rolling winner `s40_l2_leaf_reg_10` использует
`QuerySoftMax:beta=2`; mean P@20 labeled равен `0.0047389716`, а canonical
P@20 all/labeled — `0.0028803110 / 0.0039208085` (`11,530` hits). Это
существенно хуже pointwise Task 08/09 и RRF, поэтому group-aware objective не
дал прироста; текущий лучший canonical run остаётся
`task08_catboost_pointwise_v1` с `0.0047883608 / 0.0065181316`. Full
`YetiRankPairwise` признан resource-infeasible на 16 GB VRAM по изолированному
one-tree full-pool probe. Task 10 добавлен в `experiments/results.csv`, artifact
и grouped pools независимо проверены.

Задача 06 завершена: пользователь выполнил full offline materialization и RRF
ablation, после чего checksums, checkpoints, portable restore, notebook,
candidate/final metrics и semantic invariants были независимо проверены.
RRF не превзошёл standalone ALS, но offline union дал достаточный oracle
headroom CatBoost ranker. Task 09 использует неизменные Task 07 candidates и
201 features; candidate/feature materialization не повторяется.

## Что работает

- `interfaces.py` и `validation.py` задают проверенные Model/DataLoader
  lifecycle и schemas для candidates, ranker features/output и final
  recommendations. Model получает только model-specific batches и не читает
  raw data.
- `data_utils.py` реализует lazy raw/target scans, exact timestamp split,
  reusable temporal folds, независимую daily aggregation сторон и atomic
  immutable materialization. Canonical contract:
  `dt=min(raw timestamp)`, `views=count(raw rows)`, `watch_time=max`, maxima
  like/favorite и strict `watch_time > 60` для positive.
- `metrics.py` реализует отдельно `precision_at_20_all_targets` и
  `precision_at_20_labeled_users` с фиксированным denominator 20, а также
  candidate recall, user hit rate, обе oracle P@20 и count/coverage metrics.
  `utils.prec_k` оставлен совместимым wrapper с тем же фиксированным
  denominator.
- `scripts/prepare_canonical_data.py` не перезаписывает run, поддерживает явно
  маркированный limited smoke и optional half-open validation end для будущих
  rolling folds.
- `research/01_validation_protocol.ipynb` читает готовый artifact и показывает
  split/daily/ground-truth diagnostics без дублирования pipeline logic.
- `popularity.py` реализует history-only `PopularityDataLoader`, четыре
  определения popularity, `GlobalPopularityModel`, batched unseen top-k и
  reusable global fallback. `validation.py` дополнительно проверяет точное
  совпадение target universe, known items и отсутствие seen pairs.
- `scripts/run_global_popularity.py` фиксирует score только по трём ранним
  rolling folds, затем оценивает ровно одну конфигурацию на canonical holdout;
  artifact публикуется атомарно и не перезаписывается.
- `research/02_global_popularity.ipynb` читает сохранённые selection/canonical
  результаты. В `experiments/results.csv` записан первый модельный run.
- `popularity.py` дополнительно реализует portable
  `RecencyPopularityConfig`, history-only `RecencyPopularityDataLoader` и
  `RecencyPopularityModel(CandidateModel)`. Поддерживаются window, exponential
  decay, smoothed trending и weighted-window blend scores для `raw_views` и
  `positive_daily_rows`; модель выдаёт чистый source `recency_popularity` без
  подмешивания fallback и восстанавливается из валидированного fitted ranking.
- `scripts/run_recency_popularity.py` вычисляет temporal item statistics один
  раз на fold, выбирает одну из 26 конфигураций только по трём rolling folds,
  затем ровно один раз оценивает winner на canonical. Task02 global popularity
  остаётся отдельным downstream fallback; runner сохраняет переносимый
  `model_config.json`, fitted ranking и validated recommendations атомарно.
- `research/03_recency_popularity.ipynb` читает полный ablation, per-fold и
  canonical results и демонстрирует восстановление candidate model из
  portable config/ranking. В `experiments/results.csv` записан task03 run.
- `item2item.py` реализует валидируемый portable `Item2ItemConfig`, history-only
  collapse/cap loader, sparse SciPy CSR pair aggregation, raw/cosine/Jaccard
  normalization, bounded neighbor table, batched seed inference с max
  aggregation и restore через `from_fitted_neighbors`.
- `scripts/run_item2item.py` последовательно выбирает семь блоков параметров
  только на трёх rolling folds, не открывает canonical до фиксации winner,
  переиспользует временные neighbor tables для inference-only stages и
  атомарно сохраняет только canonical model/recommendations. Отдельно считает
  source hit overlap, exclusive hits и union candidate metrics с task02/task03.
- `research/04_item2item_covisitation.ipynb` импортирует production
  implementation и анализирует готовый artifact без повторного fit.
- `implicit_model.py` реализует CPU-only float32 CG ALS на `implicit==0.7.3`:
  stable UInt64/Int32 mappings, fold-history confidence CSR, official batched
  `recommend` с seen filtering и strict tie resolution, а также portable
  `save()`/`from_artifact()` без повторного fit. Зависимости зафиксированы в
  `requirements.txt`.
- `scripts/run_implicit_als.py` реализует пять rolling-only stages и canonical
  isolation. CLI показывает три уровня `tqdm` progress (общие фазы,
  config/fold, ALS iterations) и пишет компактный rotating log с
  `stage/config/fold/operation`, durations и итоговыми status. Лог ограничен
  1 MB плюс два backup-файла и не содержит ANSI progress output.
  `scripts/run_task05_overnight.sh` задаёт CPU thread limits, блокирует
  параллельный повторный запуск и запускает полный experiment с отдельным
  логом. После каждой улучшившей rolling-конфигурации атомарный callback
  сохраняет portable model последнего rolling fold, selection/fold metrics и
  SHA-256 в `artifacts/.task05_implicit_als_v1.best-model/`; старую версию он
  удаляет только после переключения `best_model.json`. Полное возобновление
  ablation state пока не реализовано. Два real-data smoke runs
  (64 targets, 2,000 context users) завершились: winner
  `stage5_iterations_iterations_15`, rolling union
  oracle gain `0.005208333333333336`, canonical smoke P@20 all targets
  `0.00234375`, ALS-exclusive hits `4`; все model/output SHA-256 и factors
  совпали bit-exact между runs и после restore.
- Full `task05_implicit_als_v1` завершён за `30496.54s`. Rolling winner —
  event-strength confidence без decay, 128 factors, regularization 0.1 и 15
  iterations. Portable canonical model, mappings, recommendations, resolved
  config и полный selection/canonical metrics сохранены атомарно; production
  restore и независимая semantic validation прошли.
- `research/05_implicit_als.ipynb` читает готовый artifact через
  `ImplicitALSModel.from_artifact()`, показывает последовательность rolling
  winners, canonical сравнение и complementary union metrics без повторного
  fit.
- `candidate_pipeline.py` материализует bounded source-aware union с
  раздельными `generated_by_*`, generator scores/ranks/normalized ranks и
  `source_count`. Для каждой union-пары добавляются независимые history-only
  cross-scores global popularity, recency popularity, sparse item2item и ALS
  factor dot product вместе с availability flags. Union обрабатывается по
  непересекающимся user shards; semantic known/unseen check выполняется одним
  lazy проходом на fold.
- `scripts/prepare_candidate_datasets.py` один раз fit/restore-ит четыре
  frozen candidate sources на каждом temporal fold, сохраняет narrow source
  candidates с SHA-256 и затем публикует шардированные offline features.
  Canonical task02–task05 models и rolling-3 ALS восстанавливаются из готовых
  artifacts; отсутствующие rolling models обучаются только в этом отдельном
  preparation job. Повторные ensemble/ranker runs их не вызывают.
- `ensemble.py` реализует переносимый deterministic RRF baseline. Отдельный
  `scripts/run_candidate_ensemble.py` читает только checksum-validated offline
  Parquet, последовательно выбирает caps, RRF constant и weights на трёх
  rolling folds, а затем ровно один раз оценивает winner на canonical.
  Runtime исключён из model-selection tie-break; равные метрики разрешаются
  стабильным config order.
- `experiment_utils.py` предоставляет nested `tqdm`, компактный rotating log
  без ANSI, atomic operation checkpoints, atomic best-config pointer и atomic
  directory publish. `scripts/run_task06_prepare_overnight.sh` и
  `scripts/run_task06_rrf.sh` задают thread limits, `flock`, PID/log paths и
  отказываются перезаписывать готовые artifacts.
- `research/06_candidate_ensemble.ipynb` является read-only анализом готовых
  task06 artifacts и не запускает candidate models. Все пять code cells
  успешно выполнены на full artifacts.

## Canonical artifact

`artifacts/task01_canonical_data_v1/` создан только из `data/train.parquet` и
`data/target_user_ids.parquet`; calendar-split snapshots не использовались.

- Cutoff: `2024-12-02 08:56:28`.
- Raw: `40,213,747` history и `4,199,689` validation events.
- Daily: `37,694,179` history и `3,963,135` validation rows; exact schema,
  nulls `0`, duplicate keys `0`, out-of-order rows `0`.
- Positive unique validation pairs: `1,868,310`; removed seen: `87,280`;
  removed cold: `40,260` pairs / `32,679` items.
- Eligible GT: `1,740,770` pairs / `212,238` users.
- Target GT: `1,284,148` pairs / `147,036` labeled users; all `200,152`
  target users сохранены отдельно для all-targets denominator.
- Runtime `10.4035s`, peak memory `6590.13 MiB`, artifact size около `477 MiB`.
- Independent full rerun: deterministic diagnostics и SHA-256 совпали; direct
  anti-joins подтвердили `seen=0`, `cold=0`, `target_diff=0`.

SHA-256 prepared parquet:

- `history_daily.parquet`:
  `e19688615495f03aa0dff861a5e7a9755b8a534288593f99f714eb817cd9a2ea`;
- `validation_daily.parquet`:
  `3b50cb9b64269adaf7f694af351e61959d70c8bd6efc9b2fff960304aa19f541`;
- `ground_truth.parquet`:
  `53b9fb32b40ea8e59c9ca34e3612a358ffba2cff720fa054bc9cfa88c33356a9`;
- `target_ground_truth.parquet`:
  `f478db1a4fd38904a39ca003014e4fcb44b7063937c6e3e4147ea36452b56581`;
- `target_users.parquet`:
  `81d108fb6a8b50e4a1ffae936436a6c120bcb3f53bc7bdbdae78b45b809ed344`.

## Rolling fold artifacts

Три полных history/next-24h fold созданы из raw events до агрегации и могут
переиспользоваться следующими моделями:

- `artifacts/task02_fold_20241129_v1/`: cutoff `2024-11-29 08:56:28`,
  target GT `1,839,297` pairs / `162,562` labeled users;
- `artifacts/task02_fold_20241130_v1/`: cutoff `2024-11-30 08:56:28`,
  target GT `1,703,659` pairs / `161,472` labeled users;
- `artifacts/task02_fold_20241201_v1/`: cutoff `2024-12-01 08:56:28`,
  target GT `1,605,551` pairs / `161,396` labeled users.

Все validation intervals half-open и имеют ровно 24 часа. Prepared calendar
split из `data/` не использовался.

## Current best run

`task08_catboost_pointwise_v1` — текущий best run. Это первый fixed
pointwise baseline, а не результат hyperparameter selection: train fold
`rolling_2`, early stopping fold `rolling_3`, binary `Logloss`, `201`
features, depth `7`, learning rate `0.08`, seed `42`, GPU. CatBoost сохранил
`1030` trees при `best_iteration=1029`.

После фиксации модели выполнена ровно одна canonical оценка:

- `precision_at_20_all_targets`: `0.004788360845757224`;
- `precision_at_20_labeled_users`: `0.006518131614026497`;
- final hits: `19,168`;
- candidate recall: `0.107002463890455`;
- candidate user hit rate: `0.4528618841644223`;
- candidate oracle P@20: all targets `0.03429693432990927`, labeled users
  `0.04668652574879622`;
- coverage `1.0`, mean/p50/p90/p95/p99 candidate count
  `632.82 / 646 / 683 / 688 / 695`.

На идентичном full materialized union RRF получил P@20 all/labeled
`0.003799112674367481 / 0.005171522620310673`; CatBoost добавил `3,960` hits и
`0.000989248171389743` P@20 all targets. Full runtime `2474.94s`, peak RSS
`45149.33 MiB`. Artifact: `artifacts/task08_catboost_pointwise_v1/`, model
SHA-256 `9e8533faa7294f8c7db884110ced2506e608f618cc22fa40e609d2df6077a049`.

## Task06 offline candidates и RRF

`artifacts/task06_candidate_datasets_v1/` содержит narrow outputs четырёх
frozen sources и 392 union shards — по 98 для каждого из трёх rolling folds и
canonical. Все перечисленные в manifest source/union checksums проверены;
checkpoint содержит все 408 завершённых операций. Canonical union:

- `126,660,321` rows для `200,152` target users, mean `632.82` candidates;
- candidate recall `0.107002463890455`, user hit rate
  `0.4528618841644223`;
- oracle P@20 all targets `0.03429693432990927`, labeled users
  `0.04668652574879622`;
- cross-score availability: global и ALS `1.0`, recency
  `0.9279215864295812`, item2item `0.31103323984154435`;
- source-exclusive relevant hits: global `4,872`, recency `6,578`,
  item2item `25,691`, ALS `58,113`.

Rolling selection в `artifacts/task06_candidate_ensemble_v1/` выбрал
`weights_personalized_heavy`: source cap `150`, total cap `600`, RRF constant
`20`, веса global/recency `0.5`, item2item/ALS `1.0`. Средняя rolling P@20 all
targets равна `0.005136596186897958`; canonical:

- P@20 all targets `0.0037963647627802864`, labeled users
  `0.0051677820397725725`, final hits `15,197`;
- selected candidate recall `0.09027853487292742`, oracle P@20 all targets
  `0.028947000279787367`, oracle-minus-RRF `0.02515063551700708`;
- coverage `1.0`, mean candidate count `479.16`, fallback positions/users `0`.

RRF уступает task05 на `0.0000634517766497471` P@20 all targets и `254`
hits, поэтому current best не изменился. Preparation занял `5398.29s`, RRF —
`1221.95s`; общий runtime `6620.24s`, максимальный peak RSS двух jobs —
`14555.66 MiB`. Результат добавлен в `experiments/results.csv`.

## Task07 ranker datasets

- `features.py` строит только из history соответствующего fold user/item
  activity, distinct counterpart counts, daily/counterpart rates, watch-time,
  recency и окна `6h/24h/72h`; item trend сравнивает последние и предыдущие
  `6h/24h`. К task06 provenance/cross-scores добавляются per-user union ranks,
  co-vis seed aggregates и ALS factor norms/cosine. Candidate models при этом
  не запускаются.
- `ranker_data.py` назначает `label UInt8` через intersection с
  `target_ground_truth`, сохраняет полный inference universe и добавляет
  label-safe `is_training_sample`. Все positives, multi-source и top-50
  ALS/item2item hard negatives сохранены; easy negatives выбираются SplitMix64
  с seed 42 и probability `0.05`.
- Full `artifacts/task07_ranker_dataset_v1/` содержит `500,967,826` строк в
  `392` shards и `74,052,862,966` bytes. На каждом fold присутствуют все
  `200,152` target users. Схема содержит `207` колонок, включая `201` model
  feature; исходные `user_id UInt64` и `item_id Int32` сохранены.
- Fold balance `(all rows / positives / training rows)`: rolling-1
  `120,231,001 / 149,772 / 45,403,518`, rolling-2
  `124,260,489 / 158,813 / 46,225,567`, rolling-3
  `129,816,015 / 180,085 / 43,004,977`, canonical
  `126,660,321 / 137,407 / 45,117,924`.
- Независимо проверены SHA-256 всех `392` ranker parts и lookup Parquet,
  точное соответствие candidate IDs task06, labels полному fold GT,
  negative-sampling policy, schema, отсутствие duplicate pairs и nulls.
  Aggregate metrics пересчитаны из опубликованных shards и точно совпали с
  `metrics.json`. Максимальный history timestamp каждого fold на одну секунду
  меньше cutoff; validation timestamp leakage отсутствует.
- Full runner завершил все `396` checkpoint operations (`4` lookups + `392`
  shards) за `911.39s`, peak RSS `11,315.59 MiB`. Compact log содержит
  `run_finish status=completed`, все shard progress events, не содержит ANSI
  или failures. Launcher ограничивает threads, использует `flock`, очищает PID
  после завершения, поддерживает resume и отказывается перезаписывать artifact.
- Два small deterministic smoke run дали byte-identical ranker/lookup Parquet;
  scale smoke прошёл на полном production shard. `research/07_ranker_dataset.ipynb`
  выполняет read-only EDA готовых features и labels. Engineering/full
  preparation не добавлены в `experiments/results.csv`, потому что здесь нет
  нового ranker score.

## Task08 CatBoost pointwise infrastructure

- `catboost==1.2.10` зафиксирован в `requirements.txt`; официальный Linux
  wheel успешно обучил 8-tree GPU smoke непосредственно на NVIDIA GeForce RTX
  5070 Ti (driver `591.86`, 16,303 MiB VRAM). Отдельный CUDA toolkit не нужен.
- `rankers.py` реализует строгий `CatBoostRankerDataLoader` и portable
  `CatBoostPointwiseModel(RankerModel)` с binary `Logloss`, immutable feature
  ordering, Float32 prediction batches, deterministic score/item tie-break,
  atomic `.cbm` artifact и checksum-validated restore. Обучение использует
  только task07 `is_training_sample`; object weight равен
  `1 / sampling_probability`, с дополнительной поправкой для
  early-stopping negative subsample.
- `scripts/run_catboost_ranker.py` не запускает candidate models. Он валидирует
  task07 manifests/checksums, потоково создаёт DSV parts, fit-ит quantization
  borders только на `rolling_2`, переиспользует их для `rolling_3`, обучает
  один fixed GPU baseline с CatBoost snapshot и затем шардированно оценивает
  `rolling_3` и ровно одну конфигурацию на canonical. Для честного сравнения
  сохраняются CatBoost/RRF метрики и рекомендации как на full materialized
  union, так и на frozen task06 RRF parity mask.
- Production config `configs/task08_catboost_pointwise_v1.json`: train
  `rolling_2`, early stopping `rolling_3`, 201 task07 features, 32 borders,
  full task07 training sample, 25% дополнительный negative subsample только
  для eval, `iterations=1200`, depth 7, learning rate 0.08, seed 42. Это один
  baseline profile, не hyperparameter selection; подбор остаётся задачей 09.
- `scripts/run_task08_catboost.sh` задаёт CPU/GPU limits, проверяет минимум
  250 GiB свободного места, использует `flock`/PID, не перезаписывает готовый
  artifact и передаёт SIGTERM дочернему runner. Pool parts, quantized pools,
  fit, best portable model и каждый inference shard имеют atomic checkpoints;
  CatBoost iteration progress попадает в nested tqdm и compact rotating log.
  После успешной atomic publication launcher по умолчанию проверяет `run_id`,
  artifact kind и model SHA, удаляет checkpoint/best-model state и сохраняет
  reusable quantized Pool. При failure state не удаляется; opt-out для
  диагностики — `TASK08_CLEANUP_ON_SUCCESS=0`.
- Real-data `task08_catboost_pointwise_smoke_v1` использовал по одному полному
  production shard на fold, 201 features и 20 GPU trees. Он завершился за
  `19.49s`, peak RSS `5220.74 MiB`; quantized train/eval содержали
  `48,588 / 45,705` строк. Smoke canonical (2,048 users) дал P@20 all/labeled
  `0.0042724609375 / 0.005868544600938967`; это engineering smoke, не full
  model result и не строка `experiments/results.csv`.
- Повторный запуск с тем же checkpoint завершился за `6.99s`, пропустил fit и
  оба inference shard. SHA-256 `model.cbm` и canonical CatBoost
  recommendations побитово совпали с первым запуском. GPU fit в общем случае
  не обязан быть bit-exact, но inference одной сохранённой модели проверен как
  детерминированный. `research/08_catboost_pointwise.ipynb` является read-only
  анализом artifact; все четыре code cells выполнены на production artifact.
- Full `task08_catboost_pointwise_v1` завершился за `2474.94s`; fit занял
  `194.23s`, модель содержит `1030` trees. Rolling-3 P@20 all/labeled равен
  `0.0069114973 / 0.0085711542`; canonical —
  `0.0047883608 / 0.0065181316`, `19,168` hits. На full union CatBoost
  превзошёл RRF на `0.0009892482` P@20 all targets и `3,960` hits.
- Production Pool `artifacts/task08_catboost_pools_v1/` занимает около
  `11 GiB`, содержит `46,225,567` train и `10,886,399` eval rows и может быть
  переиспользован pointwise-конфигурациями Task 09 с тем же fold pair,
  feature order, sampling и quantization. После проверки опубликованного
  artifact удалено `160,857,147,651` bytes production recoverable state;
  smoke checkpoint и smoke pool также очищены.

## Task09 CatBoost selection

- `catboost_selection.py` фиксирует protocol: ровно две связанные пары
  `rolling_1 -> rolling_2` и `rolling_2 -> rolling_3`; primary selection
  metric — mean `precision_at_20_labeled_users`. Tie-break: minimum labeled
  P@20, меньший fold spread, mean `precision_at_20_all_targets`, меньшее
  среднее число trees и стабильный config order. Метрики с разными
  denominators хранятся раздельно.
- `configs/task09_catboost_selection_v1.json` задаёт bounded sequential search
  из максимум 14 unique profiles: `CrossEntropy`, `scale_pos_weight=4/16`,
  дополнительный train-negative keep `0.25`, depth `6/8`, learning rate
  `0.04/0.12` с согласованными iteration budgets, `l2_leaf_reg=10`,
  `random_strength=0`, Bayesian bootstrap и две feature ablations. Каждый
  следующий stage наследует только rolling winner предыдущего stage.
- `rankers.py` поддерживает `Logloss`/`CrossEntropy`, class weighting,
  Bernoulli/Bayesian bootstrap, `random_strength` и `ignored_features`, сохраняя
  backward-compatible restore Task 08 model. Дополнительный negative sampling
  сохраняет positives и корректирует weights на произведение исходной и
  дополнительной sampling probability.
- `scripts/run_catboost_selection.py` читает только immutable Task 07 shards.
  Совместимый `rolling_2 -> rolling_3` Pool переиспользуется из Task 08;
  production создаст один новый base Pool для `rolling_1 -> rolling_2` и два
  train-only Pool для keep `0.25`, всегда с borders соответствующего base
  train fold. Full candidates/features не пересчитываются.
- До atomic `freeze_winner` runner не читает Task 07 canonical fold и даже не
  читает Task 08 canonical metrics. Для rolling reuse используется отдельный
  `evaluation/rolling_3/metrics.json`. После freeze winner ровно одна модель
  последней пары оценивается на canonical; только full mode сравнивается с
  full Task 08/RRF.
- После каждого fit, inference shard, fold evaluation, aggregate и stage
  winner сохраняются checksum-validated atomic checkpoints. CatBoost snapshots
  лежат по `config/fold`; resume пропускает завершённые операции. Nested tqdm,
  timestamped structured log, selection ETA и rotation `5 MiB + 4 backups`
  включены. Partial publication staging перестраивается из checkpoint state.
- `scripts/run_task09_catboost_selection.sh` проверяет CatBoost 1.2.10, GPU,
  disk/RAM thresholds, устанавливает thread limits, использует `flock` и PID,
  передаёт SIGTERM. Только после успешной atomic publication runner проверяет
  artifact/model SHA и удаляет DSV/checkpoint/best-model state; reusable pools
  сохраняются. При failure или interruption всё resume state остаётся.
- Interruption smoke `task09_catboost_selection_smoke_v1` намеренно остановлен
  после первого из двух profiles: aggregate сохранился, canonical не открывался
  и output не существовал. Resume пропустил готовую работу, опубликовал artifact
  и удалил checkpoint/best state только после публикации.
- Финальный `task09_catboost_selection_smoke_v3` использовал один полный Task 07
  shard каждого fold, два GPU profiles по 6 trees и завершился за `31.99s`,
  peak RSS `4805.34 MiB`. Rolling mean P@20 all/labeled —
  `0.0038208008 / 0.0047223899`; canonical shard P@20 all/labeled —
  `0.0029296875 / 0.0040241449`. Это engineering smoke, его scores нельзя
  сравнивать с full Task 08 и он не добавлен в `experiments/results.csv`.
  Пять recommendation outputs побитово совпали между независимыми smoke
  v1/v2/v3;
  GPU `.cbm` hashes могут различаться из-за floating-point reductions.
- Full `task09_catboost_selection_v1` сравнил 14 profiles на двух
  walk-forward парах. Winner `s12_pos_weight_16` использует `Logloss`,
  `scale_pos_weight=16`, все 201 features, depth 7, learning rate 0.08,
  `l2_leaf_reg=3`, Bernoulli `subsample=0.8` и `random_strength=1`.
  Rolling P@20 labeled: `0.0084432595 / 0.0086080200`, mean `0.0085256398`,
  minimum `0.0084432595`, spread `0.0001647605`. Rolling mean P@20 all targets:
  `0.0068763989`.
- Единственная canonical оценка frozen winner: P@20 all/labeled
  `0.0047716236 / 0.0064953481`, `19,101` hits, `688` trees,
  `best_iteration=687`. По сравнению с Task 08: `-0.0000167373` all,
  `-0.0000227835` labeled и `-67` hits. Цель `0.007` не достигнута; менять
  winner по canonical запрещено протоколом. Из `4,003,040` canonical slots
  изменены `868,048`: winner добавил `3,564` своих hits, но потерял `3,631`
  hits Task 08, поэтому net result практически нейтрален.
- Полный run занял `11602.59s` (`3.22h`), peak RSS `45475.70 MiB`. Первый
  запуск корректно остановился по `SIGTERM`; повторный запуск восстановил
  готовые DSV parts из checkpoints. После atomic publication удалено
  `226,520,796,849` bytes recoverable state, PID/checkpoint/best-model
  отсутствуют, reusable pools сохранены.
- Независимая проверка повторила selection aggregation/tie-break для всех 14
  profiles, сверила SHA-256 и row counts четырёх base/negative pools, а также
  пересчитала metrics/hits по всем 29 recommendation outputs. Для winner на
  обоих rolling eval folds и canonical подтверждены ровно `200,152` users,
  20 unique known unseen items на пользователя, `seen=0`, `unknown=0`.

## Task10 CatBoost Learning-to-Rank

- `rankers.py` разделяет pointwise `CatBoostPointwiseModel` и group-aware
  `CatBoostRankerModel`. `catboost_ltr.py` валидирует objective-specific
  параметры, stable dense `Int64 group_id`, непрерывность user groups,
  uniqueness `(user_id, item_id)`, неизменность labels/training mask и
  query/group weighting.
- `scripts/run_catboost_ltr.py` реализует staged objective-first selection на
  двух парах `rolling_1 -> rolling_2` и `rolling_2 -> rolling_3`, isolated fit
  workers, full-pool resource probe, nested progress, rotating structured log,
  atomic checkpoints/resume, graceful SIGTERM, portable models и единственную
  canonical evaluation после freeze winner. Production launcher учитывает
  Windows host drive `G:` и лимиты 50 GiB RAM / 16 GB VRAM.
- Reusable grouped quantized pools содержат `45,403,518 / 10,168,016` rows для
  `rolling_1 -> rolling_2` и `46,225,567 / 10,611,713` rows для
  `rolling_2 -> rolling_3`. Они используют Task 08/09 quantization borders,
  но сохраняют полный eval candidate group; Task 07 candidates, labels и все
  201 features не пересчитывались.
- Full `task10_catboost_ltr_v2` оценил семь `QuerySoftMax` profiles.
  `YetiRankPairwise:mode=Classic;permutations=2;decay=0.85` прошёл limited
  smoke, но isolated one-tree full-pool probe завершился GPU OOM
  (`requested=9539.49 MiB`, `free=7080.33 MiB`) и был явно исключён как
  `resource_infeasible` до чтения canonical.
- Rolling winner `s40_l2_leaf_reg_10`: `QuerySoftMax:beta=2`, 201 features,
  depth 7, learning rate 0.08, `l2_leaf_reg=10`, Bernoulli `subsample=0.8`,
  `random_strength=1`, unit group weighting. P@20 labeled по folds:
  `0.0045772642 / 0.0049006791`, mean `0.0047389716`, minimum `0.0045772642`,
  spread `0.0003234149`; rolling mean P@20 all targets `0.0038222201`.
- Единственная canonical оценка frozen winner: P@20 all/labeled
  `0.0028803110 / 0.0039208085`, `11,530` hits, `424` trees,
  `best_iteration=423`. Относительно Task 08 это
  `-0.0019080499 / -0.0025973231` и `-7,638` hits; относительно Task 09 —
  `-0.0018913126 / -0.0025745396` и `-7,571` hits; относительно full RRF —
  `-0.0009188017 / -0.0012507141` и `-3,678` hits. Group-aware
  `QuerySoftMax` не улучшил top-20; current best остаётся Task 08.
- Full run занял `8429.18s` (`2.34h`), peak RSS `12133.34 MiB`. После atomic
  publication удалено `1,223,117,697` bytes recoverable v2 state; ранее после
  проверки grouped pools было безопасно удалено `314,513,354,098` bytes
  повторяемых v1 raw DSV/parts. Reusable grouped pools сохранены.
- Независимый `--verify-only` проверил SHA-256 95 опубликованных файлов,
  повторил selection aggregation/tie-break и exact group mapping, восстановил
  portable model с deterministic inference и пересчитал canonical P@20/hits.
  Подтверждены `canonical_evaluated_config_count=1`, exact `200,152` target
  users, 20 unique known unseen items, корректные ID dtypes и отсутствие null.

## Проверка и воспроизведение

Из корня репозитория:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
./.venv/bin/python -m compileall -q data_utils.py metrics.py utils.py interfaces.py validation.py popularity.py item2item.py implicit_model.py candidate_pipeline.py ensemble.py experiment_utils.py features.py ranker_data.py rankers.py scripts tests
ruff check data_utils.py metrics.py utils.py interfaces.py validation.py popularity.py item2item.py implicit_model.py candidate_pipeline.py ensemble.py experiment_utils.py features.py ranker_data.py rankers.py scripts tests
# Полный task05 уже опубликован; launcher защищён от overwrite.
./scripts/run_task05_overnight.sh
# Full task06 уже опубликован; launchers защищены от overwrite.
./scripts/run_task06_prepare_overnight.sh
./scripts/run_task06_rrf.sh
# Full task07 уже опубликован; launcher защищён от overwrite и поддерживает resume.
./scripts/run_task07_prepare_overnight.sh
tail -f logs/task07_ranker_dataset_v1.log
kill -TERM "$(cat logs/task07_ranker_dataset_v1.pid)"
# Full task08 опубликован; launcher откажется перезаписывать artifact.
./scripts/run_task08_catboost.sh
tail -f logs/task08_catboost_pointwise_v1.log
watch -n 2 nvidia-smi
kill -TERM "$(cat logs/task08_catboost_pointwise_v1.pid)"
# Full task09 опубликован; launcher откажется перезаписывать artifact.
./scripts/run_task09_catboost_selection.sh
tail -F logs/task09_catboost_selection_v1.log
watch -n 2 nvidia-smi
watch -n 30 'df -h artifacts; du -sh artifacts/.task09_catboost_selection_v1.checkpoint artifacts/task09_catboost_* 2>/dev/null'
kill -TERM "$(cat logs/task09_catboost_selection_v1.pid)"
# Full task10 опубликован; launcher откажется перезаписывать artifact.
./scripts/run_task10_catboost_ltr_v2.sh
./.venv/bin/python scripts/run_catboost_ltr.py --verify-only artifacts/task10_catboost_ltr_v2
```

Последняя проверка task10: production log заканчивается
`run_finish status=completed`, PID/process/checkpoint/best-model copy
отсутствуют, ANSI в логе нет. `--verify-only` проверил 95 checksums, повторил
selection aggregation/tie-break и group mapping, подтвердил единственную
canonical config, portable deterministic inference, `11,530` hits и P@20
all/labeled `0.002880310963667613 / 0.003920808509480672`. Full artifact
добавлен в experiment log.

Последняя проверка task09: 110/110 repository tests, Ruff, `compileall`, CLI
`--help`, shell syntax и все пять code cells notebook прошли. Production log
не содержит ANSI/NUL и заканчивается `run_finish status=completed`. Проверены
atomic publication, model/feature-importance hashes, все 14 leaderboard rows,
четыре stage winners, единственная canonical config, 29 recommendation files
и четыре reusable Pool artifacts. Portable model восстановлен без fit и
повторил scores на 2,048 строках. Full результат добавлен в experiment log.

Последняя проверка task08: full run завершён `run_finish status=completed`,
portable model SHA совпадает с `metrics.json`, production artifact содержит
обе evaluation recommendations и добавлен в `experiments/results.csv`.
Canonical recommendations независимо проверены: `200,152` строк для точного
target set, ровно 20 unique known unseen items, `seen=0`, `unknown=0`,
`null=0`. Сохранённые metrics включают обе P@20, candidate oracle и
full/parity RRF comparisons. Cleanup tests подтверждают удаление только
recoverable state и отказ при чужом `run_id`; production checkpoint очищен
после отдельной проверки artifact. После cleanup-изменения прошли 102/102
repository tests, Ruff, compileall, CLI `--help`, shell syntax и все четыре
code cells notebook на production artifact.

Последняя проверка task07: 91/91 repository tests passed; `compileall`, Ruff,
CLI `--help`, shell syntax и четыре code cells task07 notebook прошли. Во full
artifact проверены SHA-256 всех `392` ranker parts и всех lookup Parquet; все
`396` checkpoint operations завершены. Полный semantic scan подтвердил 1:1
candidate IDs с task06, independently reconstructed labels, negative sampling,
schema, `null=0`, duplicate pairs `0` и точное совпадение fold metrics. History
feature timestamps строго меньше fold cutoffs. Лог не содержит ANSI/failures и
заканчивается `run_finish status=completed`; launcher очищает PID и отдельно
подтвердил отказ от overwrite.

Последняя проверка task06: 80/80 tests passed; `compileall`, Ruff, CLI
`--help`, shell syntax и все пять code cells task06 notebook прошли. Во full
preparation проверены SHA-256 всех source tables и 392 union parts; все 408
preparation и 40 RRF checkpoint operations завершены, failed events и ANSI в
логах отсутствуют. Portable RRF восстановлен из artifact. Независимый пересчёт
canonical candidate metrics, обеих P@20 и `15,197` hits совпал с сохранёнными
metrics. Full recommendations для всех `200,152` targets прошли exact target
set, 20 unique known unseen items, `seen=0`, `unknown=0`, `null=0`; SHA-256
recommendations совпал с manifest.

До full run два независимых real-data preparation smoke runs (8 target, 128
context users) дали одинаковые source-candidate и union-part SHA-256, fold
metrics и candidate counts. Два независимых RRF smoke runs дали byte-identical
recommendations/model config и одинаковые non-runtime canonical metrics.
Engineering smoke не является model experiment и не добавлен в
`experiments/results.csv`.

Ранее для task05: два независимых real-data smoke run с 64 target и
2,000 context users дали одинаковые non-runtime metrics, winner и SHA-256 всех
outputs. Full recommendations отдельно от runner проверены на всех `200,152`
targets: exact target set, ровно 20 unique known unseen items, `seen=0`,
`unknown=0`, `null=0`; повторно вычисленные обе P@20 совпали с `metrics.json`,
все output SHA-256 совпали. Production restore подтвердил float32 factors и
config winner; сам full runner после restore получил bit-exact factors и
идентичный final top-20. CLI отказывается перезаписывать существующий artifact;
для полного воспроизведения нужны новый output/run ID и отдельный checkpoint
path.

## Ограничения и риски

- Все canonical models обязаны читать общий `history_daily.parquet` через
  model-specific loader. Validation artifact не является fit input; любые
  fold-specific user/item features строятся только из history.
- Final-fit aggregate полного raw history ещё не создан: он понадобится после
  локальной проверки моделей и должен использовать тот же daily contract.
- Temporal scores используют `dt=min(raw timestamp)` daily user-item row.
  Поэтому 6-hour score — окно по первому timestamp агрегированной строки, а не
  точное event-level окно; `raw_views` сохраняет raw event count строки,
  `positive_daily_rows` считает положительные daily rows.
- Task02 global popularity остаётся независимым fallback, task03, task04 и
  task05 — отдельными candidate sources для source-aware union. Task06
  намеренно сохраняет их offline candidate tables и cross-scored union для
  обучения ranker без повторного fit/inference candidate models.
- Task06 dataset занимает около `17 GiB`, Task07 — около `69 GiB`
  (`74,052,862,966` bytes), production Task08 quantized Pool — около `11 GiB`.
  Task08 временно создаёт около `150 GiB` DSV/checkpoint state. Launcher
  требует 250 GiB свободного места и после успешной публикации автоматически
  удаляет checkpoint; при failure сохраняет его для resume.
- RTX 5070 Ti поддержана CatBoost 1.2.10. Production Pool содержит 46.2M train
  и 10.9M early-stopping rows; full fit занял `194.23s`, весь pipeline —
  `2474.94s`, peak process RSS `45149.33 MiB`. CatBoost GPU fit не bit-exact
  между независимыми обучениями из-за порядка floating-point reductions;
  сохранённый `.cbm` воспроизводит inference детерминированно.
- Task 09 production занял `3.22h`, peak RSS `44.41 GiB`; GPU profiles
  использовали `gpu_ram_part=0.85`. Сохранены base pools Task 08 и Task 09
  примерно по `11 GiB`, а также train-negative pools примерно `2.2/2.3 GiB`.
  Временный DSV/checkpoint state освобождён только после публикации.
- Task 09 показал расхождение между сильным rolling mean (`0.0085256398`) и
  canonical (`0.0064953481`). Canonical уже открыт и не может использоваться
  для дополнительного выбора pointwise parameters; любое продолжение search
  должно быть заранее задано и оцениваться только на rolling folds.
- Task 10 показал, что семь `QuerySoftMax` profiles существенно уступают
  честному pointwise comparator уже на rolling folds (`0.0047389716` против
  `0.0085256398` mean P@20 labeled) и не переносятся на canonical. Full
  `YetiRankPairwise` не помещается в 16 GB VRAM даже для одного tree при
  текущих 45.4M-row grouped data; его нельзя считать качественно сравнённым
  без другой hardware/data batching architecture.
- Постоянные Task 10 grouped pools занимают около `23 GiB`; опубликованный
  artifact — около `192 MiB`. Удалённые внутри WSL блоки остаются частью
  `G:\\WSL\\Ubuntu\\ext4.vhdx`, пока VHDX отдельно не compact-нут из Windows,
  но повторно доступны Linux и не требуют нового расширения VHDX.
- Submission пока не создавался. Specification подтверждена пользователем
  2026-09-03: CSV с header `user_id,item_ids`, одна строка на target user,
  максимум 20 recommendations; `item_ids` сериализуется как bracketed list,
  например `123,"[5678, 3456, 6789]"`. Task 11 должен реализовать CSV
  writer и round-trip parser/validation; внешняя отправка всё ещё требует
  отдельного явного разрешения.
- Каталог `.git` в workspace не содержит Git metadata, поэтому `git status`
  недоступен.
- От остановленного первого запуска остался незавершённый staging
  `artifacts/.task05_implicit_als_v1.staging-9748468e4d2546eea62283fffb38f2e7/`
  размером около 2.5 MiB с тремя fixed-source union-hit caches. Он не входит в
  опубликованный artifact и не является model checkpoint; удалить его можно
  отдельно, если больше не нужна диагностика старого запуска.
- Progress/logging real-data smoke (8 target, 128 context users) завершился за
  `14.41s`; все 9 phases дошли до конца, artifact опубликован атомарно, а log
  содержал 694 строки / 120,958 bytes. Это проверка observability, не модельный
  experiment и не строка для `experiments/results.csv`.
- Best-model callback real-data smoke (8 target, 128 context users) завершился
  за `14.25s`: rolling checkpoint четыре раза атомарно улучшился, сохранилась
  ровно одна версия, а `ImplicitALSModel.from_artifact()` успешно восстановил
  config `stage4_regularization_regularization_0.01` и factors/mappings.

## Следующие шаги

1. Перейти к Task 11 с pointwise Task 08 architecture как текущим лучшим
   canonical вариантом; Task 09 `s12_pos_weight_16` остаётся честным
   protocol-selected comparator, но не заменяет Task 08 по canonical.
2. Создать full-history daily aggregate и выполнить отдельные final fits, не
   используя последний full-history срез как supervised labels следующего дня.
3. Не продолжать tuning по уже открытому canonical Task 10; любой новый
   ranking experiment должен быть заранее зафиксирован и выбираться только по
   rolling folds.
4. Реализовать и независимо проверить submission serializer для schema
   `user_id,item_ids`; не отправлять файл на Kaggle без отдельного явного
   разрешения.
