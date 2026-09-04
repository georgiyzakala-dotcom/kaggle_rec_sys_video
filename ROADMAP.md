# План разработки рекомендательной системы

Этот документ разбивает разработку на независимые крупные задачи. Каждая
задача рассчитана на отдельное окно Codex и должна завершаться работающим,
проверенным результатом, который можно использовать в следующем окне.

## Как работать с планом

- Выполнять задачи по порядку, если в задаче явно не указано обратное.
- В начале каждого окна читать `AGENTS.md`, `ROADMAP.md`, `PROJECT_STATE.md` и
  `experiments/results.csv`, если последние два файла уже существуют.
- Перед началом работы менять статус только текущей задачи с `[ ]` на `[~]`.
- После выполнения всех критериев готовности менять `[~]` на `[x]` и добавлять
  в задачу итоговый `run_id`, результат и точную команду воспроизведения.
- Не смешивать реализацию нескольких крупных задач в одном окне.
- Для каждой новой модели создавать отдельный notebook в `research/`.
- Notebook должен импортировать готовую реализацию из корневых `.py`-модулей.
  В notebook допустимы анализ, графики, запуск экспериментов и выводы, но не
  единственная копия логики модели.
- Вспомогательные функции, интерфейсы, модели и pipeline-компоненты хранить в
  отдельных `.py`-файлах в корне проекта, например `interfaces.py`,
  `data_utils.py`, `metrics.py`, `popularity.py`, `item2item.py`,
  `implicit_model.py`, `rankers.py` и существующий `utils.py`.
- Не складывать всю новую логику в `utils.py`: функции должны находиться в
  модуле, соответствующем их назначению.
- Все пути должны быть относительны корню проекта. Notebook следует запускать
  из корня проекта командой `./.venv/bin/jupyter lab`.
- Каноническую валидацию всегда строить из `data/train.parquet` по timestamp.
  Подготовленные calendar-split файлы из `data/` для неё не использовать.

Статусы:

- `[ ]` — не начато;
- `[~]` — выполняется;
- `[x]` — завершено и проверено;
- `[!]` — заблокировано, причина описана в `PROJECT_STATE.md`.

## Общая архитектура

Целевой pipeline:

```text
raw interactions
    -> timestamp-level temporal split
    -> shared daily user-item aggregation per fold side
    -> immutable prepared fold artifacts
    -> candidate models
    -> known-item and seen-pair filtering
    -> candidate union and source features
    -> ranker
    -> popularity fallback
    -> deterministic top-20 and final validation
```

Базовый формат кандидатов — long table:

| Field | Dtype | Description |
|---|---|---|
| `user_id` | `UInt64` | Target user |
| `item_id` | `Int32` | Candidate item |
| `score` | `Float32` or `Float64` | Source-specific score |
| `rank` | integer | Rank inside this source and user |
| `source` | string or categorical | Stable candidate-source name |

Каждая candidate model обязана возвращать не более `k` уникальных items на
пользователя, использовать только items из своего history и применять
детерминированный tie-break: `score DESC`, затем `item_id ASC`.

## Общие критерии для любого модельного эксперимента

Каждое новое модельное окно должно:

1. Создать отдельный notebook `research/<nn>_<model_name>.ipynb`.
2. Реализовать переиспользуемый код модели в корневом `.py`-модуле.
3. Добавить или обновить быстрые tests для новой логики.
4. Запустить smoke test, а затем полный canonical holdout, если позволяют
   ресурсы.
5. Сравнить модель с текущим best run на одинаковом split и seed.
6. Сохранить `config.json` и `metrics.json` в новом
   `artifacts/<run_id>/`, не перезаписывая старые runs.
7. Добавить строку в append-only `experiments/results.csv`.
8. Обновить `PROJECT_STATE.md` и статус задачи в этом файле.
9. Записать runtime и peak memory, если они существенны.

Для candidate model дополнительно сохранять:

- `candidate_recall` по relevant pairs;
- `candidate_user_hit_rate` — долю labeled users хотя бы с одним hit;
- `candidate_oracle_p20_all_targets`;
- `candidate_oracle_p20_labeled_users`;
- coverage, mean и quantiles числа кандидатов;
- число final hits и exclusive hits источника после появления ensemble.

`candidate_oracle_p20` вычисляется как среднее
`min(20, relevant_candidates_for_user) / 20`. Это верхняя граница результата
ranker на данном candidate set и более точно соответствует основной метрике,
чем обычный recall.

---

## Задача 00 `[x]`: интерфейсы и каркас проекта

### Цель

Создать минимальные стабильные контракты, через которые будут работать все
candidate models, rankers и experiment runners.

### Подзадачи

- Создать `interfaces.py`.
- Реализовать generic-интерфейсы `Model` и `DataLoader` с отдельными
  fit/predict lifecycle и типами batches.
- Реализовать специализированные `CandidateDataLoader` и `RankerDataLoader`.
- Реализовать специализированный `CandidateModel(Model)`:
  - `fit(loader, **kwargs) -> self`;
  - `predict(loader, *, k, **kwargs) -> pl.DataFrame`;
  - стабильное имя источника через `source_name`;
  - сериализуемая конфигурация через `get_config()`.
- Реализовать специализированный `RankerModel(Model)`:
  - `fit(loader, **kwargs) -> self`;
  - `predict(loader, **kwargs) -> pl.DataFrame`;
  - выходной столбец `ranker_score`.
- Зафиксировать schemas для candidate table, feature table и final
  recommendations.
- Вынести проверки контрактов в `validation.py`:
  - корректные dtypes ID;
  - отсутствие null;
  - отсутствие duplicate `(user_id, item_id, source)`;
  - допустимое число кандидатов;
  - детерминированный порядок при равных scores.
- Реализовать простую тестовую `DummyCandidateModel`, только внутри tests.
- Создать структуру `tests/`, `artifacts/`, `experiments/` и `configs/`.
- Создать `PROJECT_STATE.md` с фактическим текущим состоянием.
- Создать `experiments/results.csv` с колонками, заданными в `AGENTS.md`, и
  дополнительными candidate-метриками из этого плана.
- Добавить test на общий цикл `fit -> predict -> validate`.

### Ожидаемые файлы

- `interfaces.py`;
- `validation.py`;
- `tests/test_interfaces.py`;
- `PROJECT_STATE.md`;
- `experiments/results.csv`.

### Критерии готовности

- Абстрактные интерфейсы не зависят от конкретной модели.
- Dummy model проходит контрактные проверки.
- IDs нигде не преобразуются через floating-point representation.
- Tests запускаются одной документированной командой.
- `PROJECT_STATE.md` содержит следующий этап и команды проверки.

### Итог

- `run_id`: `task00_interfaces_v1` (архитектурная задача, не модельный run).
- Результат: 13 contract/lifecycle tests пройдены; `ruff` и `compileall`
  пройдены; модельный artifact намеренно не создавался.
- Воспроизведение:
  `./.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`.

---

## Задача 01 `[x]`: canonical split, ground truth и метрики

### Цель

Реализовать единственный доверенный validation protocol и создать immutable
canonical fold dataset, который будут переиспользовать все последующие модели.
Raw events обрабатываются один раз при подготовке fold; модели читают только
его агрегированный history snapshot через свои `DataLoader`.

### Подзадачи

- Создать `data_utils.py`.
- Реализовать lazy-загрузку raw train и target users через Polars с проверкой
  исходных schemas.
- Реализовать timestamp-level split raw events до любой агрегации:
  - `cutoff = max(date) - 1 day`;
  - history: `date < cutoff`;
  - validation: `date >= cutoff`.
- Реализовать единый `aggregate_daily_interactions`, применяемый независимо к
  raw history и raw validation. Результат должен иметь одну строку на
  `(user_id, item_id, calendar date)` и схему:
  - `user_id UInt64`, `item_id Int32`, `date Date`, `dt Datetime[us]`;
  - `dt = min(raw timestamp)`;
  - `views UInt32 = count(raw rows)`;
  - `watch_time Int64 = max(watch_time)`;
  - `is_like Int32` и `is_favorite Int32` — maxima event indicators;
  - `is_positive Int32 = (watch_time > 60) OR is_like OR is_favorite`.
- Физически сортировать daily snapshots по `(user_id, item_id, date)` и
  валидировать отсутствие null и duplicate keys. Не объединять строки одной
  пары/даты через timestamp cutoff.
- Реализовать reusable подготовку произвольного temporal fold по переданному
  cutoff, чтобы тот же код позднее создавал rolling-fold snapshots.
- Для canonical fold materialize один общий data artifact
  `artifacts/task01_canonical_data_v1/`, не изменяя файлы в `data/`:
  - `history_daily.parquet`;
  - `validation_daily.parquet`;
  - `ground_truth.parquet` для всех eligible users;
  - `target_ground_truth.parquet`;
  - `target_users.parquet`;
  - `config.json` и `metrics.json` с aggregation/split config, diagnostics,
    runtime, peak memory и checksums.
- Строить ground truth из подготовленной validation части: выбрать
  `is_positive == 1`, затем дедуплицировать по `(user_id, item_id)` за весь
  24-часовой период.
- Удалить из ground truth пары, встречавшиеся в подготовленном history при
  любом event type.
- Отдельно удалить cold validation items, отсутствующие в history, и вернуть
  pair/item counts и shares в diagnostics.
- Ограничить основной validation universe пользователями из
  `target_user_ids.parquet`, сохранив все target users для метрики
  `precision_at_20_all_targets`.
- Создать `metrics.py` и реализовать:
  - `precision_at_20_all_targets` с фиксированным знаменателем 20;
  - `precision_at_20_labeled_users` с фиксированным знаменателем 20;
  - candidate recall;
  - candidate user hit rate;
  - обе версии candidate oracle P@20;
  - coverage и candidate-count statistics.
- Исправить существующий `utils.prec_k`: текущая реализация делит на
  `min(k, len(true))`, что не соответствует метрике соревнования. Оставить
  корректный wrapper или перенести все вызовы в `metrics.py`.
- Добавить synthetic tests:
  - несколько event types одной пары/дня дают одну строку и корректный `views`;
  - повторные watch rows дают `views = row count` и максимальный `watch_time`;
  - одинаковый `(user_id, item_id, date)` по разные стороны cutoff даёт две
    независимые fold rows;
  - relevance, strict `watch_time > 60`, deduplication, seen/cold filtering;
  - сохранение `UInt64` выше `2**53` и обе версии P@20.
- Создать CLI `scripts/prepare_canonical_data.py` с относительными/configurable
  input/output paths. Он должен отказываться перезаписывать существующий run,
  поддерживать явно помеченный limited smoke и выполнять full preparation.
- Создать `research/01_validation_protocol.ipynb` с проверкой split и
  artifact diagnostics, не дублируя функции из `.py`-модулей.
- Выполнить limited smoke, затем full raw preparation; повторным запуском в
  отдельный временный output проверить совпадение diagnostics и checksums.
- На полном raw train сверить canonical sanity checks из `AGENTS.md`. Counts
  history/validation относятся к raw events; ground-truth counts — к
  отфильтрованным unique pairs после daily aggregation.
- Это data-preparation run, а не модельный experiment: не записывать пустую
  score-строку в `experiments/results.csv`; зафиксировать artifact и команду в
  `PROJECT_STATE.md`.

### Ожидаемые файлы

- `data_utils.py`;
- `metrics.py`;
- обновлённый `utils.py`;
- `tests/test_data_utils.py`;
- `tests/test_metrics.py`;
- `scripts/prepare_canonical_data.py`;
- `research/01_validation_protocol.ipynb`;
- `artifacts/task01_canonical_data_v1/` с prepared fold files и manifest
  diagnostics.

### Критерии готовности

- Все synthetic tests проходят.
- Full split совпадает с sanity checks из `AGENTS.md`.
- `history_daily.parquet` и `validation_daily.parquet` уникальны по
  `(user_id, item_id, date)`, имеют зафиксированную схему и созданы только после
  raw timestamp split.
- Все будущие модели canonical fold могут читать один и тот же immutable
  `history_daily.parquet` без повторной подготовки raw events.
- Full preparation детерминирована по diagnostics и checksums.
- Calendar-split prepared files не участвуют в вычислении canonical metrics.
- В metrics явно присутствуют обе версии Precision@20.

### Итог

- `run_id`: `task01_canonical_data_v1`.
- Full raw split: `40,213,747` history и `4,199,689` validation events;
  daily snapshots: `37,694,179` и `3,963,135` rows.
- Eligible ground truth: `1,740,770` pairs / `212,238` users; target ground
  truth: `1,284,148` pairs / `147,036` users при `200,152` target users.
- Результат: 23 tests, `compileall` и Ruff пройдены; повторный full run дал
  идентичные deterministic diagnostics и SHA-256 всех пяти parquet.
- Artifact: `artifacts/task01_canonical_data_v1/`; runtime `10.40s`, peak
  memory `6590.13 MiB`.
- Воспроизведение:
  `./.venv/bin/python scripts/prepare_canonical_data.py --output-dir artifacts/task01_canonical_data_v1 --run-id task01_canonical_data_v1`.

---

## Задача 02 `[x]`: Global Popularity baseline

### Цель

Получить первый полный воспроизводимый baseline и гарантированный fallback для
всех следующих моделей.

### Подзадачи

- Создать `popularity.py`.
- Реализовать `GlobalPopularityModel(CandidateModel)`.
- Проверить минимум следующие варианты item score:
  - raw interaction count;
  - distinct interacting users;
  - relevant interaction count;
  - distinct users с relevant interaction.
- Все статистики считать только по history текущего fold.
- Сформировать достаточно глубокий global list, чтобы после удаления seen
  items можно было заполнить 20 позиций каждому target user.
- Централизованно удалить seen pairs и duplicate items.
- Реализовать deterministic top-k и fallback filling.
- Добавить tests на fit, tie-break, seen filtering и заполнение пользователей
  с большим history.
- Создать `research/02_global_popularity.ipynb`.
- На ранних rolling folds сравнить варианты scoring и зафиксировать baseline.
- После фиксации конфигурации один раз оценить её на canonical holdout.
- Сохранить первый полный run в artifacts и experiment log.

### Ожидаемые файлы

- `popularity.py`;
- `tests/test_popularity.py`;
- `research/02_global_popularity.ipynb`;
- `artifacts/<run_id>/config.json`;
- `artifacts/<run_id>/metrics.json`.

### Критерии готовности

- Для каждого target user построено ровно 20 валидных рекомендаций.
- Нет seen, unknown или duplicate items.
- Получены обе версии Precision@20 и candidate diagnostics.
- Baseline полностью воспроизводится указанной в state командой.

### Итог

- `run_id`: `task02_global_popularity_v1`.
- На трёх ранних rolling folds выбран `relevant_interaction_count` со средним
  `precision_at_20_all_targets = 0.0023764439`; canonical interval не
  участвовал в выборе.
- Canonical результат: `precision_at_20_all_targets = 0.0014036832`,
  `precision_at_20_labeled_users = 0.0019107565`, candidate recall@200
  `0.0245493510`, coverage `1.0`; сохранено ровно 20 unseen known items для
  всех `200,152` target users.
- Full workflow: `285.50s`, peak memory `10859.60 MiB`; повторный top-20
  inference совпал полностью. Результат: 31 tests, `compileall` и Ruff прошли.
- Воспроизведение:
  `./.venv/bin/python scripts/run_global_popularity.py --config configs/task02_global_popularity_v1.json --output-dir artifacts/task02_global_popularity_v1 --run-id task02_global_popularity_v1`.

---

## Задача 03 `[x]`: Recency и Trending Popularity baseline

### Цель

Усилить неперсонализированный baseline временной динамикой и получить сильный
candidate source для ensemble.

### Подзадачи

- В `popularity.py` реализовать отдельный
  `RecencyPopularityModel(CandidateModel)`.
- Добавить configurable exponential time decay.
- Добавить независимые scores по окнам, например 6 часов, 24 часа, 3 дня и
  весь history; точные окна и веса выбирать только по temporal folds.
- Реализовать trending score как сравнение короткого и длинного окон с
  устойчивым smoothing для редких items.
- Сравнить raw-event и relevant-event variants.
- Не использовать canonical validation interval при подборе весов и decay.
- Добавить tests на границы временных окон и отсутствие future leakage.
- Создать `research/03_recency_popularity.ipynb`.
- Провести ablation по источникам, окнам и decay.
- Сравнить с run из задачи 02 на том же canonical split.

### Ожидаемые файлы

- обновлённый `popularity.py`;
- обновлённый `tests/test_popularity.py`;
- `research/03_recency_popularity.ipynb`.

### Критерии готовности

- Выбранная конфигурация основана на temporal-fold результате, а не на
  heuristic preference.
- Есть source-level candidate metrics и полный top-20 результат.
- Global popularity сохранён как fallback независимо от победителя.

### Итог

- `run_id`: `task03_recency_popularity_v1`.
- Реализованы portable `RecencyPopularityConfig`, history-only
  `RecencyPopularityDataLoader` и переиспользуемый
  `RecencyPopularityModel(CandidateModel)`. Модель возвращает отдельный long
  source `recency_popularity`, пригодный для candidate union и ranker; для
  нового fold/full history выбранная конфигурация переобучается через тот же
  loader. Artifact содержит `model_config.json` и fitted canonical ranking.
- Из 26 window/decay/trending/blend вариантов на трёх ранних folds выбран
  `window_positive_6h` со средним
  `precision_at_20_all_targets = 0.0028989635`; canonical interval не участвовал
  в выборе.
- Canonical результат: `precision_at_20_all_targets = 0.0015438267`,
  `precision_at_20_labeled_users = 0.0021015262`, candidate recall@200
  `0.0259417139`, coverage `1.0`, `6,180` hits и `0` fallback positions. Прирост
  к task02: `+0.0001401435` P@20 all targets и `+561` hits.
- Full workflow: `2281.37s`, peak memory `10585.15 MiB`; 38 tests,
  `compileall`, Ruff, notebook execution, два deterministic smoke и independent
  recommendation validation прошли.
- Воспроизведение:
  `./.venv/bin/python scripts/run_recency_popularity.py --config configs/task03_recency_popularity_v1.json --output-dir artifacts/task03_recency_popularity_v1 --run-id task03_recency_popularity_v1`.

---

## Задача 04 `[x]`: Item-to-Item Co-Visitation

### Цель

Добавить первый сильный персонализированный candidate generator, который
предлагает новые items по недавней истории пользователя.

### Подзадачи

- Создать `item2item.py`.
- Реализовать `Item2ItemModel(CandidateModel)`.
- Построить sparse top-neighbor table без полного item-item квадрата.
- Ограничить число history items на пользователя до configurable значения,
  чтобы контролировать квадратичную генерацию пар.
- Сравнить варианты co-vis signal:
  - все interactions;
  - только relevant/positive interactions;
  - направленные переходы по времени;
  - time-distance weighting;
  - cosine/Jaccard-like normalization против raw pair count.
- Для inference использовать несколько последних пользовательских seed items.
- Учитывать recency и strength каждого seed.
- Объединять соседей нескольких seeds, удалять seen items и сохранять лучший
  score пары.
- Добавлять popularity fallback, если персональных соседей недостаточно.
- Сохранять neighbor table только внутри уникального artifact run.
- Добавить synthetic tests на построение пар, направление, normalization,
  seed aggregation, seen filtering и deterministic ordering.
- Создать `research/04_item2item_covisitation.ipynb`.
- Провести ablations на одинаковых folds и сравнить с popularity baselines.

### Ожидаемые файлы

- `item2item.py`;
- `tests/test_item2item.py`;
- `research/04_item2item_covisitation.ipynb`.

### Критерии готовности

- Full co-vis computation укладывается в зафиксированные runtime/memory.
- Neighbor table имеет configurable top-k на item и не содержит duplicates.
- Измерены standalone recall, oracle P@20 и exclusive hits относительно
  popularity sources.
- Есть воспроизводимая лучшая co-vis конфигурация.

### Итог

- `run_id`: `task04_item2item_v1`.
- Реализованы portable `Item2ItemConfig`, history-only
  `Item2ItemDataLoader`, sparse SciPy CSR fit, bounded neighbor table,
  частичный `Item2ItemModel(CandidateModel)` и restore без повторного fit.
- Семь последовательных ablation stages выполнены только на трёх rolling
  folds. Победили: `all`, `undirected`, time-distance pair weight с half-life
  24h, `raw`, `min_pair_users=1`, `neighbor_k=100`, `seed_k=5`, seed half-life
  6h и `event_strength`. Средний rolling P@20 all targets `0.0040754860`.
- Единственная canonical оценка: P@20 all targets `0.0027968744`, P@20
  labeled users `0.0038072309`, candidate recall@200 `0.0372091067`, oracle
  P@20 all targets `0.0119359287`, coverage `0.9981764`, `11,196` final hits.
  Прирост к task03: `+0.0012530477` P@20 all targets и `+5,016` hits.
- Item2item даёт `35,635` candidate hits, отсутствующих в union task02/task03;
  union трёх sources достигает recall `0.0617483343` и oracle P@20 all targets
  `0.0197982034`.
- Artifact содержит `22,805,416` links для `740,940` seed items, portable
  config и validated recommendations для всех `200,152` targets. Full runtime
  `5246.89s`, peak RSS `27856.02 MiB`; 47 tests, compileall, Ruff, notebook,
  два deterministic limited smoke и artifact restore/validation прошли.
- Воспроизведение:
  `./.venv/bin/python scripts/run_item2item.py --config configs/task04_item2item_v1.json --output-dir artifacts/task04_item2item_v1 --run-id task04_item2item_v1`.

---

## Задача 05 `[x]`: Implicit ALS

### Цель

Получить latent collaborative candidate source, дополняющий локальные
item-to-item связи.

### Подзадачи

- До установки проверить доступные зависимости в `.venv`.
- Если требуется новая библиотека, согласовать и зафиксировать dependency и
  причину её добавления.
- Создать `implicit_model.py`.
- Реализовать `ImplicitALSModel(CandidateModel)`.
- Построить стабильные integer mappings для `UInt64 user_id` и `Int32 item_id`.
  Никогда не преобразовывать исходные IDs во float.
- Строить sparse user-item matrix только из history fold.
- Сделать configurable interaction confidence:
  - event strength;
  - `watch_time > 60`;
  - like/favorite weights;
  - repeated views;
  - time decay.
- Реализовать batch top-k inference и seen-item filtering.
- Сериализовать factors, ID mappings и config в artifact run.
- Добавить tests на mappings, score consistency, unknown users и seen items.
- Создать `research/05_implicit_als.ipynb`.
- Провести ограниченный ablation по factor count, regularization, iterations и
  confidence weighting.
- Сравнить standalone и complementary recall с popularity/co-vis.

### Ожидаемые файлы

- `implicit_model.py`;
- `tests/test_implicit_model.py`;
- `research/05_implicit_als.ipynb`.

### Критерии готовности

- Повторный fit с seed 42 даёт детерминированный результат в пределах
  документированной численной погрешности.
- Model artifact можно загрузить без повторного fit.
- Зафиксированы runtime, peak memory и candidate metrics.
- Показана добавочная ценность ALS относительно уже существующих sources.

### Итог

- Rolling-only winner: `event_strength`, без decay, `factors=128`,
  `regularization=0.1`, `iterations=15`; средний прирост union oracle P@20
  относительно task02+task03+task04 равен `0.015399546344777968`, ALS-exclusive
  rolling hits — `185425`.
- Единственная canonical оценка: P@20 all targets `0.0038598165394300335`,
  P@20 labeled users `0.005254155444925053`, candidate recall@200
  `0.0674260287754994`, standalone oracle P@20 `0.021622566849194613`.
- Union task02+task03+task04+ALS: candidate recall `0.107002463890455`, oracle
  P@20 all targets `0.03429693432990927`; canonical ALS-exclusive candidate
  hits `58113`, final exclusive hits `6709`.
- Artifact `artifacts/task05_implicit_als_v1/` восстановлен без fit; top-20 и
  factors совпали точно. Все `200152` targets имеют ровно 20 unique known
  unseen items. Runtime `30496.54s`, peak RSS `21516.93 MiB`.
- ALS canonical P@20 выше task04 на `0.0010629421639553944`, поэтому task05
  становится текущим best run и одновременно сохраняется как отдельный source
  для task06.

---

## Задача 06 `[x]`: Offline Candidate Union, cross-model scoring и RRF

### Цель

Заранее материализовать global, recency, co-vis и ALS candidates для каждого
temporal fold, объединить их в ограниченный candidate set и рассчитать score
каждой модели для каждой пары из union. Candidate generation и будущий ranker
training должны быть физически разделены воспроизводимыми offline artifacts.

### Подзадачи

- Создать `candidate_pipeline.py`.
- Реализовать запуск или portable restore нескольких `CandidateModel` с
  отдельным cap на источник.
- Для каждого rolling/canonical fold один раз сохранить narrow source
  candidate tables до построения union. Повторный ensemble/ranker experiment
  не должен заново fit или запускать candidate models.
- Реализовать union и deduplication по `(user_id, item_id)`.
- Хранить provenance отдельно от cross-model scores:
  - `generated_by_<source>`;
  - `generator_score_<source>`;
  - `generator_rank_<source>`;
  - `generator_rank_norm_<source>`;
  - `source_count`.
- Для каждой пары union независимо вычислить:
  - `cross_score_global_popularity` и полный history-only popularity rank;
  - `cross_score_recency_popularity` и temporal rank;
  - sparse `cross_score_item2item` по recent user seeds;
  - `cross_score_implicit_als` как dot product user/item factors;
  - отдельный availability flag для каждого cross-score.
- Кандидат, не вошедший в top-k конкретного source, может иметь cross-score
  этого source; это не должно менять его provenance.
- Корректно заполнять missing source values без смешивания смысла score,
  rank, availability и generator membership.
- Реализовать rank normalization внутри source.
- Материализовать union features шардированно под уникальным artifact path;
  не держать одновременно в памяти полные outputs всех sources или полный
  wide union.
- Создать `ensemble.py` и реализовать простой `RRFEnsembleModel`.
- Подобрать source caps и RRF weights только по temporal folds.
- Контролировать итоговый candidate cap, ориентировочно 300–800 items/user.
- Измерить pair recall, user hit rate, обе oracle P@20 и exclusive hits каждого
  источника.
- Создать отдельный production CLI/launcher для долгой материализации source
  candidates и cross-scores, а также отдельный CLI для RRF ablation, читающий
  только готовые datasets. Codex выполняет unit tests и ограниченные smoke
  runs; все запуски дольше 20 минут пользователь запускает из терминала.
- Добавить многоуровневый terminal progress: общий run, stage, config/fold и
  model iterations/batches, с elapsed time и ETA там, где объём работы известен.
- Писать отдельный компактный rotating log без ANSI-кодов. По timestamp и полям
  `stage/config/fold/operation` должно быть сразу видно, какая конфигурация
  обучается или оценивается, что уже завершено, текущая/best metric и сколько
  заняла каждая фаза.
- Добавить atomic checkpoints после каждого завершённого `(stage, config,
  fold)`, atomic best-model callback и безопасное resume без повторного
  обучения уже завершённых моделей. Финальный artifact публиковать атомарно и
  не перезаписывать.
- Launcher должен задавать thread/resource limits, блокировать параллельный
  duplicate run и печатать команды запуска, `tail -f` для наблюдения и
  корректной остановки.
- Добавить tests на union, source provenance, missing values и caps.
- Добавить synthetic tests на progress/log events, checkpoint/resume,
  best-model callback, atomic publish и отказ от overwrite.
- Создать `research/06_candidate_ensemble.ipynb`.

### Ожидаемые файлы

- `candidate_pipeline.py`;
- `ensemble.py`;
- `experiment_utils.py`;
- `scripts/prepare_candidate_datasets.py`;
- `scripts/run_candidate_ensemble.py`;
- `scripts/run_task06_prepare_overnight.sh`;
- `scripts/run_task06_rrf.sh`;
- `configs/task06_candidate_ensemble_v1.json`;
- `tests/test_candidate_pipeline.py`;
- `tests/test_ensemble.py`;
- `tests/test_experiment_utils.py`;
- `research/06_candidate_ensemble.ipynb`.

### Критерии готовности

- Source candidates и union features воспроизводимы, имеют checksums и не
  содержат seen/unknown/duplicate pairs.
- Повторный RRF/ranker experiment читает offline datasets и не вызывает fit
  или inference candidate models.
- Provenance не изменяется при добавлении cross-score другой модели.
- Oracle P@20 заметно выше результата простого RRF; иначе ranker пока не имеет
  достаточного пространства для улучшения.
- Размер candidate table подходит для последующего построения features.
- Limited smoke подтверждает progress/logging, checkpoints, resume и portable
  restore; full run передан пользователю одной воспроизводимой terminal
  командой и не запускается Codex автоматически.

### Итог

- Full `task06_candidate_datasets_v1` материализовал четыре source candidate
  tables и 392 checksum-protected union shards: по 98 частей на каждый из
  трёх rolling folds и canonical. Canonical содержит `126,660,321` union rows;
  полный artifact занимает около `17 GiB`.
- Для каждой union-пары сохранены generator provenance и независимые
  cross-scores всех четырёх моделей. Canonical availability: global `1.0`,
  ALS `1.0`, recency `0.9279215864`, sparse item2item `0.3110332398`.
- Rolling selection выбрал cap `150` на source (`600` total), RRF constant
  `20`, веса global/recency `0.5` и item2item/ALS `1.0`. Средняя rolling P@20:
  all targets `0.0051365962`, labeled users `0.0063538326`.
- Единственная canonical оценка RRF: P@20 all targets `0.0037963648`, labeled
  users `0.0051677820`, `15,197` hits. Это на `0.0000634518` и `254` hits ниже
  task05 ALS, поэтому current best не изменился.
- Selected union имеет recall `0.0902785349` и oracle P@20 all targets
  `0.0289470003`; oracle-minus-RRF равен `0.0251506355`. Полный materialized
  cap-800 union имеет recall `0.1070024639` и oracle P@20 `0.0342969343`, то
  есть supervised CatBoost ranker получает существенный headroom.
- Preparation занял `5,398.29s`, RRF `1,221.95s`; peak RSS соответственно
  `14,555.66 MiB` и `12,098.72 MiB`. Все `408` preparation и `40` RRF
  checkpoints завершены; portable restore, logs, checksums, обе метрики,
  candidate metrics и exact top-20 semantic invariants независимо проверены.

---

## Задача 07 `[x]`: Ranker dataset и fold-specific features

### Цель

Добавить labels и fold-specific history features к offline candidate union из
задачи 06, не запуская candidate models повторно.

### Подзадачи

- Создать `ranker_data.py` и `features.py`.
- Определить минимум 2–3 rolling 24-hour folds, сохранив canonical final
  holdout как главный сравнимый результат.
- Читать immutable per-fold candidate union, model scores и provenance из
  artifacts задачи 06; ranker-data pipeline не вызывает candidate model fit
  или inference.
- Назначить binary label по relevant new pairs следующего 24-hour периода.
- Не добавлять случайные negatives из всего каталога как замену реальным
  candidate negatives.
- Если нужен downsampling, сохранять все positives и hard negatives, а seed и
  sampling probability записывать в config.
- Реализовать fold-specific user/item features:
  - activity и distinct counterpart counts;
  - positive/like/favorite rates;
  - statistics по нескольким временным окнам;
  - item trend;
  - user recency/activity;
  - source generator scores/ranks/presence и cross-model scores из task06;
  - дополнительные co-vis aggregate features;
  - дополнительные доступные ALS factor-derived features.
- Все aggregates считать только из history соответствующего fold.
- Добавить validators на отсутствие validation timestamps в features.
- Сохранять datasets под уникальным `artifacts/<run_id>/`, а не в `data/`.
- Добавить synthetic tests на labels, feature cutoff и negative sampling.
- Создать `research/07_ranker_dataset.ipynb` для EDA готовых features и labels.

### Ожидаемые файлы

- `ranker_data.py`;
- `features.py`;
- `tests/test_ranker_data.py`;
- `tests/test_features.py`;
- `research/07_ranker_dataset.ipynb`.

### Критерии готовности

- Dataset содержит только candidates, реально доступные ranker на inference.
- Нет feature или label leakage.
- Схема, class balance, число users/pairs и размер файла записаны в artifact.
- Dataset детерминирован при том же seed и configs.

### Итог

- Full run `task07_ranker_dataset_v1` опубликован в
  `artifacts/task07_ranker_dataset_v1/`: `500,967,826` строк в `392`
  ranker shards для трёх rolling folds и canonical, по `200,152` users на
  fold. Размер artifact — `74,052,862,966` bytes.
- Схема содержит `207` колонок, из них `201` model feature. Все labels,
  `is_training_sample`, source provenance/cross-scores и fold-specific
  history features материализованы без повторного запуска candidate models.
- Positive/training rows: rolling-1 `149,772 / 45,403,518`, rolling-2
  `158,813 / 46,225,567`, rolling-3 `180,085 / 43,004,977`, canonical
  `137,407 / 45,117,924`. Все positives и hard negatives сохранены.
- Независимая проверка всех shards подтвердила manifest SHA-256, точное 1:1
  совпадение candidate IDs с task06, labels с fold target ground truth,
  sampling policy, отсутствие duplicates/nulls и timestamp leakage. Feature
  history каждого fold заканчивается за одну секунду до cutoff.
- Runtime `911.39s`, peak RSS `11,315.59 MiB`; все `396` checkpoint operations
  (`4` lookups + `392` shards) завершены. Полный запуск воспроизводится
  командой `./scripts/run_task07_prepare_overnight.sh` с config
  `configs/task07_ranker_dataset_v1.json`.

---

## Задача 08 `[x]`: CatBoost pointwise baseline и training infrastructure

### Цель

Получить первый CatBoost ranker на immutable datasets задачи 07 и подготовить
воспроизводимую infrastructure для долгого обучения без candidate generation.

### Подзадачи

- Зафиксировать CatBoost dependency и причину его добавления.
- Создать `rankers.py` и реализовать portable pointwise CatBoost baseline с
  binary `Logloss` на candidate-level labels.
- Создать отдельный training CLI/config/launcher, который читает только
  готовые datasets и никогда не вызывает candidate models.
- Добавить temporal train/eval split, early stopping, CatBoost snapshots,
  atomic outer checkpoints, best-model callback, progress и rotating logs по
  правилам `AGENTS.md`.
- При необходимости один раз создать и переиспользовать quantized CatBoost
  pools; quantization borders fit только на training side.
- Сравнить с RRF на идентичном candidate set.
- Добавить tests на feature ordering, dataset manifest, model persistence,
  resume и deterministic inference.
- Создать `research/08_catboost_pointwise.ipynb`.

### Ожидаемые файлы

- `rankers.py`;
- `tests/test_rankers.py`;
- `scripts/run_catboost_ranker.py`;
- shell launcher и CatBoost config;
- `research/08_catboost_pointwise.ipynb`.

### Критерии готовности

- CatBoost model загружается из artifact и воспроизводит scores/top-20.
- Training runner поддерживает progress/logging/checkpoint/resume и передан
  пользователю для долгого запуска.
- Результат сравнивается с RRF на том же candidate union и temporal split.

### Итог

- Full GPU run `task08_catboost_pointwise_v1` обучен на immutable Task 07
  `rolling_2`, использовал `rolling_3` для early stopping и после фиксации
  модели ровно один раз оценён на canonical. CatBoost остановился на `1030`
  деревьях (`best_iteration=1029`); полный runtime `2474.94s`, peak RSS
  `45149.33 MiB`.
- Canonical full-union результат: `precision_at_20_all_targets =
  0.0047883608`, `precision_at_20_labeled_users = 0.0065181316`, `19,168`
  hits. На том же candidate set RRF дал соответственно `0.0037991127` и
  `0.0051715226`; прирост CatBoost равен `0.0009892482` P@20 all targets.
- Candidate recall равен `0.1070024639`, oracle P@20 all/labeled —
  `0.0342969343 / 0.0466865257`, coverage `1.0`, mean candidate count
  `632.82`. Portable model и canonical recommendations сохранены с SHA-256.
- Reusable quantized pools `rolling_2 -> rolling_3` содержат
  `46,225,567 / 10,886,399` строк и `201` features, занимают около `11 GiB`.
  Они сохраняются для Task 09. Launcher после успешной atomic publication
  проверяет финальный artifact и удаляет только resume checkpoint и временную
  best-model копию; при ошибке они остаются для resume.
- Для завершённого production run очищено `160,857,147,651` bytes
  recoverable state (около `149.8 GiB`), production Pool и Task 07 datasets
  сохранены. Результат добавлен в `experiments/results.csv`.

---

## Задача 09 `[x]`: CatBoost loss и hyperparameter selection

### Цель

Выбрать устойчивую CatBoost pointwise конфигурацию по walk-forward temporal
validation, не меняя offline candidates и features.

### Подзадачи

- Использовать только CatBoost backend и datasets задачи 07.
- Обучать binary objective на candidate-level labels; корректно учитывать
  negative downsampling и sampling probability.
- Использовать отдельный temporal fold для early stopping.
- Не выбирать модель по training loss; основной критерий — обе версии P@20 на
  честном holdout.
- Сохранять feature list, model params, best iteration и feature importance.
- Проверить устойчивость к удалению потенциально нестабильных features.
- Добавить tests на feature ordering, model persistence и inference schema.
- Подбирать loss/profile и гиперпараметры отдельным долгим runner с nested
  progress, rotating logs, snapshots, atomic checkpoints и resume.
- Создать `research/09_catboost_selection.ipynb`.
- Сравнить с RRF и task08 baseline на идентичном candidate set.

### Ожидаемые файлы

- обновлённый `rankers.py`;
- обновлённый `tests/test_rankers.py`;
- `research/09_catboost_selection.ipynb`.

### Критерии готовности

- GBDT воспроизводимо загружается из artifact и выдаёт те же scores.
- Зафиксированы обе P@20, runtime, memory и candidate oracle ceiling.
- Улучшение или отсутствие улучшения подтверждено на одинаковом split.

### Итог

- Full run `task09_catboost_selection_v1` честно сравнил `14` bounded profiles
  на двух связанных walk-forward парах: `rolling_1 -> rolling_2` и
  `rolling_2 -> rolling_3`. Основной критерий — mean
  `precision_at_20_labeled_users`; canonical не открывался до фиксации
  winner и был оценён ровно для одной конфигурации.
- Выбран `s12_pos_weight_16`: binary `Logloss`, `scale_pos_weight=16`, все
  `201` features, depth `7`, learning rate `0.08`, `l2_leaf_reg=3`, Bernoulli
  bootstrap `subsample=0.8`, `random_strength=1`, seed `42`. На двух selection
  folds P@20 labeled равен `0.0084432595 / 0.0086080200`, mean
  `0.0085256398`, minimum `0.0084432595`, spread `0.0001647605`; mean P@20 all
  targets равен `0.0068763989`.
- Единственная canonical оценка дала `precision_at_20_all_targets =
  0.0047716236`, `precision_at_20_labeled_users = 0.0064953481` и `19,101`
  hits. Цель `0.007` не достигнута; относительно Task 08 это
  `-0.0000167373` P@20 all, `-0.0000227835` P@20 labeled и `-67` hits.
  Это зафиксированный negative result, а не основание подбирать другую
  конфигурацию по canonical. Winner изменил `868,048` из `4,003,040`
  canonical top-20 slots: получил `3,564` новых hits, но потерял `3,631`
  baseline hits, то есть крупная перестановка не дала устойчивого net gain.
- Class weighting `16` улучшил rolling mean относительно Task 08 profile на
  `0.0000860917`. Ближайшие `random_strength=0`, `l2_leaf_reg=10` и learning
  rate `0.04` находятся почти на том же уровне. Negative keep `0.25` не помог,
  а удаление ALS-scale и fold-scale feature groups заметно ухудшило rolling
  mean до `0.0081719036 / 0.0081956196`.
- Full runtime `11602.59s` (`3.22h`), peak RSS `45475.70 MiB`. Переиспользован
  Task 08 Pool `rolling_2 -> rolling_3`; сохранены новый base Pool
  `rolling_1 -> rolling_2` и два reusable train-only negative Pool.
  Остановленный `SIGTERM` запуск успешно продолжен по checkpoints. После
  atomic publication очищено `226,520,796,849` bytes recoverable state.
- Независимо пересчитаны обе метрики и hits для всех `29` recommendation
  outputs; winner recommendations на обоих rolling eval folds и canonical
  содержат все `200,152` target users, ровно 20 unique known unseen items и
  проходят checksum/semantic validation. Portable model восстановлен без fit
  и дважды выдал побитово одинаковые scores на 2,048 строках. Результат
  добавлен в `experiments/results.csv`.

---

## Задача 10 `[x]`: CatBoost Learning-to-Rank

### Цель

Проверить, улучшает ли group-aware ranking objective top-20 относительно
pointwise GBDT.

### Подзадачи

- В `rankers.py` реализовать `CatBoostRankerModel(RankerModel)` с выбранными
  group-aware objectives, например `QuerySoftMax` и `YetiRankPairwise`.
- Группировать training rows строго по `user_id` и валидировать непрерывность
  групп перед fit.
- Настроить ranking objective с фокусом на верхние позиции до 20.
- Использовать тот же candidate set, folds и features, что в задаче 09.
- Сравнить ranking objective с pointwise без изменения candidate generation.
- Провести минимальный ablation по depth/leaves, learning rate и
  regularization, не подбирая параметры на canonical holdout.
- Добавить tests на group construction и сохранение порядка rows.
- Создать `research/10_lambdarank.ipynb`.

### Ожидаемые файлы

- обновлённый `rankers.py`;
- обновлённый `tests/test_rankers.py`;
- `research/10_lambdarank.ipynb`.

### Критерии готовности

- Pointwise и ranking модели сравниваются на полностью одинаковых данных.
- Best ranker выбран по заранее определённым temporal folds.
- Canonical holdout используется для итогового отчёта, а не для скрытого
  подбора гиперпараметров.

### Результат

- Full run `task10_catboost_ltr_v2` завершён с теми же immutable Task 07
  candidates, labels, 201 features и двумя walk-forward парами, что и
  pointwise comparator Task 09. Stable dense `Int64 group_id` построен из
  исходного `user_id UInt64`; grouped pools и непрерывность групп независимо
  проверены.
- По rolling folds выбран `s40_l2_leaf_reg_10`: `QuerySoftMax:beta=2`, depth
  `7`, learning rate `0.08`, `l2_leaf_reg=10`, Bernoulli `subsample=0.8`, unit
  group weighting. Mean P@20 all/labeled равен
  `0.0038222201 / 0.0047389716`.
- Ровно одна canonical оценка frozen winner дала P@20 all/labeled
  `0.0028803110 / 0.0039208085`, `11,530` hits. Это хуже Task 08 на
  `0.0019080499 / 0.0025973231` и `7,638` hits, поэтому current best остаётся
  Task 08 pointwise.
- `YetiRankPairwise` успешно прошёл limited GPU smoke, но full-pool one-tree
  resource probe не поместился в 16 GB VRAM и был заранее предусмотренно
  исключён как `resource_infeasible`; canonical при этом не открывался.
- Full runtime `8429.18s` (`2.34h`), peak RSS `12133.34 MiB`. Независимый
  `--verify-only` повторил checksums 95 файлов, selection aggregation/tie-break,
  exact group mapping, deterministic portable inference и обе canonical
  метрики. Artifact: `artifacts/task10_catboost_ltr_v2/`.

---

## Задача 11 `[ ]`: полный fit, inference и подготовка submission

### Цель

Переобучить выбранный pipeline на всей доступной истории и получить строго
проверенные top-20 для всех target users.

### Подзадачи

- Создать `pipeline.py` с orchestration полного fit/predict.
- Зафиксировать best candidate и ranker configs из предыдущих задач.
- Пересчитать все aggregates и features из полного `data/train.parquet`.
- Не использовать текущие `item_features_data.parquet` и
  `user_features_data.parquet`, потому что они заканчиваются до последнего
  calendar date.
- Fit всех выбранных candidate models на full history.
- Финальный supervised ranker обучить на candidate-label examples из всех
  доступных rolling folds. После фиксации архитектуры и гиперпараметров можно
  добавить examples из canonical fold, но нельзя заново подбирать по нему
  параметры.
- Не пытаться обучать ranker непосредственно на последнем full-history срезе:
  для него ещё нет labels следующего дня.
- Сгенерировать и rerank candidates для всех target users.
- Применить seen filtering и global fallback до ровно 20 unique items.
- Проверить:
  - ровно одна группа рекомендаций на target user;
  - нет missing/extra/duplicate users;
  - ровно 20 unique items на пользователя;
  - все items известны из train;
  - нет train-seen pairs;
  - нет null;
  - dtypes ID сохранены.
- Повторить inference с тем же seed и проверить детерминированность.
- Создать `research/11_full_fit_and_inference.ipynb` с итоговым отчётом.
- Сохранить внутренний validated prediction artifact.
- Official submission specification подтверждена пользователем 2026-09-03:
  CSV с header `user_id,item_ids`, одна строка на target user, `item_ids` —
  строковый bracketed list целых ID, например
  `123,"[5678, 3456, 6789]"`; максимум 20 items на пользователя.
- Добавить serializer через стандартный CSV writer и отдельный round-trip test:
  после чтения CSV `item_ids` должен однозначно разбираться в исходный ordered
  list без изменения ID или rank order.
- Не отправлять submission на Kaggle без явного запроса пользователя.

### Ожидаемые файлы

- `pipeline.py`;
- `tests/test_pipeline.py`;
- `research/11_full_fit_and_inference.ipynb`;
- полный artifact выбранного pipeline;
- submission serializer и validated `submission.csv`.

### Критерии готовности

- Все output invariants проходят на полном target set.
- Full run воспроизводится одной точной командой.
- `PROJECT_STATE.md` содержит best validation metrics, full-fit config и
  artifact paths; `submission.csv` точно соответствует подтверждённой schema.

---

## Отложенные эксперименты

Эти модели не начинать до завершения основного каскада и анализа его ошибок.
Каждая из них также требует отдельного Codex-окна, отдельного `.py`-класса и
отдельного notebook:

- `research/12_implicit_bpr.ipynb` — BPR на той же sparse matrix;
- `research/13_lightgcn.ipynb` — graph collaborative retrieval;
- `research/14_sequence_model.ipynb` — SASRec/GRU-like retrieval;
- user clustering и segment-specific popularity;
- sparse top-k SLIM-like model, только после оценки вычислительной стоимости.

Переходить к отложенной модели стоит только при наличии конкретной гипотезы:
какие users/items или типы relevant interactions она должна покрыть сверх
текущего candidate union.
