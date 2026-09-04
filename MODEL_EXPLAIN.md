# Как устроена итоговая рекомендательная система

## Краткий ответ

Финальный submission построен двухэтапной рекомендательной системой:

1. Четыре независимых candidate generator предлагают каждому пользователю до
   200 новых видео каждый:
   `global_popularity`, `recency_popularity`, `item2item` и `implicit_als`.
2. Их объединение, в среднем 632.48 уникального кандидата на пользователя,
   ранжирует pointwise `CatBoostClassifier` с 201 числовым признаком.
3. Из результата берутся 20 строк с максимальным `RawFormulaVal`; окончательный
   tie-break — `item_id ASC`.
4. Если CatBoost не дал бы 20 строк, список дополнялся бы глобальной
   популярностью. В production run fallback не понадобился ни одному
   пользователю.

Главная часть качества системы — сочетание широкого candidate union и
supervised reranking. Самый сильный отдельный генератор — ALS, но CatBoost
научился использовать не только ALS score, а также согласие источников,
поведение пользователя, популярность и тренд видео, co-visitation evidence и
геометрию ALS factors.

Финальный CatBoost не является просто скопированной моделью Task08. Из Task08
были заморожены архитектура, гиперпараметры, число деревьев и границы
квантования. Затем новая модель была обучена с нуля на примерах из всех четырёх
исторических folds и применена к кандидатам, заново построенным на полном
`train.parquet`.

## Что именно было предсказано

Цель — для каждого из 200,152 пользователей вернуть 20 видео, с которыми в
следующий 24-часовой период вероятно произойдёт релевантное взаимодействие.
Пара `(user_id, item_id)` считается релевантной, если есть хотя бы одно из:

- `like`;
- `favorite`;
- `watch_time > 60`, строго больше 60.

Повторные релевантные события одной пары дают одну положительную пару. Видео,
уже встречавшиеся у пользователя в history с любым типом события, запрещены и
в ground truth, и в рекомендациях. Кандидатами могут быть только видео,
известные в history.

Основная локальная метрика — macro Precision@20 с фиксированным знаменателем
20. В проекте всегда отдельно хранятся:

- `precision_at_20_all_targets`: среднее по всем target users, в том числе без
  релевантных событий;
- `precision_at_20_labeled_users`: среднее только по пользователям с
  непустым ground truth.

Это разные знаменатели, поэтому их нельзя сравнивать друг с другом как одну
метрику.

## Как развивалась система по ROADMAP

| Task | Что было сделано | Главный итог |
|---|---|---|
| 00 | Контракты `Model`, `DataLoader`, candidate/ranker schemas и validators | Единый воспроизводимый интерфейс всех моделей |
| 01 | Exact timestamp split, daily aggregation, ground truth и метрики | Канонический holdout без temporal leakage |
| 02 | `GlobalPopularityModel` | Первый baseline и гарантированный fallback |
| 03 | `RecencyPopularityModel` | Выбрана популярность положительных daily rows за последние 6 часов |
| 04 | `Item2ItemModel` | Персональные кандидаты по co-visitation последних видео пользователя |
| 05 | `ImplicitALSModel` | Самый сильный отдельный candidate generator |
| 06 | Offline union, cross-model scores и RRF | Четыре источника объединены; создан простор для supervised ranker |
| 07 | Labels, negative sampling и 201 fold-specific feature | Immutable обучающие datasets без повторного запуска генераторов |
| 08 | Pointwise CatBoost baseline | Лучший canonical score: `0.0047883608 / 0.0065181316` |
| 09 | Pointwise loss/hyperparameter search | `scale_pos_weight=16` выиграл rolling selection, но чуть проиграл Task08 на canonical |
| 10 | Group-aware CatBoost LTR | `QuerySoftMax` оказался существенно хуже pointwise; `YetiRankPairwise` не поместился в 16 GB VRAM |
| 11 | Full-history fit, inference и serializer | Создан и строго проверен `submission.csv` |

На момент этого разбора статус Task11 в `ROADMAP.md` и верхняя часть
`PROJECT_STATE.md` ещё не обновлены, а `experiments/results.csv` заканчивается
Task10. Фактический artifact `artifacts/task11_full_fit_v1/` при этом полностью
опубликован и имеет `status=complete`. Поэтому источником истины для финальной
системы здесь являются Task11 config, manifests, model artifact и production
log.

## Общий pipeline

```text
raw train events
    -> split raw timestamps for each historical fold
    -> aggregate each side to daily user-item rows
    -> fit four candidate models on fold history only
    -> generate top-200 per source and remove seen pairs
    -> union by (user_id, item_id)
    -> add provenance and cross-model scores
    -> add user/item/co-vis/ALS features
    -> label by the next 24-hour ground truth
    -> deterministic negative sampling + inverse-probability weights
    -> frozen CatBoost quantization
    -> pointwise CatBoost Logloss

For final inference:
full raw history
    -> full daily aggregate
    -> refit all four candidate models
    -> 126,592,350 candidate pairs
    -> rebuild the same 201 features
    -> CatBoost RawFormulaVal
    -> per-user top-20
    -> global-popularity fallback if needed
    -> validated submission.csv
```

## Предобработка событий

### Разбиение делалось до агрегации

Это принципиальное решение. Raw events сначала делились точным timestamp
cutoff, и лишь затем history и validation агрегировались независимо. Благодаря
этому события одной пары в один календарный день, но по разные стороны cutoff,
не сливались.

Canonical cutoff вычислялся как:

```text
cutoff = max(train.date) - 1 day
       = 2024-12-02 08:56:28
```

Canonical history содержит `date < cutoff`, validation — `date >= cutoff`.
Три selection folds имеют cutoffs:

- `rolling_1`: `2024-11-29 08:56:28`;
- `rolling_2`: `2024-11-30 08:56:28`;
- `rolling_3`: `2024-12-01 08:56:28`;
- `canonical`: `2024-12-02 08:56:28`.

У rolling folds validation interval half-open и ровно 24 часа. Каждый fold
обучает candidate models и вычисляет features только по своей левой, прошлой
части.

### Shared daily interaction contract

Внутри каждой уже разделённой части raw rows группировались по
`(user_id, item_id, calendar date)`:

- `dt = min(raw timestamp)`;
- `views = count(raw event rows)`;
- `watch_time = max(watch_time)`;
- `is_like = max(event_type == "like")`;
- `is_favorite = max(event_type == "favorite")`;
- `is_positive = watch_time > 60 OR is_like OR is_favorite`.

`views` — число event rows, а не гарантированное число физических просмотров.
Временные окна ниже используют `dt`, то есть первый timestamp агрегированной
daily row. Это приближение: например, `6h` означает окно по первому событию
пары за день, а не повторный просмотр всех raw events на уровне секунд.

Canonical artifact содержит:

- 40,213,747 raw history events;
- 4,199,689 raw validation events;
- 37,694,179 daily history rows;
- 3,963,135 daily validation rows.

После relevance, deduplication, seen filtering и cold-item filtering осталось
1,740,770 eligible GT pairs. Для target users — 1,284,148 pairs у 147,036
labeled users.

### Ground truth

Ground truth строился в следующем порядке:

1. В validation оставить `is_positive == 1`.
2. Дедуплицировать `(user_id, item_id)` за весь следующий 24-часовой период.
3. Удалить любую пару, уже встречавшуюся в history.
4. Удалить validation items, которых вообще не было в history.
5. Для обучения и основной оценки пересечь users с `target_user_ids`.

Calendar-split snapshots из `data/prepared_*` для canonical validation и
final fit не использовались.

### Полная история перед submission

Для Task11 все 44,413,436 raw events были заново агрегированы тем же кодом.
Получено:

- 41,637,896 daily rows;
- 408,025 users;
- 1,884,722 items;
- 40,744,692 уникальных observed `(user_id, item_id)` pairs;
- время от `2024-11-27 08:56:29` до `2024-12-03 08:56:28`.

Для inference был определён условный следующий период:

```text
prediction_start         = 2024-12-03 08:56:28.000001
prediction_end_exclusive = 2024-12-04 08:56:28.000001
```

## Candidate model 1: Global Popularity

Это не обучаемая ML-модель, а history-only статистический ranking всего
каталога.

На rolling folds сравнивались четыре item score:

- raw interaction count;
- distinct interacting users;
- relevant interaction count;
- distinct relevant users.

Победил `relevant_interaction_count`:

```text
score(item) = sum(is_positive) over daily history rows of item
```

То есть считаются положительные daily user-item rows, а не число уникальных
пользователей. Items сортируются по `score DESC, item_id ASC`. Для каждого
пользователя ranking просматривается сверху, already-seen items пропускаются.

Роль источника двойная:

- он даёт до 200 кандидатов;
- он является гарантированным fallback для финального top-20.

На full history ranking содержит все 1,884,722 известных item. Источник дал
ровно 200 кандидатов всем 200,152 target users.

Canonical standalone результат Task02:

- candidate recall@200: `0.0245493510`;
- P@20 all/labeled: `0.0014036832 / 0.0019107565`.

## Candidate model 2: Recency Popularity

Здесь также нет параметрического ML fit: на каждом fold пересчитывается
временной ranking items.

В Task03 было проверено 26 вариантов окон, exponential decay, trending и
blends. Победитель выбирался только по трём rolling folds:

```text
config_id    = window_positive_6h
signal       = positive_daily_rows
window_hours = 6
```

Финальный score:

```text
score(item) = sum(is_positive)
              for daily rows with dt >= prediction_start - 6 hours
```

Items с нулевым score не входят во fitted ranking. На полной истории
положительный 6-часовой ranking содержит 220,761 item. После seen filtering
источник всё равно дал по 200 кандидатов каждому target user.

Canonical standalone результат Task03:

- candidate recall@200: `0.0259417139`;
- P@20 all/labeled: `0.0015438267 / 0.0021015262`.

## Candidate model 3: Item-to-Item Co-Visitation

Это персонализированный граф соседей item-to-item.

### Подготовка fit data

Daily history сначала схлопывается до одной строки на `(user_id, item_id)`:

- `last_dt = max(dt)`;
- `views = sum(views)`;
- `watch_time`, `is_like`, `is_favorite`, `is_positive` — maxima.

У каждого пользователя оставляются 10 самых недавних уникальных items
(`history_cap=10`). Победивший `profile=all` использует все взаимодействия, не
только positive.

### Построение item graph

Для каждой пары items среди последних 10 items одного пользователя создаётся
undirected связь. Вклад пользователя в связь зависит от временной дистанции:

```text
pair_contribution = 2 ** (-delta_hours / 24)
```

Вклады суммируются по пользователям. Выбран `normalization=raw`: cosine или
Jaccard normalization не применяется. `min_pair_users=1`. Для каждого seed
item сохраняются top-100 neighbors по `score DESC, neighbor_item_id ASC`.

Поскольку профиль пользователя уже уникален по item, одна item-pair получает
не более одного вклада от одного пользователя. Full-history neighbor table
содержит 22,962,473 связей.

### Inference

Берутся пять последних items пользователя (`seed_k=5`). Сила seed:

```text
seed_recency = 2 ** (-age_hours / 6)

seed_strength = 1
                + log1p(views)
                + I(watch_time > 60)
                + is_like
                + is_favorite

contribution(seed, candidate)
    = neighbor_score * seed_recency * seed_strength
```

Если candidate найден от нескольких seeds, generator score равен максимуму
вкладов, а не сумме. Seen items удаляются, затем остаются top-200.

На full history источник сгенерировал 37,575,195 rows и покрыл 199,804 из
200,152 users (`99.826%`). Пользователи без подходящих соседей остаются
покрыты другими источниками.

Canonical standalone результат Task04:

- candidate recall@200: `0.0372091067`;
- P@20 all/labeled: `0.0027968744 / 0.0038072309`.

## Candidate model 4: Implicit ALS

Это основной latent collaborative retrieval source и самый сильный отдельный
генератор.

### Матрица и confidence

Daily rows схлопываются до уникальных `(user_id, item_id)` pairs. Стабильные
integer mappings строятся сортировкой исходных IDs; `UInt64 user_id` никогда
не проходит через float. Full fit использует sparse matrix размером
`408,025 x 1,884,722` с 40,744,692 nonzero pairs.

Confidence для observed pair:

```text
repeated_views = log1p(max(views - 1, 0))

signal = 1
         + 1 * repeated_views
         + 1 * I(watch_time > 60)
         + 2 * is_like
         + 2 * is_favorite

confidence = 1 + signal
```

У победителя нет time decay (`half_life_hours=null`). Таким образом, первое
любое взаимодействие уже имеет положительную implicit confidence, repeated
views дают сублинейный прирост, а like/favorite сильнее long watch.

### Обучение

Использована библиотека `implicit==0.7.3` и CPU
`AlternatingLeastSquares`:

- 128 factors;
- `regularization=0.1`;
- 15 alternating iterations;
- float32 factors;
- conjugate-gradient solver (`use_cg=true`);
- `alpha=1`;
- 8 threads;
- seed 42.

Это weighted implicit-feedback matrix factorization. Она обучает user и item
vectors так, чтобы observed pairs с большей confidence получали высокий dot
product, а ненаблюдаемые пары — низкий.

На inference используется `user_factor · item_factor`, observed items
исключаются внутри `implicit.recommend`. Для корректного детерминированного
tie-break модель запрашивает дополнительные позиции и сортирует равные scores
по `item_id ASC`.

Full-history ALS дал ровно 200 candidates всем target users. Model artifact
включает factors и оба ID mapping; его размер около 1.1 GiB.

Canonical standalone результат Task05:

- candidate recall@200: `0.0674260288`;
- candidate oracle P@20 all: `0.0216225668`;
- P@20 all/labeled: `0.0038598165 / 0.0052541554`.

## Candidate union и cross-scoring

Каждый источник даёт до 200 строк. Union строится по
`(user_id, item_id)` без дополнительного heuristic pruning: сумма source caps
равна hard cap 800, а пересечения источников естественно уменьшают размер.

Для каждой пары сохраняется provenance каждого генератора:

- `generated_by_<source>`;
- `generator_score_<source>`;
- `generator_rank_<source>`;
- `generator_rank_norm_<source>`;
- `source_count`.

Normalized generator rank считается как:

```text
(source_cap - rank + 1) / source_cap
```

Затем каждая модель независимо оценивает уже весь union, даже если пара не
попала в её собственный top-200:

- global score и полный global rank;
- recent score и recent rank, если item был активен в выбранном окне;
- sparse item2item score, если candidate связан хотя бы с одним seed;
- ALS dot product для любой mapped user-item pair.

Generator membership и cross-score намеренно разделены. Например, видео может
быть сгенерировано ALS, но одновременно получить global score, recent rank и
item2item evidence.

Full-history union:

| Показатель | Значение |
|---|---:|
| Rows | 126,592,350 |
| Users | 200,152 |
| Mean candidates/user | 632.48 |
| Min / p50 / p90 / p95 / p99 / max | 418 / 648 / 684 / 688 / 693 / 699 |
| Global source rows | 40,030,400 |
| Recency source rows | 40,030,400 |
| Item2item source rows | 37,575,195 |
| ALS source rows | 40,030,400 |
| Global cross-score availability | 100% |
| Recency cross-score availability | 91.16% |
| Item2item cross-score availability | 31.20% |
| ALS cross-score availability | 100% |

Все пары к этому моменту уже known и unseen.

### Почему не RRF

Task06 проверил Reciprocal Rank Fusion. Лучший rolling profile использовал
`rrf_constant=20`, source cap 150, веса `0.5/0.5/1.0/1.0` для
global/recency/item2item/ALS.

На canonical он дал P@20 all `0.0037963648`, то есть немного хуже standalone
ALS. При этом полный union имел candidate recall `0.1070024639` и oracle P@20
all `0.0342969343`. Большой разрыв между RRF и oracle показал, что проблема уже
не столько в генерации, сколько в выборе правильных 20 из примерно 633. Это и
стало основанием обучать supervised ranker.

## Как готовился dataset для ranker

### Единица наблюдения

Одна строка ranker dataset — одна реально доступная на inference candidate
pair `(user_id, item_id)`.

Label:

```text
label = 1, если candidate pair есть в target_ground_truth следующего fold
label = 0, иначе
```

Таким образом, random negatives из всего каталога не создавались. Все
negatives — реальные конкуренты, предложенные хотя бы одним candidate source.
Это важнее для top-20 ranking, чем множество очевидно нерелевантных случайных
items.

Положительный GT pair, который не нашёл ни один генератор, отсутствует в
ranker dataset: ranker физически не может его восстановить. Именно поэтому
candidate recall и oracle P@20 измерялись отдельно.

### Task07: первый уровень negative sampling

Полный inference universe сохранялся целиком, но для fit добавлялся
`is_training_sample`:

- все positives сохранялись;
- все multi-source negatives сохранялись;
- negatives из top-50 `item2item` сохранялись;
- negatives из top-50 ALS сохранялись;
- остальные easy negatives сохранялись с вероятностью 0.05.

Случайность детерминирована hash-функцией SplitMix64 от исходных `UInt64`
`user_id`, `Int32 item_id`, fold и seed 42. Поэтому порядок файлов и
шардирование не влияют на выбор строки.

`sampling_probability` равна 1 для positive/hard rows и 0.05 для easy
negative. Task07 artifact сохранил все 500,967,826 inference rows в 392 shards;
из них training rows:

- `rolling_1`: 45,403,518;
- `rolling_2`: 46,225,567;
- `rolling_3`: 43,004,977;
- `canonical`: 45,117,924.

### 201 признак

В CatBoost не передаются `user_id` и `item_id`: они нужны только для
идентичности пары и группировки результата. Все 201 model features доступны и
на historical folds, и на full-history inference.

Группы признаков:

| Группа | Количество | Содержание |
|---|---:|---|
| Candidate provenance и cross-scores | 27 | membership, source scores/ranks, normalized ranks, availability, `source_count` |
| User history | 66 | activity, views, distinct items, positives/likes/favorites, rates, recency |
| Item history и trend | 88 | activity, users, rates, windows, previous windows и log-trends |
| Rank cross-score внутри union | 4 | ALS и item2item score ranks и normalized ranks у конкретного пользователя |
| Дополнительное co-vis evidence | 11 | число matched seeds, sum/mean/max contributions, neighbor scores/support/rank |
| ALS geometry | 5 | user/item norms, norm product, availability и cosine similarity |

User и item aggregates вычислялись за весь доступный history и отдельно за
окна 6h, 24h и 72h. Среди них:

- число daily rows, views и distinct counterparts;
- positive/like/favorite counts;
- mean/max watch time;
- доли positive/like/favorite rows;
- доли positive/liked/favorited distinct counterparts;
- доля недавней активности относительно полного history;
- часы с последнего interaction/positive/like/favorite и availability flags.

Для items дополнительно строились предыдущие окна `[12h, 6h)` и `[48h, 24h)`
относительно cutoff. Trend — сглаженный log-ratio:

```text
trend = log(current_count + 1) - log(previous_count + 1)
```

Он вычислялся для daily rows, views и positive rows на 6h и 24h горизонтах.

Co-vis features, в отличие от generator score=max, сохраняют более полную
картину: сколько seeds подтвердили candidate, сумму/среднее/максимум их
вкладов, базовые neighbor scores, co-user support и лучший neighbor rank.

ALS features включают raw dot product, ранги внутри union, norms vectors и:

```text
als_cosine_similarity = als_dot_product /
                        (user_factor_norm * item_factor_norm)
```

Missing numerical score заполняется нулём только вместе с отдельным
availability flag. Это позволяет модели отличить настоящий нулевой score от
отсутствия sparse evidence.

## Как готовился финальный обучающий dataset Task11

После выбора архитектуры использованы все четыре temporal datasets:
`rolling_1`, `rolling_2`, `rolling_3` и `canonical`. Canonical здесь
использован только как дополнительный final-training fold после заморозки
модели; после его включения никакой локальной оценки или нового tuning уже не
проводилось.

Чтобы общий Pool оставался около 46 млн строк, каждому fold выделили примерно
11.5 млн rows. Все positives сохранились, negatives из Task07 были ещё раз
детерминированно subsampled. Получилось:

| Fold | Rows | Positives | Negatives | Second-stage negative probability |
|---|---:|---:|---:|---:|
| rolling_1 | 11,500,357 | 149,772 | 11,350,585 | 0.250813 |
| rolling_2 | 11,496,707 | 158,813 | 11,337,894 | 0.246190 |
| rolling_3 | 11,499,139 | 180,085 | 11,319,054 | 0.264330 |
| canonical | 11,503,551 | 137,407 | 11,366,144 | 0.252611 |
| **Всего** | **45,999,754** | **626,077** | **45,373,677** | — |

Итоговая observed positive rate в sampled Pool — около 1.361%.

### Зачем нужны веса

После sampling каждой строке назначен inverse-probability weight:

```text
sample_weight = 1 / full_sampling_probability
```

Для positives вероятность равна 1 и weight равен 1. Для hard negatives
учитывается только второй sampling stage, поэтому weights примерно 3.78–4.06.
Easy negatives сначала проходили Task07 с вероятностью 0.05, затем Task11 с
вероятностью около 0.25; их итоговые weights примерно 75.66–81.24.

Это не class balancing и не `scale_pos_weight`. В финальной модели
`scale_pos_weight=1`. IPW пытается восстановить вклад полного candidate
universe после двух уровней downsampling. Поэтому редкие сохранённые easy
negatives получают большой вес.

Fold provenance сохранялся в sampled Parquet, но `fold_id`, `user_id`,
`item_id` и `sampling_stratum` не были CatBoost features. В Pool попали только
`label`, `sample_weight` и 201 признаков.

## Как выполнялась квантовка CatBoost

Все признаки числовые. Boolean и integer features, как и Float64 scores, перед
записью в CatBoost transport приводились к Float32. Категориальных признаков у
ranker нет.

DSV имел строго такой порядок колонок:

```text
0: label
1: sample_weight
2..202: 201 numerical features в зафиксированном порядке
```

`columns.cd` помечает первые две колонки как `Label` и `Weight`, остальные как
`Num`.

Границы квантования не подбирались заново на данных Task11. Они были получены
в Task08 только на training fold `rolling_2`:

- `border_count=32` — максимум 32 thresholds на feature;
- `feature_border_type=GreedyLogSum`;
- quantization выполнялась на CPU;
- seed 42.

После fit borders на `rolling_2` те же thresholds использовались для
`rolling_3` early-stopping Pool. В Task11 этот frozen `borders.tsv` с SHA-256
`7616264960...08409ef8` был снова использован через `input_borders`. То есть
новые full-training значения лишь раскладывались по уже выбранным bins.
Binary/constant-like признаки естественно имеют меньше 32 borders; всего файл
содержит 5,492 border rows.

Размеры final training transport:

- временный TSV DSV: 65,545,871,663 bytes;
- quantized binary Pool: 9,476,907,080 bytes;
- rows: 45,999,754;
- features: 201.

Квантовка выполнялась `catboost.utils.quantize(..., task_type="CPU",
input_borders=...)`. Уже сохранённый quantized Pool затем загружался как
`quantized://...` для GPU fit. Это отделяет дорогую сериализацию и quantization
от обучения и гарантирует одинаковый feature order и bins.

## Как выбирался ranker

### Task08: pointwise baseline, который оказался лучшим

Task08 обучал `CatBoostClassifier` как binary classifier по отдельным
candidate rows:

- train: `rolling_2`, 46,225,567 rows;
- early stopping: sampled `rolling_3`, 10,886,399 rows;
- objective: `Logloss`;
- максимум 1,200 trees;
- depth 7;
- learning rate 0.08;
- L2 leaf regularization 3;
- `Plain` boosting;
- Bernoulli bootstrap, `subsample=0.8`;
- `random_strength=1`;
- `scale_pos_weight=1`;
- seed 42;
- GPU.

Early stopping с patience 80 оставил 1,030 trees (`best_iteration=1029`).
После freeze модель ровно один раз оценили на canonical full union. Это дало:

- P@20 all: `0.004788360845757224`;
- P@20 labeled: `0.006518131614026497`;
- 19,168 hits.

На том же union RRF дал P@20 all `0.0037991127`, то есть CatBoost улучшил его
примерно на 26.0% относительно и добавил 3,960 hits.

### Task09: pointwise hyperparameter selection

На двух walk-forward парах было проверено 14 profiles: другие losses,
`scale_pos_weight`, depth, learning rate, L2, bootstrap, random strength,
дополнительный negative sampling и feature ablations.

Rolling winner `scale_pos_weight=16` чуть улучшил средний rolling score, но
после единственной canonical проверки получил 19,101 hits и P@20 all
`0.0047716236`: на 67 hits хуже Task08. Поэтому он не был перенесён в финальный
submission pipeline.

### Task10: Learning-to-Rank

Group-aware `QuerySoftMax:beta=2` обучался с user groups, но оказался сильно
хуже pointwise: canonical P@20 all `0.0028803110`, 11,530 hits.
`YetiRankPairwise` прошёл smoke, однако full Pool не поместился в доступные
16 GB VRAM даже для одного дерева. Поэтому итоговый ranker остался pointwise.

### Почему final ranker именно такой

Финальный pipeline взял фактически лучшую canonical архитектуру Task08:

```text
model              = CatBoostClassifier, pointwise
loss_function      = Logloss
iterations         = 1030, fixed
depth              = 7
learning_rate      = 0.08
l2_leaf_reg         = 3
boosting_type       = Plain
bootstrap_type     = Bernoulli
subsample           = 0.8
random_strength     = 1
scale_pos_weight    = 1
border_count        = 32
random_seed         = 42
task_type           = GPU
```

В Task11 отсутствовали eval Pool и early stopping. Поле
`early_stopping_rounds=80` сохранилось в portable config ради совместимости,
но при `fixed_tree_budget=true` оно не используется. Модель обязана построить
ровно 1,030 trees; поэтому `best_iteration=1029` в final artifact означает
последнее дерево, а не новый выбор на validation.

Objective оптимизирует weighted binary Logloss, а не Precision@20 напрямую и
не pairwise/listwise loss. После fit берётся raw logit `RawFormulaVal`; его
калибровка как вероятности не нужна, потому что используется только порядок
внутри пользователя.

## Что выучил финальный CatBoost

Top feature importance финальной модели:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `cross_score_implicit_als` | 14.177 |
| 2 | `als_cosine_similarity` | 12.987 |
| 3 | `als_user_factor_norm` | 9.135 |
| 4 | `user_views_all` | 5.612 |
| 5 | `source_count` | 4.303 |
| 6 | `item_distinct_users_all` | 3.036 |
| 7 | `union_rank_cross_score_implicit_als` | 2.663 |
| 8 | `union_rank_norm_cross_score_implicit_als` | 2.462 |
| 9 | `als_factor_norm_product` | 2.282 |
| 10 | `user_positive_daily_rate_all` | 2.133 |
| 11 | `user_positive_counterpart_rate_all` | 2.128 |
| 12 | `user_positive_daily_rate_72h` | 1.829 |
| 13 | `generator_score_implicit_als` | 1.713 |
| 14 | `item_positive_rows_prev_6h` | 1.695 |
| 15 | `item_daily_rows_all` | 1.284 |
| 16 | `covisit_matched_seed_count` | 1.275 |
| 17 | `user_daily_rows_all` | 1.263 |
| 18 | `covisit_best_neighbor_rank` | 1.066 |
| 19 | `item_trend_daily_rows_log_ratio_24h` | 1.003 |
| 20 | `item_daily_row_share_6h` | 0.952 |

Это CatBoost `FeatureImportance`, а не причинный анализ. Тем не менее картина
очень ясная:

- ALS является главным personalized signal;
- raw ALS dot product недостаточен сам по себе — cosine и norms дают модели
  информацию о масштабе/уверенности vectors;
- согласие нескольких generators (`source_count`) сильно повышает доверие;
- activity/positivity пользователя меняют интерпретацию scores;
- item popularity и trend корректируют latent recommendation;
- co-visitation помогает как независимое локальное подтверждение.

Иными словами, CatBoost в значительной степени является нелинейным
калибратором ALS, но не только им.

## Финальный fit и inference

Task11 выполнил следующие production phases:

1. Проверил версии, ресурсы, checksums Task07 schema и Task08 borders.
2. Создал full-history daily aggregate.
3. Заново fit-нул четыре frozen candidate configurations на полной истории.
4. Для 98 user shards сгенерировал и cross-scored candidate union.
5. Построил full-history user/item lookups и 201 признаков для каждого
   candidate shard.
6. Подготовил 45,999,754 weighted training rows из четырёх folds.
7. Создал quantized Pool с frozen borders.
8. Обучил новый CatBoost на всех temporal examples с fixed 1,030 trees.
9. Посчитал scores и top-20 по 98 full-history shards.
10. Повторил inference из сохранённого portable `.cbm` и сравнил outputs.
11. Объединил shards, провалидировал Parquet и CSV, затем атомарно опубликовал
    artifact.

Production запуск был возобновляемым. Первая сессия успела подготовить
history, candidate/features, sampled rows и quantized Pool. Финальная сессия
восстановила checkpoints и завершила fit/inference за 1,088.77 секунды.
CatBoost fit занял около 190 секунд. Зафиксированы peak RSS 12.56 GiB и peak
VRAM usage 14,395 MiB.

Для каждого candidate модель выдала `RawFormulaVal`. Внутри каждого user:

```text
sort by ranker_score DESC, item_id ASC
take first 20
```

Global fallback был подготовлен из `generated_by_global_popularity`, но
`fallback_users=0` и `fallback_positions=0`.

## Локальные результаты компонентов

| Run | Модель/ranker | P@20 all targets | P@20 labeled users | Hits |
|---|---|---:|---:|---:|
| Task02 | Global popularity | 0.0014036832 | 0.0019107565 | 5,619 |
| Task03 | 6h positive recency | 0.0015438267 | 0.0021015262 | 6,180 |
| Task04 | Item2item | 0.0027968744 | 0.0038072309 | 11,196 |
| Task05 | ALS | 0.0038598165 | 0.0052541554 | 15,451 |
| Task06 | RRF union | 0.0037963648 | 0.0051677820 | 15,197 |
| Task08 | Pointwise CatBoost | **0.0047883608** | **0.0065181316** | **19,168** |
| Task09 | Tuned pointwise CatBoost | 0.0047716236 | 0.0064953481 | 19,101 |
| Task10 | QuerySoftMax LTR | 0.0028803110 | 0.0039208085 | 11,530 |

Task11 не имеет честной локальной P@20: canonical labels вошли в final
training, поэтому оценивать эту же модель на canonical было бы leakage.
Её истинная внешняя оценка — результат отправленного Kaggle submission. Само
число leaderboard score в текущих локальных файлах не записано.

## Проверки финального submission

Файл `artifacts/task11_full_fit_v1/submission.csv` прошёл следующие проверки:

- ровно 200,152 строки и одна строка на target user;
- missing users: 0;
- extra users: 0;
- duplicate users: 0;
- ровно 20 уникальных item на пользователя;
- unknown items: 0;
- train-seen pairs: 0;
- null values: 0;
- fallback positions: 0;
- повторное portable inference полностью совпало;
- CSV round-trip восстановил те же ordered item lists;
- SHA-256 submission:
  `100613900ba1bb2660a5c32e3245790be6ccd9c4aa9dab382fb625c389aeea46`.

CSV сериализован как `user_id,item_ids`, где `item_ids` — JSON-подобный
bracketed list целых IDs, например `"[5678, 3456, 6789]"`.

## Почему система получилась сильной

1. Temporal validation повторяет next-day задачу и не смешивает события через
   cutoff.
2. Candidate sources дополняют друг друга: popularity ловит массовый спрос,
   recency — краткосрочный тренд, item2item — локальную последовательность,
   ALS — глобальную collaborative структуру.
3. Cross-scoring превращает union из четырёх отдельных списков в совместное
   представление каждой пары.
4. Ranker обучается на настоящих hard competitors, а не на случайном каталоге.
5. Features вычисляются относительно времени конкретного fold и доступны на
   inference без leakage.
6. CatBoost моделирует нелинейные взаимодействия: один и тот же ALS score
   интерпретируется по-разному в зависимости от norms, user behavior, item
   trend и подтверждения другими источниками.
7. Final fit использует все доступные исторические supervised windows и затем
   refit retrieval на максимальном history.

## Главные ограничения и точки дальнейшего роста

- Candidate recall на canonical равен только 10.70%. Всё, чего нет в union,
  ranker восстановить не может. Новые retrieval sources могут дать больший
  эффект, чем тонкий tuning CatBoost.
- Canonical oracle P@20 all `0.0342969343` намного выше лучшего ranker
  `0.0047883608`. Значит, даже внутри текущего union остаётся большой ranking
  headroom, хотя oracle использует недоступное знание labels.
- В данных нет item/user metadata. Система целиком collaborative и behavioral;
  она не понимает content.
- Pointwise Logloss не совпадает напрямую с top-20 objective. Но проверенный
  QuerySoftMax оказался хуже, а YetiRankPairwise был resource-infeasible.
- Временные признаки опираются на `dt=min(timestamp)` daily row. Более точный
  raw-event temporal feature pipeline потенциально лучше передал бы последние
  часы.
- Final borders заморожены на `rolling_2`. Это обеспечивает consistency, но
  не обязательно оптимально использует распределение всех четырёх folds.
- GPU fit CatBoost в общем случае не гарантирует bit-exact model bytes между
  независимыми fit из-за floating-point reductions. Сохранённая `.cbm` даёт
  детерминированный inference, что и было проверено.
- Task09 показал, что небольшое улучшение rolling mean не обязательно
  переносится на canonical. Дальнейший tuning нельзя вести по уже открытому
  canonical без риска overfitting validation protocol.

## Основные файлы и артефакты

- `data_utils.py` — raw split, daily aggregation и ground truth.
- `popularity.py` — global и recency generators, fallback.
- `item2item.py` — co-visitation fit и inference.
- `implicit_model.py` — confidence matrix, ALS fit и portable restore.
- `candidate_pipeline.py` — candidate union, provenance и cross-scores.
- `features.py` — history, trend, co-vis и ALS-derived features.
- `ranker_data.py` — labels и первый уровень negative sampling.
- `rankers.py` — CatBoost data loader, pointwise/LTR models и score-to-top-k.
- `pipeline.py` — full-fit building blocks и второй sampling stage.
- `scripts/run_full_fit_inference.py` — production orchestration Task11.
- `submission.py` — CSV writer/parser и round-trip validation.
- `artifacts/task11_full_fit_v1/model/model.cbm` — финальный ranker.
- `artifacts/task11_full_fit_v1/candidate_models/` — четыре full-history
  retrieval models.
- `artifacts/task11_full_fit_v1/feature_importance.parquet` — importance 201
  признака.
- `artifacts/task11_full_fit_v1/submission.csv` — отправленный prediction file.

Исходная команда production launcher:

```bash
./scripts/run_task11_full_fit.sh configs/task11_full_fit_v1.json
```

Сейчас она намеренно откажется перезаписывать существующий artifact. Для
полного повторного fit нужен новый `run_id`, новый output/work path и достаточно
свободного места; уже опубликованный artifact лучше считать immutable.
