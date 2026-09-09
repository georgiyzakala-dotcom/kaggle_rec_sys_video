# Задача 15: SASRec как источник кандидатов

Актуализация2026-09-07 по следующему запросу пользователя: порядок выполнения
изменён на **single GPU benchmark -> Optuna только rolling_1/next day ->
обучение выбранной модели на остальных rolling и анализ recall/ALS overlap**.
Архитектурные исправления из этого плана приняты. Текущий launcher и инструкции:
`SASREC_BENCHMARK.md`. Старое предложение selection на rolling_1/2 ниже
сохраняется как первоначальный вариант и не является текущим заданием.

Benchmark и Optuna выполнены пользователем. Frozen winner — trial12/epoch12,
objective800=-0.0008428261; новая модель пока не улучшила равный бюджет.
Готов launcher `./scripts/run_task15_sasrec_folds.sh`; актуальные результаты
и protocol полного сравнения описаны в `SASREC_FOLDS.md`.

Ниже исходный план и разбор реализации, 2026-09-07. Encoder/loader/benchmark,
Optuna и frozen fold runner реализованы и проверены; полный прогон фолдов
ожидает пользовательского запуска. Цель — измерить добавочный
recall позитивов для ранкера и решить, нужны ли одновременно ALS и SASRec.
Повышение качества остаётся гипотезой.

## 1. Что уже есть и с чем сравнивать

Изучены `AGENTS.md`, `ROADMAP.md`, `PROJECT_STATE.md`, `experiments/results.csv`,
интерфейсы, candidate/feature/metric pipeline, ALS и Task13 profiles;
проверены сохранённые конфигурации и метаданные Task05/06/07/13/14.
В таблице — метрики опубликованных Task06 artifacts, не новые эксперименты.

| Fold | Cutoff | ALS recall@200 | Union recall, до 800 | Mean union count |
|---|---|---:|---:|---:|
| rolling_1 | 2024-11-29 08:56:28 | 0.04469316 | 0.08142894 | 600.70 |
| rolling_2 | 2024-11-30 08:56:28 | 0.05449623 | 0.09321877 | 620.83 |
| rolling_3 | 2024-12-01 08:56:28 | 0.06633673 | 0.11216399 | 648.59 |
| canonical | 2024-12-02 08:56:28 | 0.06742603 | 0.10700246 | 632.82 |

`candidate_recall` в проекте — micro recall по уникальным `(user_id,item_id)`:
`sum(candidate_hits_u) / sum(relevant_items_u)`. Это не средний user recall.
Во всех folds сохранены все 200,152 target users, включая пользователей без
истории на ранних cutoffs. Например, ALS coverage rolling_1 = 0.94384268;
нельзя исключать этих пользователей только из оценки SASRec.

Текущие источники, каждый до 200 items:

- Global popularity: число positive daily rows; также независимый fallback.
- Recency popularity: positive daily rows с первым timestamp в последних 6h.
- Item2item: undirected co-vis, history_cap=10, seed_k=5, neighbor_k=100,
  time-distance weighting, raw normalization, seed decay=6h, event strength.
- ALS: CPU `implicit==0.7.3`, 128 factors, regularization=0.1, 15 iterations,
  seed=42, 8 threads, event-strength confidence без time decay.

ALS сначала сворачивает daily rows в user-item pairs. Выбранная confidence:

```text
repeat_signal = log1p(max(total_views - 1, 0))
event_signal = 1 + repeat_signal + is_long_watch + 2*is_like + 2*is_favorite
confidence = 1 + event_signal
```

Её рецепт выбран пятью последовательными stages на трёх rolling folds.
30,496.54s Task05 — время всего selection run, а не одного обучения ALS.
В Task06 canonical models и rolling_3 ALS восстановлены, остальные rolling
models обучены один раз. Все их model/candidate paths существуют.

Canonical standalone recall@200 остальных sources: global=0.02454935,
recency=0.02594171, item2item=0.03720911. Union содержит 137,407 из
1,284,148 positive pairs. ALS даёт 86,585 candidate hits, из них 58,113
отсутствуют у всех трёх других sources. Поэтому одного низкого пересечения
SASRec с ALS недостаточно для замены: нужны именно новые positive hits.

Лучший validation run — `artifacts/task13_history_profiles_v1`, `D_all`:

- `precision_at_20_all_targets = 0.005077640992845437`;
- `precision_at_20_labeled_users = 0.006911912728855518`;
- 20,326 final hits; candidate recall = 0.107002463890455;
- oracle P20 all/labeled = 0.03429693432990927 / 0.04668652574879622;
- CatBoost Logloss, 1,030 trees, 221 features, fresh train-only 32 borders,
  бюджет около 30m sampled rows, все candidate positives и IPW.

Task13 выбирал варианты на rolling_3 после train rolling_1+rolling_2;
canonical ranker refit использовал rolling_1+rolling_2+rolling_3.
201 исходный признак дополнен 20 скалярами ALS history profiles.
Команда исходного run:
`./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.
Её нельзя повторять с тем же run ID поверх опубликованного результата.

Task14 уже опубликован: `artifacts/task14_full_fit_v1/submission.csv` существует,
его SHA256 совпал с metrics. В старой сводке он ошибочно оставался ожидаемым.
Это full-history fit, будущих labels нет, обе P20 и recall равны null.
Task14 weights и features нельзя использовать для rolling validation.
Здесь не выполнялся повторный полный semantic verifier Task14 и не проверялся
leaderboard; проверка наличия/метаданных/SHA не подменяет эти действия.

## 2. Оценка предложенного кода

Это подходящий небольшой SASRec-like encoder для первого опыта. Его сильные
стороны: causal attention, positional embeddings, pre-LayerNorm residual
blocks, dropout, компактные defaults 64/2/2/50 и возможность tied embeddings.
Представление последней позиции можно вычислить один раз на пользователя и
использовать как retrieval query. Это добавляет порядок и недавний контекст,
которые ALS напрямую не моделирует. Насколько это поможет на короткой истории
и большом разреженном каталоге, определят folds.

Код отличается от точного воспроизведения оригинального SASRec: GELU и FFN 4d,
нормализация Q/K/V, отдельная output projection по умолчанию. Это допустимые
варианты; сами по себе отличия не являются доказательством улучшения.
Оригинальная реализация использует shared item table, маскирование padded
states, финальную normalization и обучение на positive/negative logits.
[Код авторов](https://github.com/kang205/SASRec/blob/master/model.py).

| Приоритет | Наблюдение | Изменение для первой версии |
|---|---|---|
| Обязательно | Left padding + causal/key mask оставляют padded queries без разрешённых keys; второй блок получает NaN и на valid positions | Right padding, исключение empty sequences из encoder, явные lengths и gather последней valid position; zero padded states после embedding и каждого блока |
| Обязательно | `x[:, -1]` верен лишь если последний tensor slot действительно последний item | `last_index = lengths - 1`; gather только для lengths>0 |
| Обязательно | Xavier после создания Embedding перезаписывает нулевую padding row | Обнулить row 0 после init; PAD исключён из targets, negatives и retrieval; проверить нулевые gradient/state padding row |
| Обязательно | Полные logits `[batch,length,num_items]` не масштабируются | Отдельные encode / score sampled items / score pairs / chunked top-k APIs |
| Обязательно | Loss, shifted labels, negative sampler, seen filtering и trainer отсутствуют | Реализовать и тестировать их явно; по encoder нельзя оценить готовность всей модели |
| Желательно | `need_weights=True` вычисляет неиспользуемые attention weights | Передавать `need_weights=False`, проверить causal equivalence |
| Желательно | У pre-norm stack нет финального LayerNorm | Добавить `final_norm` перед scoring; зафиксировать в initial recipe |
| Желательно | Отдельный `fc` почти удваивает item parameters | Default `use_output_projection=False`, tied dot-product |
| Желательно | `use_item_normalization` нормализует item+position input, не output item vectors | Переименовать в `use_input_layer_norm`; не трактовать как cosine retrieval |
| Желательно | Нет явной проверки длины, shapes, range IDs, empty inputs | Проверять L<=max_length, integer indices, bool mask, согласованность mask с PAD |
| Желательно | `__setstate__` вручную заменяет штатную логику nn.Module | Portable `state_dict` + JSON config/version/mappings; миграция config при restore |

Проблема NaN воспроизведена на установленном PyTorch 2.13.0+cu130, CPU,
двух блоках из присланного кода, seed=42, dropout=0:

```text
input [0,0,1,2]: finite after block 1 = [false,false,true,true]
                finite after block 2 = [false,false,false,false]
input [1,2,0,0]: finite after block 2 = [true,true,true,true]
input [0,0,0,0]: finite after block 2 = [false,false,false,false]
padding_embedding_norm after Xavier = 0.78663176 (tiny test)
```

В right-padded случае padded outputs всё равно не равны последнему valid
output. Исправление только `need_weights=False` не заменяет контракт padding.
Не лечить NaN умножением на ноль: `NaN * 0` остаётся NaN. Empty histories
направляются в fallback отдельно. Финальные scores должны быть finite.

Инверсия `~causal_mask` в вашем коде корректна для boolean masks MHA:
там True запрещает attention. При переходе на прямой SDPA нужно проверить
его отдельную семантику масок. `padding_idx` относится к embedding lookup,
а не ко всем возможным использованиям той же weight matrix в output head.
Для AMP заменить магическую константу `-1e9` на явное исключение PAD или
dtype-aware masking; проверять loss на полностью замаскированных targets.
[MHA API](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html),
[Embedding API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding).

Xavier у огромной item table даёт scale, зависящий от размера каталога:
`item_std = sqrt(2 / (num_items + embedding_dim))`. При 1.81m items и d=64
это около 0.00105; у position table 50x64 — около 0.132. Позиции могут
доминировать на входе. Предлагается явная normal initialization std=0.02
для обеих tables с PAD=0, мониторинг норм. Это стартовая настройка, не
валидированное улучшение. Input normalization и sqrt(d) scaling не включать
сразу вместе без отдельного сравнения.

Отдельный `fc` математически тоже допускает retrieval:
`score(u,i) = dot(query_u, fc.weight[i]) + fc.bias[i]`.
Тогда для поиска нужны output vectors и bias, а не input embedding table.
В первой версии tied head проще, меньше и согласован с pair scoring.

## 3. Данные и обучение

Использовать существующие immutable folds:

- rolling_1: `artifacts/task02_fold_20241129_v1`;
- rolling_2: `artifacts/task02_fold_20241130_v1`;
- rolling_3: `artifacts/task02_fold_20241201_v1`;
- canonical: `artifacts/task01_canonical_data_v1`.

В каждом случае модель читает только `history_daily.parquet`. Fit — на всех
history users, inference — на полном target universe. Dense item mapping
строится по всему history catalog, PAD=0, реальные items=1..N; исходные
UInt64 user IDs остаются integer при joins и сериализации. Отдельно считать
items с нулём positive targets: известность item не означает обученность
его embedding. Не вводить частотный vocabulary cutoff без отчёта о потерянном
GT и верхней границе recall.

Последовательность — daily rows в порядке `(user_id,dt,item_id)`; item_id
разрешает timestamp ties детерминированно. `dt` — первый timestamp агрегата,
flags — максимум за день. Поэтому это порядок daily interactions, а не
точная последовательность raw событий. Повторы item между днями сохраняются
в контексте; нельзя молча превратить её в уникальный набор.

Начальный вариант `positive_only`: выбрать positive daily rows, обучать
shifted next-positive prediction: `input=items[:-1]`, `target=items[1:]`.
У каждого valid query собственный next target, padding positions из loss
исключены. Диагностика цели `new_item_only`: loss mask выключает targets,
которые уже встречались до query в полной доступной истории пользователя,
включая weak events и prefix до обрезки L. Сравнить её с обычной shifted loss
как отдельную семантическую ablation; в базовом recipe сначала оставить
обычную loss, чтобы не смешать сразу архитектуру и новую постановку.

Второй вариант — весь daily context и prediction следующего positive item
после query. Targets нужно строить явным forward lookup по той же истории:
это не loss на всех watch events. При нескольких queries с одной целью
контролировать их вес, иначе длинный weak-event gap многократно повторит
один positive. Same-target weight = 1 / число его supervised queries.
Признаки событий и time-gap embeddings оставить для последующей ablation.

Для batching подготовить flat integer arrays + user offsets/lengths или
memory-mapped shards; не хранить миллионы Python lists/windows. На первом
этапе один seeded sampled window длины до L+1 на eligible user за epoch,
endpoint меняется между epochs; хранить число фактически увиденных targets.
Последний history window используется на inference. Если после профиля видна
сильная недоэкспозиция длинных историй, отдельно сравнить до 4 windows/user,
сопоставляя число optimizer steps и supervised targets, а не только epochs.

Initial recipe (все численные параметры — предложения для проверки):

| Параметр | Начальное значение |
|---|---|
| embedding_dim / heads / blocks | 64 / 2 / 2 |
| max_length / dropout | 50 / 0.1 |
| head / input layer norm | tied dot-product / disabled |
| optimizer / learning_rate | AdamW / 0.001 |
| weight_decay / gradient_clip_norm | 0.0001 / 1.0 |
| train microbatch | 128, увеличить до 256 только по resource probe |
| negatives per valid target | 64 |
| epoch upper bound / checkpoints | 20 / epochs 2,5,10,20 |
| precision | BF16 autocast при поддержке GPU; FP32 loss reduction |
| seed | 42 |

Начать с sampled BCE, без полного softmax:

```text
positive_score = dot(query, item_embedding[positive_id])
negative_score_j = dot(query, item_embedding[negative_id_j])
loss_per_target = softplus(-positive_score) + mean_j(softplus(negative_score_j))
loss = weighted_sum(valid_target_losses) / sum(valid_target_weights)
```

Такой objective — sampled binary discrimination, не вероятность softmax по
всему каталогу. Uniform negatives выбираются из history catalog своего fold;
исключаются PAD и все items пользователя, наблюдавшиеся в этом fit history.
Это использует только outer history, не validation positives. Как training
convention это не имитация online sampler внутри каждого исторического prefix.
Хранить sampler config и RNG state; все-positive/no-negative примеры пропускать
с явным счётчиком. Не называть sampled-negative Recall реальным recall каталога.

Loss scoring делать только для valid positions и небольшими блоками queries:
даже tensor `[B,L,negatives,d]` может занять сотни MiB. Dense embedding gradients
и optimizer moments также входят в memory budget. Popularity negatives,
in-batch negatives, sampled softmax/log-Q и hard negatives — последующие
отдельные ablations с контролем false negatives; ALS candidates нельзя брать
из другого fold. Не переносить автоматически CatBoost class weights/IPW в
этот objective.

Основной разрыв задачи: SASRec учится следующему item, а GT содержит набор
новых позитивных items за 24h. Начать с one-query retrieval последнего state;
если прироста нет, заранее определённая следующая гипотеза — multi-horizon
positive targets или pooling нескольких recent states. Не запускать их
автоматически и не подбирать после просмотра canonical.

## 4. Rolling protocol и ограниченный поиск

1. Preflight и bounded smoke: CPU numerical tests, затем GPU forward/backward
   и retrieval на малом sample. GPU throughput измерить на full vocabulary
   отдельно от quality smoke. Ни один smoke не становится full experiment.
2. Baseline SASRec на rolling_1/rolling_2. Для epoch checkpoints — фиксированная
   выборка 8,192 target IDs, выбранная без labels, с полным eligible catalog и
   полными top-k lists. Выбрать checkpoint по mean incremental union recall;
   full-users метрики finalists посчитать отдельно. Validation только читает
   evaluator: gradients, sampler, mappings и fallback используют history.
3. Ограничить первый selection максимум четырьмя recipes: baseline;
   winner + L=100; winner + all-context/next-positive; winner + new-item loss
   mask. Каждый stage меняет одну ось. D=128, другие losses и ANN не входят
   в обязательную сетку. Recipe и epoch budget фиксируются до rolling_3.
4. Finalists оценить на всех target users rolling_1/rolling_2. Выбрать
   source-budget policy из заранее определённой таблицы ниже. Train на каждом
   fold с нуля; более поздние weights на ранний fold не переносить.
5. Один frozen recipe и policy обучить на rolling_3 history с фиксированным
   числом epochs, без early stopping по rolling_3 labels. Это проверка переноса
   для новой SASRec. Если не подтверждается, результат отрицательный/неустойчивый;
   дальнейший поиск оформляется отдельно.
6. После freeze сделать один canonical fit recipe и оценку заранее заданных
   ALS/SASRec/union counterfactuals. Сохранить canonical_previously_opened=true;
   этот день уже много раз использовался в проекте и не является новым blind
   test. Не выбирать новый checkpoint, cap или recipe по canonical.

Rolling_1/2 — selection estimates. Rolling_3 тоже исторически участвовал в
выборе существующих моделей; указать это ограничение в сравнении. Все старые
источники используют уже выбранные recipes, которые не переселекционируются.
История всего около шести суток: rolling scores не доказывают устойчивость
на других неделях. Если эффект мал, ограниченный repeat winner/control с
seed=43 назначить отдельно до окончательного решения; сравнивать одинаковые
seed для действительно переобучаемых компонентов, reuse ALS явно отмечать.

## 5. Recall, пересечение с ALS и равный бюджет

Обозначения для пользователя u: `A_u` — ALS candidates, `S_u` — SASRec,
`O_u` — union global+recency+item2item, `G_u` — eligible GT,
`B_u = O_u union A_u` — текущий baseline.

Для каждого source и union сохранять recall@50/100/150/200, coverage,
mean/p50/p90/p95/p99 counts, candidate_user_hit_rate, обе oracle P20;
standalone top20 с независимым fallback и обе обычные P20. Native coverage
считать до fallback, его items не приписывать SASRec. Сохранять counts
исключённых seen/cold GT из split manifest и проверять неизменность universe.

Для ALS/SASRec отдельно считать:

```text
intersection_u = count(A_u intersect S_u)
jaccard_u = intersection_u / count(A_u union S_u)
share_of_sasrec_in_als_u = intersection_u / count(S_u)
share_of_als_in_sasrec_u = intersection_u / count(A_u)

shared_positive_hits = sum_u count(A_u intersect S_u intersect G_u)
sasrec_only_vs_als = sum_u count((S_u minus A_u) intersect G_u)
als_only_vs_sasrec = sum_u count((A_u minus S_u) intersect G_u)
new_hits_vs_all_sources = sum_u count((S_u minus B_u) intersect G_u)
als_still_needed_hits = sum_u count((A_u minus (O_u union S_u)) intersect G_u)
delta_recall_add = new_hits_vs_all_sources / sum_u count(G_u)
```

У overlap fractions с нулевым знаменателем сохранять null и число таких
users; mean/quantiles отдельно для всех определённых cases и users с обоими
непустыми sources. Дополнительно micro intersection/union по всем парам.
Для positive overlaps указывать counts и доли относительно hits каждого
source. Матрицы считать со всеми четырьмя sources, а не только ALS.

Обязательные сравнения используют существующие top200; переобучать ALS не нужно:

| Policy | Candidates | Сумма source caps |
|---|---|---:|
| baseline | O(3x200) + ALS200 | 800 |
| replace_als | O(3x200) + SASRec200 | 800 |
| blend_150_50 | O(3x200) + ALS150 + SASRec50 | 800 |
| blend_100_100 | O(3x200) + ALS100 + SASRec100 | 800 |
| blend_50_150 | O(3x200) + ALS50 + SASRec150 | 800 |
| add_source | O(3x200) + ALS200 + SASRec200 | 1000 |

Это одинаковые верхние source budgets, но число уникальных candidates после
dedup может различаться. Его и фактическое ranker время показывать рядом.
`CandidateUnionConfig` требует sum(caps)<=total_cap: добавить 5x200 и просто
оставить total_cap=800 нельзя. Существующий контракт не допускает молчаливое
post-union pruning. Ранкер Task13 использует materialized 200/800, а не
выбранные для исторического RRF 150/600.

Если выигрыш есть только у add_source, обязателен дополнительный контроль
`O600+ALS400` против `O600+ALS200+SASRec200`, при необходимости `O600+SASRec400`.
Для ALS400 восстановить ту же fold model и сделать новый inference в Task15
artifact, не fit; его стоимость заранее измерить. До этого нельзя заключать,
что новая модель выгоднее простого углубления ALS. При разных unique counts
показать frontier recall/oracle vs фактическое число rows и latency; если нужен
строго одинаковый unique cap, реализовать отдельную явную deterministic policy,
выбранную только на rolling_1/2, а не менять нынешний union контракт.

Primary selection — mean по rolling_1/2 `delta candidate_recall` при budget800;
tie-break: minimum fold delta, mean delta oracle P20 all, стабильный config
order. Метрики усреднять по folds явно; не подменять mean суммарным micro recall.
Для дополнительного бюджета отдельно оценивать incremental hits per 100 added
unique candidates. Proposed practical gate: mean delta recall >=0.001
(0.1 percentage point), положительный oracle gain и отсутствие fold regression;
это заранее предложенный порог, а не доказанный предел значимости.

Сегменты только по history: длина positive/all history, давность активности,
число уникальных items, like/favorite activity, history item popularity,
empty/short histories. Для каждого — counts, source coverage, new/lost hits,
recall и обе oracle P20. Paired user bootstrap seed42 даёт интервалы для
delta recall/oracle/P20; при micro recall заново считать отношение сумм в каждой
реплике. Интервалы по users не заменяют проверку другого дня и seed.

## 6. Retrieval и интеграция

Сохранить существующие Model/DataLoader contracts, добавить:

- `sasrec_data.py`: `SASRecDataLoader(CandidateDataLoader)`, mappings, sequences,
  lengths, shifted targets, sampler, полный seen CSR. Тяжёлая подготовка в
  prepare hooks, iter hooks возвращают готовые batches; epoch sampling получает
  явный seed/epoch и не читает заново raw data.
- `sasrec_model.py`: `SASRecEncoder(nn.Module)` и
  `SASRecCandidateModel(CandidateModel)`, `source_name='sasrec'`, fit/predict,
  `encode_users`, `score_pairs`, `save/from_artifact`. Декоратор `map_model`
  внешнего проекта здесь не требуется.
- `sasrec_experiment.py`: selection, reference loading, policy metrics,
  overlap/unique-positive analysis и atomic resume state.
- `scripts/run_sasrec.py`, `scripts/run_task15_sasrec.sh`, full/smoke JSON
  configs и `research/15_sasrec_candidates.ipynb` для чтения результатов.

Repository сейчас хранит reusable modules в корне; следовать этому соглашению
без побочной миграции всех старых модулей в src.

Выход predict строго `CANDIDATE_SCHEMA`: user_id UInt64, item_id Int32,
score Float64, rank UInt32, source String. Модель считает float32 scores,
а wrapper приводит их к schema. Tie-break score DESC/item_id ASC, PAD excluded.
В seen filtering используется вся fold history, не последние L items и
не только positives. Для unavailable sequence source выдаёт пустой список;
downstream global fallback обеспечивает final20.

Первый retrieval — exact blockwise matrix multiplication, users batch=128,
item chunk=32768 как старт probe. Сначала замаскировать все seen внутри каждого
chunk, затем incremental deterministic top-k. Хранить только текущий top-k,
а не user-by-catalog matrix. При ничьей на границе torch.topk недостаточно:
нужно явно разрешить tie по исходному item_id, проверить на synthetic равных
scores. User state вычислить один раз и сохранить для pair scoring.

Exact retrieval по 200k users и 1.81m items всё равно вычисляет около 362 млрд
scores. Chunking решает память, не объём вычислений. Если full-catalog probe
даёт неприемлемое время, отдельно добавить MIPS ANN и проверить top200 recovery
относительно exact на ID-only sample, например целевой recall ANN >=0.99.
Измерять и ANN recall, и реальный positive recall после seen filtering;
адаптивно расширять поиск/использовать exact fallback. Не применять cosine
нормализацию к dot-product index без изменения model contract.

Task06 source tables и fold-local models переиспользовать по manifests с
SHA checks. Rolling_3 ALS хранится по внешнему pointer в
`artifacts/.task05_implicit_als_v1.best-model/versions/.../model` — не удалять.
Task15 создаёт новые source tables/union artifacts с явными reference hashes.

`build_candidate_union` уже принимает произвольные sources. Но source dispatch
в `scripts/prepare_candidate_datasets.py` перечисляет четыре модели явно;
добавление класса само по себе pipeline не расширит. Через новый Task15 runner
переиспользовать старые outputs, затем построить новый union и добавить:
`generated_by_sasrec`, `generator_score_sasrec`, `generator_rank_sasrec`,
`generator_rank_norm_sasrec`, `cross_score_sasrec`,
`cross_score_available_sasrec`, per-user cross-score rank.

SASRec cross-score нужен на каждой union pair, в том числе найденной ALS;
скалярная оценка не означает `generated_by_sasrec=true`. Для новых SASRec pairs
получить cross-scores всех старых моделей. History lookups Task07 можно reuse;
новые pair features, source_count, normalized ranks, union ranks, labels и
negative sampling нужно пересчитать. Старые Task07 shards/quantized pools
нельзя просто расширить столбцами: для новых пар там нет строк. Новые schemas,
manifests, feature order и train-only borders версионируются отдельно.

## 7. Ранкер и решение о судьбе ALS

Сначала завершить candidate-only сравнение. Для перспективного frozen recipe
подготовить matched ranker backtest с Task13 настройками, budget30m/all positives/
IPW и одинаковой политикой sampling. Selection train rolling_1+rolling_2,
evaluation rolling_3; canonical refit train rolling_1+rolling_2+rolling_3.
Каждый historical SASRec source обучен только на history своего fold.

Минимальные arms: текущий union с ALS; замена ALS-кандидатов SASRec; лучший
совместный budget800 union. Add_source включить, если его resource frontier
выгоднее. Для arms пересчитать candidate-dependent features и sample из
своего universe одним seed и правилами; сравнить positive/negative counts,
не объявлять разные фактические pools одинаковыми. Сохранить обе P20,
recall/oracle, final hits, runtime, RAM/VRAM, final SASRec-exclusive hits.
Это attribution по provenance, а причинный эффект удаления source измеряется
разностью полного counterfactual arm.

Если нужно отделить пользу score от новых candidates, дополнительный arm:
прежний union + SASRec cross-score features, без новых candidates. Старый
CatBoost не обучен этим features; необходим новый matched fit, а не загрузка
новых колонок в старую модель. Такой arm также требует SASRec на inference.

ALS сейчас выполняет три роли: generator, cross-scorer, источник 20 history
profile scalars плюс ALS norm/cosine features. Поэтому:

1. Удаление только ALS generator сохраняет ALS обучение/factors ради ранкера.
   Это чистое сравнение candidate sets, но не экономия всего ALS pipeline.
2. Полный отказ от ALS требует отдельного ranker без всех ALS-derived features,
   cross-scores и profiles. Сравнить его заново; убрать artifacts заранее нельзя.
3. Замена ALS profiles на SASRec embedding profiles — новая feature гипотеза,
   не автоматическое следствие большего recall.

Оставить обе candidate models, если их совместный budget даёт устойчивый
recall/oracle gain на rolling и matched ranker не теряет P20 при приемлемой
стоимости. Выбрать одну, если другая не даёт достаточных unique positives
после остальных sources и сопоставимого бюджета. Более высокий standalone
recall сам по себе не выбирает лучший компонент ансамбля. При неопределённом
эффекте сохранить текущий ALS baseline и записать отрицательный результат.
Решение и tolerances зафиксировать до canonical; canonical показать отдельно.

## 8. PyTorch, ресурсы и runner

Рекомендация — обычный PyTorch. В `.venv` уже есть `torch==2.13.0+cu130`;
в `requirements.txt` он пока не записан. Lightning и FAISS не установлены.
При реализации закрепить проверенную torch dependency и CUDA-wheel provenance,
не менять рабочее окружение автоматически. Нужен GPU training smoke именно
PyTorch: успешный CatBoost GPU fit не проверяет Torch CUDA kernels.

Репозиторий уже содержит `experiment_utils.py`, atomic checkpoints, progress,
rotating logs и resource supervisor (`scripts/task13_resources.py`). Их
переиспользование с маленьким явным train loop проще интегрирует outer folds,
retrieval и union metrics. Lightning умеет управлять training loop, precision,
callbacks и checkpointing, но не решит shared snapshots, каталог, seen filtering
и наши experiment manifests. Рассмотреть его при DDP/multi-GPU или нескольких
семействах torch models с общим trainer.
[Lightning Trainer](https://api.lightning.ai/docs/pytorch/stable/common/trainer.html).

По прежним runs машина имеет RTX 5070 Ti около 16GiB VRAM; в текущей sandbox
NVML заблокирован, GPU throughput и текущая свободная VRAM не проверены.
CPU probe: Python3.12.3, RAM около49GiB, available44GiB, Linux free650GiB.
Свободное место на Windows host G: перед запуском проверить отдельно.

При canonical catalog 1,810,005 items, d=64:

- одна item table FP32: 0.432GiB;
- её weights+gradient+два Adam moments FP32: 1.726GiB;
- отдельный fc добавляет почти столько же optimizer state;
- полный `[256,50,N+1]` logits tensor: 86.308GiB FP32 или 43.154GiB BF16,
  ещё без backward; такой путь запрещён для full run;
- score block 128x32768: 16MiB FP32, без временных top-k buffers.

Это оценки компонентов, не измеренный peak. Стартовые лимиты: CPU8/BLAS1,
RSS40GiB, availableRAM stop4GiB, один GPU worker, reserveVRAM>=2GiB.
Record torch allocated/reserved peak и внешние RSS/device metrics; лимиты и
batch sizes проверить probe. Прежний supervisor адаптировать к PyTorch,
не считать CatBoost `gpu_ram_part` лимитом Torch.

Надёжная оценка времени возможна только после измерения preparation,
train steps/s, exact retrieval users/s, metric joins и disk bytes/shard.
Передавать пользователю рассчитанную оценку по числу configs/folds/checkpoints
и отдельный worst-case disk estimate. Не обещать время по размеру encoder:
retrieval и feature regeneration могут оказаться дороже обучения.

Runner: run/stage/config/fold/epoch/batch progress + elapsed/ETA; ограниченный
plain-text rotating log с timestamp, stage/config/fold/operation, current/best
metrics; CPU limits, `flock`, PID, TERM forwarding. Atomic checkpoints содержат
model/optimizer/scheduler/scaler, RNG Python/NumPy/Torch CPU/CUDA, epoch/global
step, window sampler state, configs и input hashes. Минимальный exact resume —
на границе epoch; после interruption повторяется только незавершённый epoch
из его начального RNG state. Если нужен mid-epoch resume, добавить cursor и
test, а не обещать его при обычном DataLoader prefetch.

После каждого `(stage,config,fold)` и retrieval shard сохранять completed
operation, checksum и timing; portable best weights отдельно от optimizer.
Atomic publish, отказ overwrite, bounded cache retention, verify-only и
resume без повторения completed work обязательны.

Планируемые команды (пока не существуют и сейчас не запускались):

```bash
./scripts/run_task15_sasrec.sh configs/task15_sasrec_v1.json
tail -F logs/task15_sasrec_v1.log
kill -TERM "$(cat logs/task15_sasrec_v1.pid)"
.venv/bin/python scripts/run_sasrec.py --verify-only artifacts/task15_sasrec_v1
```

Повтор launcher возобновляет незавершённый run; completed ID не перезаписывается.
Outputs: `artifacts/task15_sasrec_v1/{config.json,metrics.json,selection/,
folds/,overlap/,models/,report.md}`; в folds — narrow candidate shards, per-user
hit counts и manifests. Ranker extensions — отдельный artifact/run ID.
Full-history fit нового рецепта и submission — следующий этап после решения
по метрикам; внешний запуск выполняется отдельно, Kaggle upload не входит.

## 9. Критерии готовности

- Tests: causal prefix invariance, right padding/last valid, empty histories,
  finite forward/backward, PAD gradient=0, shifted-target/no-self leakage,
  repeat/new-item semantics, seeded sampling, oversized UInt64 IDs, fold-local
  mappings, seen filtering по полной истории, deterministic ties и unknown items.
- Retrieval: tiny brute-force vs chunked top-k, score_pairs совпадает с
  retrieval scores, saved model restore, AMP finite loss. Если ANN используется,
  отдельная exact-vs-ANN quality проверка.
- Metrics: synthetic relevance с watch_time=60/61, seen/cold exclusion,
  dedup, empty GT, обе P20, micro recall, overlap/unique hits и budget policies.
- Pipeline: новый source provenance/cross-score availability, immutable old
  artifacts, no validation access in fit/sampler; canonical isolation до freeze.
- Runner: `--help`, `bash -n`, unit suite, limited end-to-end smoke, TERM/resume,
  independent same-seed CPU repeat и portable deterministic inference. Для GPU
  задокументировать numerical tolerance/variation и версии backend; не обещать
  bit-exact training между разными CUDA environments.
- На каждом fold: полные source/union metrics, обе standalone/final P20,
  overlap с ALS, new/lost positives, matched budget и resource comparisons.
  Каждый final20 output проходит exact target set, 20 unique known unseen items,
  no nulls. Candidate lists до k могут быть короче — это отражается в coverage.
- Freeze winner/policy и conclusion keep_both/als_only/sasrec_generator_only/
  sasrec_without_als обоснованы метриками и перечисляют зависимости ранкера.
- Notebook читает готовые artifacts. `experiments/results.csv` дополняется
  только meaningful model runs; smoke/planning review туда не добавляются.
  `PROJECT_STATE.md` содержит фактические команды, best result и ограничения.

Текущий review проверен без обучения модели. Воспроизведение numerical probe
и сводки сохранённых metadata:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python artifacts/task15_sasrec_review_v1/review.py
```

Результат: `artifacts/task15_sasrec_review_v1/metrics.json`; все reference
source/model paths доступны, Task14 CSV hash совпал. Полный пересчёт recall
из 500m ranker rows в рамках планирования не выполнялся. Численного
пересечения SASRec/ALS пока нет: SASRec weights/candidates ещё не построены.

Два независимых запуска CPU review дали идентичный JSON. Дополнительно прошли
20 существующих tests (5 metric, 9 candidate pipeline, 6 data preparation):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_metrics.py' -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_candidate_pipeline.py' -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_data_utils.py' -v
git diff --check
```
