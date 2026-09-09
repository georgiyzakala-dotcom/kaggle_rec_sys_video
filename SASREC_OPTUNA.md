# Task15: Optuna на rolling_1

Обновление2026-09-07: пользовательский search завершён, все24trials обработаны.
Победитель trial12/epoch12, objective -0.0008428261012708299; равный budget800
пока не улучшен. Artifact прошёл verifier. Результаты и следующий launcher
полного frozen сравнения описаны в [SASREC_FOLDS.md](SASREC_FOLDS.md):
`./scripts/run_task15_sasrec_folds.sh`.

Назначение этого этапа — подобрать ограниченный набор гиперпараметров SASRec,
используя positives следующего дня только для оценки. Подбор завершён;
следующий этап — отдельное обучение и сравнение на остальных rolling.

## История восстановления после ошибки pruning (до завершения search)

Исправлена ошибка runner: `study.tell` теперь передаёт objective только для
COMPLETE. Для PRUNED и FAIL значения не передаются; Optuna берёт оценку PRUNED
из последнего `trial.report`. Обучение, sampler и правила оценки не менялись.

Пользовательская study восстановлена без обучения: 10 trials COMPLETE,
trial №10 PRUNED, следующий №11 (нумерация с нуля). Продолжение:

```bash
./scripts/run_task15_sasrec_optuna.sh
```

Оставить прежние run_id и config. Сохранены результаты, веса, best checkpoint
и накопленные 9313.956607696004s активного времени. Контрольные суммы 136 файлов
совпали до/после восстановления. SQLite snapshots и отчёт находятся в
`artifacts/task15_optuna_pruning_fix_v1/`; `metrics.json` подтверждает отсутствие fit.

Resume допускает только явно проверенную пару старого/нового SHA256 runner из
`configs/task15_sasrec_optuna_resume_compatibility.json`. Остальные настройки и
hashes должны совпадать. Исходный frozen config и его digest сохранены;
`runtime_patch.json` фиксирует исправление и включается в final artifact.
Это не разрешает произвольные изменения кода или параметров посреди поиска.

Проверены 48 tests, включая принудительный pruning, сбой после сохранения result,
resume без повторного fit и защиту от несовместимых изменений config/code.
Проверки CLI: `./scripts/run_task15_sasrec_optuna.sh --help` и
`bash -n scripts/run_task15_sasrec_optuna.sh`. Полный поиск в сессии не возобновлялся.

## Результат пользовательского benchmark

`artifacts/task15_sasrec_benchmark_v1` прошёл повторный portable verifier.
Полное обучение346,947eligible users, catalog1,337,354items:

| Измерение | Значение |
|---|---:|
| Весь запуск,3epochs | 164.83s |
| Training каждой epoch | 42.37–43.28s |
| Подготовка windows/negatives каждой epoch | 7.20–7.34s |
| Median epoch с подготовкой | 49.77s |
| Steady training step, batch128 | 15.4ms |
| Пик RSS | 2.89GiB |
| PyTorch peak allocated/reserved | 1.66/1.84GiB |
| Внешний sampled peak device memory | 3700MiB |
| Пробная выдача128users по полному каталогу | 0.604s |

Loss снизился0.902 ->0.593 ->0.477. Benchmark не читал labels; из его loss
нельзя заключить, что recall улучшился. Обе Precision@20 и recall там null.

## Запуск поиска

```bash
./scripts/run_task15_sasrec_optuna.sh
```

Та же команда с явной конфигурацией:

```bash
./scripts/run_task15_sasrec_optuna.sh configs/task15_sasrec_optuna_v1.json
```

Поиск выполняет trials последовательно на GPU, без Lightning. Установлена и
закреплена в `requirements.txt` необходимая зависимость `optuna==4.9.0`.
Torch/NumPy/Polars и остальные существующие зависимости не обновлялись.
Launcher ничего не устанавливает и не переключается молча на CPU.

1. Проверяются ресурсы, hashes фолда и четырёх готовых candidate sources.
2. Переиспользуется проверенный sequence cache benchmark. При его отсутствии
   cache строится из immutable daily history в новом work directory.
3. Один раз выбираются8192target IDs по hash(seed42), без фильтра по labels,
   history length или наличию SASRec/ALS candidates. Кэшируются их labels и
   четыре прежних source lists. Пользователи без labels/истории остаются в оценке.
4. Первый trial — исходный recipe benchmark, обучаемый заново; остальные
   параметры предлагает TPE. Каждый trial начинает обучение с seed42.
5. После каждой второй epoch — exact top200 по полному history catalog,
   seen filtering, recall/overlap и решение о pruning/early stopping.
6. Публикуются лучший recipe/epoch/policy, portable model, SQLite study,
   таблица trials, candidates и метрики выбранного checkpoint.

В терминале есть progress/ETA по trials, epochs, подготовке windows, training
batches и validation batches. Plain rotating logs отдельно сохраняют параметры,
loss, текущий/лучший objective, причины остановки, длительности и ресурсы.

## Что оптимизируется

History: `artifacts/task02_fold_20241129_v1/history_daily.parquet`.
Целевой период: `[2024-11-29 08:56:28, 2024-11-30 08:56:28)`.
GT берётся из `target_ground_truth.parquet` того же фолда: relevance,
dedup, seen и cold filtering уже выполнены общим data pipeline. Его funnel
и checksums сохраняются. Ни validation events, ни labels не попадают в
последовательности, item vocabulary, negatives, gradients или fallback.

Фиксированные источники: `artifacts/task06_candidate_datasets_v1/folds/rolling_1/sources`.
ALS, global/recency popularity и item2item не переобучаются. Все до200/source.
Обозначим три источника кроме ALS как `other_sources`, их суммарный cap600.

```text
recall = count(unique candidate pairs intersect relevant pairs)
         / count(relevant pairs)
objective = best_union_recall_at_source_budget_800 - baseline_union_recall
```

Это micro recall по парам, как в существующем проекте. Для каждого checkpoint
оцениваются заранее заданные варианты:

| Policy | ALS cap | SASRec cap | Общий source budget |
|---|---:|---:|---:|
| baseline | 200 | 0 | 800 |
| replace_als | 0 | 200 | 800 |
| blend_150_50 | 150 | 50 | 800 |
| blend_100_100 | 100 | 100 | 800 |
| blend_50_150 | 50 | 150 | 800 |

Baseline не является вариантом выбора новой модели. Objective может быть
отрицательным. В этом случае лучший SASRec recipe всё равно сохраняется, но
улучшение над baseline не заявляется. При равном recall policy выбирается по
oracle P20all, затем по порядку таблицы. Между одинаковыми objective epochs
сохраняется более ранняя; между trials — меньший номер.

Количество уникальных candidates после dedup может различаться даже при
одинаковых source caps; coverage и mean/quantiles counts сохраняются рядом.
Добавление200SASRec сверх baseline, budget1000, считается только диагностикой.
Оно не влияет на objective. Контроль ALS400 и окончательное решение о составе
источников относятся к следующему этапу.

Каждая оценка сохраняет:

- Native SASRec recall@200, coverage/counts, candidate user hit rate, обе oracle P20.
- Recall/oracle/counts для каждой policy и её delta к тому же sample baseline.
- Raw и positive overlap SASRec со всеми четырьмя sources: intersection,
  Jaccard, доли пересечения, unique positive hits с обеих сторон.
  Доли с нулевым знаменателем — null; указаны counts определённых/неопределённых users.
- Новые positive hits сверх всех прежних sources.
- Обе обычные Precision@20 для SASRec top20 с независимым global fallback;
  это диагностическая выдача без ranker. Native coverage считается до fallback.

Выбранные на этом sample метрики являются selection estimates. Это не оценка
на всех200,152users и не новый canonical результат. После поиска recipe и
epoch фиксируются; остальные rolling/canonical не участвуют в Optuna.

## Пространство и бюджет

| Параметр | Значения |
|---|---|
| embedding_dim | 32,64,128 |
| max_length | 25,50,100 |
| num_blocks | 1,2 |
| dropout | 0,0.1,0.2,0.3 |
| negative_count | 32,64 |
| learning_rate | log-uniform от0.0001 до0.003 |
| weight_decay | log-uniform от0.000001 до0.001 |

Фиксированы2heads, batch128, sampled BCE, tied embeddings, AdamW, BF16,
gradient clipping1, initialization std0.02. Все eligible history users
участвуют в каждой epoch; одна epoch — один случайный window на пользователя.
Benchmark weights не используются как warm start: меняется архитектура, а
сравнение trials должно начинаться с одинакового training protocol.

Максимум24trials по12epochs, evaluation каждые2epochs. MedianPruner включается
после6завершённых trials и не раньше epoch4. Ему передаётся лучший objective
trial на текущем epoch budget. Early stopping —3оценки подряд без улучшения.
Общий soft limit —8часов активного времени: новые trials после него не
начинаются, текущий завершается на ближайшей границе epoch с оценкой.
Поэтому итог может содержать меньше24trials и немного превысить8часов.
Это ограниченный поиск, не гарантия глобально оптимальных параметров.

По benchmark исходный recipe:12epochs обучения/подготовки около10минут;
шесть retrieval на8192users — ориентировочно ещё4минуты. Расчёт union/overlap,
checkpoints и d128/L100 добавят время; pruning сокращает число epochs.
Практический ориентир всего поиска — **4–8часов**, уточняется progress/логами.

CPU8/BLAS1; RSS stop40GiB и availableRAM stop4GiB. CUDA allocator cap75%,
старт freeVRAM>=10000MiB, stop<2048MiB. Собственные run files<=30GiB;
Linux free start40/stop10GiB, реальный Windows G: free start80/stop50GiB.
Хранятся лучшие portable weights каждого trial, только один активный optimizer
checkpoint и небольшой sample candidates. Ожидаемый объём заметно меньше30GiB;
cap является условием остановки, не предварительной резервацией.
Monitor проверяет каждые2s, поэтому короткий пик между проверками возможен.

## Результаты, наблюдение и resume

- `artifacts/task15_sasrec_optuna_v1/best_recipe.json`: параметры, epoch и source policy.
- `artifacts/task15_sasrec_optuna_v1/metrics.json`: выбранная оценка, baseline,
  обе P20, standalone recall и вложенные overlap/policy diagnostics.
- `artifacts/task15_sasrec_optuna_v1/trials.json`: параметры, состояния и причины остановки trials.
- `artifacts/task15_sasrec_optuna_v1/model/`: лучший portable model.
- `artifacts/task15_sasrec_optuna_v1/study.sqlite3`: опубликованный snapshot study.
- `artifacts/.task15_sasrec_optuna_v1.work/`: рабочая SQLite study, кэши,
  `trials/trial_*/epoch_metrics.json`, лучшие weights и активный checkpoint.
- `artifacts/.task15_sasrec_optuna_v1.work/best_so_far.json`: ссылка на текущий
  лучший проверенный checkpoint во время работы.
- `logs/task15_sasrec_optuna_v1.log`, `.resources.log`, `.resources.json`: логи/ресурсы.

```bash
tail -F logs/task15_sasrec_optuna_v1.log
cat logs/task15_sasrec_optuna_v1.resources.json
```

Остановка: Ctrl-C, либо из второго терминала:

```bash
kill -TERM "$(cat logs/task15_sasrec_optuna_v1.pid)"
```

Для resume повторить ту же launch command. Завершённые trials/epochs
пропускаются; незавершённая epoch повторяется от последнего optimizer/RNG
checkpoint. Не менять config или код посреди search: hashes проверяются,
для изменённого эксперимента нужен новый run ID. TPE получает seed отдельно
для каждого trial/parameter, чтобы восстановление между suggestions не меняло
параметры. Exact GPU reproducibility между разным hardware/version не обещается.
Полные run не запускаются автоматически после восстановления/проверки.

Готовый artifact защищён от overwrite. Проверка без обучения:

```bash
.venv/bin/python scripts/run_sasrec_optuna.py \
  --verify-only artifacts/task15_sasrec_optuna_v1
```

При публикации full search записывается отдельная строка `experiments/results.csv`
с явным split `rolling_1_ID_sample_next_day_selection`. Smoke туда не попадает.
Если публикация успела завершиться, а append был прерван, `--verify-only`
восстановит отсутствующую строку без дублирования.

## Проверки и файлы

Проверено45tests: прежние model/data/metric/candidate tests плюс7Optuna/selection.
Real-data CPU smoke прошёл epoch pause/resume; GPU smoke с d128/L100 и полным
каталогом прошёл pause после первого trial и продолжение со второго. GPU peak
allocated3.28GiB, внешний sampled device5550MiB/RSS2.76GiB. Tiny smoke P20=0
не характеризует обученную на полном history SASRec. Полный search не запускался.
Проверены hashes, portable CPU scores, точные top20 known/unseen recommendations
и пересчёт всех selection metrics; отчёт:`artifacts/task15_optuna_preflight_v1/`.
Дополнительный independent CPU repeat совпал с epoch-resume запуском по
параметрам trials, weights и candidates точно. При audit обнаружено округление
старого oracle mean на1ULP; итоговый evaluator делит целую сумму clipped hits
на знаменатель, чтобы floating reduction не менял tie-break между policies.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_sasrec*.py' -v
bash -n scripts/run_task15_sasrec_optuna.sh
./scripts/run_task15_sasrec_optuna.sh --help
```

Новые файлы: `sasrec_selection.py`, `scripts/run_sasrec_optuna.py`, исполняемый
`scripts/run_task15_sasrec_optuna.sh`, три `configs/task15_sasrec_optuna*_v1.json`,
`tests/test_sasrec_optuna.py`, этот документ. Обновлены dependency file и handoff.
Существующие model/loader/benchmark weights, raw data, Task06 sources и ranker
не изменяются этим search runner.

Официальные API, использованные в реализации:
[Optuna Study ask/tell](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html),
[MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html),
[Optuna4.9.0 package](https://pypi.org/project/optuna/4.9.0/).
