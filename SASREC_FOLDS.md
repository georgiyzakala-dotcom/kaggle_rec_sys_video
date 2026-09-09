# Task15: frozen SASRec на rolling-фолдах

Обновление 2026-09-08: пользователь выполнил полный запуск
`artifacts/task15_sasrec_folds_v1`, verifier прошёл. Runtime11846,36s (3ч17мин).
Для оценки более глубоких списков теперь подготовлен отдельный inference-only
launcher: [SASREC_TOP600.md](SASREC_TOP600.md). Повторять обучение не требуется.

Полные micro recall, проценты; другие три источника в union всегда по200:

| Fold | SASRec200 | ALS200 | Baseline ALS200 | ALS200 + SASRec200 | ALS400 без SASRec |
|---|---:|---:|---:|---:|---:|
| rolling_1 | 3.6059 | 4.4693 | 8.1429 | 9.9115 | 10.4838 |
| rolling_2 | 4.3755 | 5.4496 | 9.3219 | 11.4667 | 12.1214 |
| rolling_3 | 5.2806 | 6.6337 | 11.2164 | 13.6827 | 14.4216 |
| canonical | 5.3030 | 6.7426 | 10.7002 | 13.2489 | 14.0035 |

SASRec добавляет позитивы сверх baseline, но ALS400 сильнее двух моделей
по200 при одинаковой сумме caps1000 во всех фолдах. При бюджете800 замена
ALS и frozen blend ALS150/SASRec50 тоже не улучшают recall. На canonical
blend даёт10.4528%, замена9.9666%. Решения о замене ALS этими результатами
не поддерживаются; преимущество двух моделей при бюджете1200 ещё проверяется.

Canonical overlap @200: средняя доля SASRec items в ALS16.9973%, micro
Jaccard9.2845%; shared positive hits27658, SASRec-only40441, ALS-only58927.
SASRec добавляет32728hits сверх всех четырёх старых источников; у ALS остаётся
42149hits, которых нет в остальных источниках вместе с SASRec.
Canonical standalone SASRec P20 all/labeled:
`0.0030479335704864303 / 0.0041489839223047414`;
ALS: `0.003859816539430033 / 0.0052541554449250525`.
Это P20 отдельных моделей с global fallback; новый ранкер не обучался.
Полные source curves, overlap и обе oracle P20 — в исходном `metrics.json`.

## Что показала Optuna

Проверен опубликованный `artifacts/task15_sasrec_optuna_v1`: 24 trials,
17977.06s активной работы (4 ч 59 мин). Portable CPU inference, сохранённый
recall и обе Precision@20 прошли verifier. Подбор использовал только 8192
target IDs rolling_1; эти метрики нельзя сравнивать с полными canonical scores.

| Вариант на selection sample | Recall позитивов |
|---|---:|
| ALS@200 отдельно | 0.04268124 |
| SASRec@200 отдельно | 0.03587279 |
| Текущий union, сумма caps 800 | 0.07846184 |
| ALS 150 + SASRec 50 + остальные 3x200 | 0.07761902 |
| SASRec 200 вместо ALS 200 + остальные 3x200 | 0.07570949 |
| Добавление SASRec 200 к baseline, сумма caps 1000 | 0.09654310 |

Победитель среди проверенных SASRec recipes — trial 12, epoch 12. При том же
бюджете800 он уступил baseline на 0.0008428261 recall (0.0843 процентного пункта).
Добавление сверх baseline принесло 1373 новых positive pairs на этой выборке.
Поэтому новый запуск включает контроль увеличения выдачи прежнего ALS до 400.
Оснований автоматически заменять ALS или менять production ranker пока нет.

Пересечение SASRec 200/ALS 200: 199599 общих пар, micro Jaccard 0.06950307;
средняя доля списка SASRec, встречающаяся у ALS, 0.13069605 (только определённые
знаменатели). Общих positive hits 949, только у SASRec 1775, только у ALS 2292.
Standalone SASRec top20 с независимым global fallback:
`precision_at_20_all_targets=0.0033508300781250003`,
`precision_at_20_labeled_users=0.004124718256949662`.

## Зафиксированный рецепт

- Embedding 128, 1 block, 2 heads, sequence length 100, dropout 0.2.
- Tied embeddings, pre-norm/final LayerNorm, right padding и last-valid query.
- AdamW, learning_rate 0.0014935660033749737, weight_decay 0.0002979293642858734.
- Batch 128, 32 negatives/query, gradient clipping 1.0, query chunk 512.
- 12 epochs, BF16 training, seed 42; обычный PyTorch без Lightning.
- Epoch = одно детерминированно выбранное случайное окно на eligible user.

Параметры, число эпох и primary policy `blend_150_50` читаются из защищённого
SHA256 файла `best_recipe.json`. На новых фолдах нет Optuna, early stopping
или выбора эпох по их labels. Все альтернативные budgets публикуются как
заранее заданные сравнения, primary policy по ним не перевыбирается.

| Fold | История до cutoff | Обучение SASRec | Оценка |
|---|---|---|---|
| rolling_1 | 2024-11-29 08:56:28 | Reuse полностью обученного победителя Optuna | Все target users, selection diagnostic |
| rolling_2 | 2024-11-30 08:56:28 | С новых весов, 12 epochs на всей history | Все target users, следующий день |
| rolling_3 | 2024-12-01 08:56:28 | С новых весов, 12 epochs на всей history | Все target users, следующий день |
| canonical | 2024-12-02 08:56:28 | С новых весов, 12 epochs на всей history | Полный canonical holdout |

Cutoffs, history catalog и counts в коде берутся из fold manifests.
Используются immutable daily history, собранные после точного raw timestamp
split; календарные snapshots из `data/` не используются. Labels не попадают
в training windows, negatives, vocabularies или fallback. Cold/seen GT funnel
копируется из общего data pipeline. На полном запуске ожидаются все 200152
target IDs, включая пользователей без истории или без positive history.

## Запуск, остановка, продолжение

```bash
./scripts/run_task15_sasrec_folds.sh
```

Явная конфигурация:

```bash
./scripts/run_task15_sasrec_folds.sh configs/task15_sasrec_folds_v1.json
```

Запуск последовательно проверяет входы, готовит history-only sequences,
обучает/восстанавливает модель, выдаёт candidates и вычисляет метрики каждого
фолда. Progress/ETA отображаются по фолдам, эпохам, windows/batches и блокам
evaluation users. Нет полного final-history fit для submission: этот этап
нужен для проверки кандидатов на наблюдаемом следующем дне.

```bash
tail -F logs/task15_sasrec_folds_v1.log
cat logs/task15_sasrec_folds_v1.resources.json
```

Остановка — Ctrl-C либо из другого терминала:

```bash
kill -TERM "$(cat logs/task15_sasrec_folds_v1.pid)"
```

Повторить исходную команду для resume. Atomic checkpoints сохраняют веса,
optimizer и RNG после каждой эпохи; interrupted epoch повторяется с предыдущей
границы. Готовая модель и завершённые блоки оценки не вычисляются заново.
Конфигурацию/run_id/код посреди запуска менять нельзя: hashes проверяются.
Существующий завершённый результат launcher не перезаписывает. Для проверки:

```bash
.venv/bin/python scripts/run_sasrec_folds.py --verify-only artifacts/task15_sasrec_folds_v1
```

## Сравнения и выходные файлы

Существующие Task06 sources переиспользуются с исходными scores/ranks.
ALS@400 считается из той же сохранённой fold model, без fit. Первые200 items
сравниваются с сохранённым ALS@200; несовпадение прерывает запуск.

- Для всех пяти источников: native recall@50/100/150/200, macro recall по
  labeled users отдельно от micro recall, coverage, candidate counts и их
  quantiles, user hit rate, обе oracle P20, обе standalone P20 с fallback.
- Union: baseline 800, replace_als, blends150/50,100/100,50/150, add_source 1000,
  ALS 400 control 1000. Сохраняются recall, oracle и фактические unique counts.
- SASRec против каждого существующего источника: raw intersection/Jaccard,
  средние/quantiles долей, определённые и неопределённые знаменатели, вариант
  только с двумя непустыми списками; общие и уникальные positive hits.
- Новые hits сверх всех источников, ALS hits, которые не покрывает даже
  SASRec вместе с другими источниками, и разница add_source против ALS 400.

Шардирование по 2048 target IDs ограничивает RAM. Метрики агрегируются из
целых per-user counts; recall и quantiles не усредняются по shards. Итог
сверяет baseline и каждый старый source с исходными полными Task06 metrics.
Каждая top20 проверяется на точный universe, 20 уникальных известных unseen
items, отсутствие null и дубликатов. Native metrics не включают fallback.

Результат публикуется атомарно в `artifacts/task15_sasrec_folds_v1/`:

- `metrics.json`, `comparison.csv`, `report.md`, `frozen_recipe.json`, `config.json`.
- `folds/<fold>/trained/model/`: самостоятельный portable SASRec artifact;
  рядом training metrics и CPU inference probe.
- `folds/<fold>/metrics.json`, `per_user.parquet`, target universe/GT.
- `folds/<fold>/evaluation/part-*/`: SASRec 200, ALS 400, source top20,
  per-user counts, duration/metric checkpoint и SHA256 manifest.

Candidate parquet shards подходят для следующего этапа интеграции с ranker.
Файл submission не создаётся; текущая production модель не меняется.
Полные fold runs добавляются в `experiments/results.csv`; smokes не добавляются.
Mean по rolling_2/3 назван отдельно от canonical и rolling_1; это среднее
fold scores, а не micro recall по объединённым labels разных дней.

## Ресурсы и проверка

Лимиты: CPU 8/BLAS 1, RSS<=40GiB, GPU allocator <= 75%, freeVRAM stop 2GiB,
собственные run files<=30GiB; Linux free start 40/stop 10GiB и Windows G:
free start 80/stop 50GiB. Launcher использует общий Task15 lock и не допускает
параллельного дублирующего запуска. Новые зависимости не устанавливались.
Publication использует hard links внутри диска, сохраняя portable файлы;
supervisor консервативно считает каждый путь в work/output в disk budget.

Ориентир полного запуска — **3–5 часов**, с уточнением ETA после первого фолда.
Inference probe на 256 target users rolling_1 с рабочими batch sizes дал
SASRec 1.01s, ALS 400 1.09s и metrics 0.81s. Каталоги остальных фолдов больше,
а стоимость checkpoint/I/O и оценки 2048 users за раз не масштабируется строго
линейно. Это оценка длительности, не обещание верхней границы.

```bash
bash -n scripts/run_task15_sasrec_folds.sh
./scripts/run_task15_sasrec_folds.sh --help
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_sasrec_folds.py'
```

Real-data GPU smoke: `configs/task15_sasrec_folds_gpu_smoke_v2.json`,
512 training users, 16 evaluation users, 2 epochs на rolling_2/canonical,
полный catalog каждого фолда. Его качество не интерпретируется как результат
полного эксперимента. Первый smoke v1 выявил отличие средней overlap доли
на 1ULP при повторной parallel reduction; verifier теперь допускает только
численную погрешность floating metrics, целые counts проверяет точно.
Synthetic tests также проверяют forced epoch/shard resume, отсутствие ALS fit,
independent same-seed repeat, empty GT, UInt64 IDs, seen/cold и dedup.
Найден и исправлен крайний случай ALS: при k больше числа unseen items
backend может вернуть seen items с конечным score -FLT_MAX; они явно исключены.

GPU smoke v2 опубликован за 89.04s; training ограничен 512 users/2epochs,
eval — 16 users на каждом из двух фолдов. Independent GPU repeat дал точное
совпадение SHA256 weights, SASRec/ALS 400 candidates и per-user counts.
Portable verifier опубликованного artifact прошёл. Результаты проверки,
ресурсы и inference probe: `artifacts/task15_folds_preflight_v1/metrics.json`.
Прошли 65 tests (SASRec/folds/ALS/data/metrics/candidates), Ruff, --help и
bash syntax. GPU smoke: peak RSS 3.78GiB, device 6714MiB, Torch allocated 4.40GiB;
обе P20 равны 0 в обоих фолдах. Smokes проверяют реализацию, а не качество.

Ограничения: rolling_1 участвовал в подборе, canonical уже многократно открыт
в проекте, вся timeline короткая, seed один. Даже положительный candidate
recall требует отдельного matched ranker backtest перед production promotion.
