# Ранкер на ALS600 + SASRec600 и новый submission

Запуск из корня репозитория в существующей `.venv`:

```bash
./scripts/run_task15_sasrec_ranker.sh
```

Конфигурация: `configs/task15_sasrec_ranker_v1.json`.
Результат: `artifacts/task15_sasrec_ranker_v1/submission.csv`.
Предыдущий `artifacts/task14_full_fit_v1/submission.csv` сохраняется.
Скрипт не отправляет файл в Kaggle.

## Что выполняется

1. Проверка закреплённых артефактов, исходных данных и временных границ.
2. Ранкер на rolling_1 + rolling_2; оценка на всех target users rolling_3.
3. Ранкер на rolling_1 + rolling_2 + rolling_3; оценка canonical.
4. Финальный ранкер на всех четырёх размеченных фолдах.
5. SASRec на полной истории, включая последний неполный календарный день:
   12 эпох, рецепт Optuna trial 12, BF16, batch128, d128, L100, 32 negatives.
6. Кандидаты на полной истории, их ранжирование и top20 для каждого target user.
7. Строгая проверка рекомендаций, запись CSV и атомарная публикация артефакта.

Во всех этапах лимиты источников одинаковые: ALS600, SASRec600,
global/recency/item2item по200; максимум1800 уникальных кандидатов.
Готовые fold predictions используются из `task15_sasrec_top600_v1`.
Четыре модели для будущего периода уже обучены на полной истории в Task14;
их веса, история и признаки переиспользуются с проверкой SHA256.
Для будущего периода отдельно обучается SASRec; fold-модели не подменяют
модель полной истории.

## Признаки и обучение

232 числовых признака: 221 признак Task13 D_all плюс 11 признаков SASRec.
Новые признаки включают попадание в источник, native score/rank,
cross score/availability, ранг cross score внутри полного пользовательского
пула, нормы векторов и cosine. Cross scores считаются для всех пар в union,
в том числе для кандидатов, которых сам SASRec не предложил.
Сырые координаты embeddings не передаются ранкеру.

Для каждой из трёх моделей бюджет — примерно30 млн обучающих строк,
распределённых поровну между доступными training folds. Все найденные
кандидатами позитивы сохраняются; негативы выбираются равномерно и
детерминированно по `(fold, seed, user_id, item_id)`.
`sample_weight = 1 / inclusion_probability`; для позитива weight=1.
Размер30M — математическое ожидание, а не точное ограничение числа строк.
Это новый sampler: прежний upstream hard-negative sampler Task07 не переносится
на расширенный пул. Поэтому сравнение оценивает весь обновлённый pipeline,
а не изолированный эффект одного признака.

Ранги внутри пользовательского пула рассчитываются **до** сэмплирования.
Дорогие pair/history features для обучения материализуются только для
оставшихся строк. На валидации и финальном inference оцениваются полные пулы.
Labels используются только для обучения ранкера/его оценки; fit candidate
models, history features и SASRec sequences используют соответствующую историю.

Рецепт CatBoost зафиксирован по Task13: Logloss, 1030 деревьев, depth7,
learning_rate0.08, l2=3, Bernoulli0.8, border_count32, scale_pos_weight1,
seed42. Новые границы квантования обучаются только на training rows данного
этапа. Выбора гиперпараметров или числа деревьев по canonical здесь нет.
Torch Lightning не требуется: используется существующий PyTorch runner
с сохранением весов, optimizer state и RNG после каждой эпохи.

## Оценка качества

`metrics.json` содержит `validation_folds` с двумя отдельно названными P20:
`precision_at_20_all_targets` и `precision_at_20_labeled_users`.
Полный target universe —200152 пользователей; в каждом фолде свой набор GT.
Recall — доля уникальных позитивных пар, найденных union, после удаления
seen-пар и cold items. Проверяется совпадение counts/hits с готовым top600.

Зафиксированный reference Task13 D_all:

| Fold | P20 all targets | P20 labeled users | Hits |
|---|---:|---:|---:|
| rolling_3 | 0.0072682261481274235 | 0.009013544325757763 | 29095 |
| canonical | 0.005077640992845437 | 0.006911912728855518 | 20326 |

`delta_p20_all_targets` показывает разницу с reference.
`improved_on_both_validation_folds` становится true только при положительной
разнице на обоих фолдах. Новый CSV создаётся и при отрицательной разнице,
чтобы результат можно было исследовать; автоматической замены старого submission
или продвижения модели нет. Метрики будущего периода остаются null.
Canonical уже использовался в предыдущих исследованиях; это повторно открытый
holdout. Рост candidate recall до21.4281% сам по себе не доказывает рост P20
и не гарантирует улучшения в лидерборде.

## Ресурсы, прогресс и продолжение

Ориентир для полного запуска на RTX5070Ti — **6–9 часов**. Это экстраполяция,
не измерение полного эксперимента. Замер256 пользователей на текущем железе:
training features1.61s; полные evaluation features1.29s;
score1030 деревьев0.49s; production retrieval2.96s,
production cross scores0.62s и остальные features0.78s.
В замере ранжирования использовался сохранённый Task13 с221 признаками и1030
деревьями; новый232-feature fit ещё не выполнен в полном масштабе.
Для production retrieval использовался SASRec smoke с теми же размерами
архитектуры/каталога, но одной эпохой: это замер времени, не качества.
Артефакт измерения: `artifacts/task15_sasrec_ranker_timing_v1/`.
На итог влияют чтение холодного кэша, запись30M строк, размер пользовательской
истории и конкурирующая нагрузка. Предыдущее квантование30M занимало752s,
CatBoost fit130s; теперь таких этапов три. Предыдущие12 эпох SASRec canonical
занимали1122s без всех затрат на сохранение/публикацию.

Основная порция —256 пользователей; числовые признаки Float32, IDs сохраняют
UInt64/Int32. Одновременно обрабатывается один фолд. Промежуточные sampled
features и quantized pool удаляются после атомарного сохранения модели этапа
и его оценки. Не сохраняется полная многомиллиардная таблица признаков union.

Лимиты: CPU8, BLAS1, RSS40GiB, CatBoost GPU memory fraction0.7,
SASRec CUDA fraction0.7. Внешний supervisor проверяет RAM, GPU, Linux filesystem
и `/mnt/g` даже во время native CatBoost вызовов. На G требуется155GiB в начале,
остановка при остатке меньше50GiB. Временный TSV ограничен60GiB;
ориентир по дополнительному пиковому месту —80–100GiB. Это оценка для полного
30M-run, а не измерение smoke. При resume уже занятые reusable caches учитываются
в стартовом резерве. На WSL удаление файлов освобождает место внутри filesystem;
размер VHDX на Windows может не уменьшиться автоматически.

Прогресс: общий этап, fold/user shards, деревья CatBoost, эпохи и batches SASRec.
У измеримых шагов отображаются elapsed/ETA; во время квантования пишется heartbeat.
Обычный лог ограничен ротацией и не содержит ANSI:

```bash
tail -f logs/task15_sasrec_ranker_v1.log
cat logs/task15_sasrec_ranker_v1.resources.json
```

Остановка —Ctrl-C или:

```bash
kill -TERM "$(cat logs/task15_sasrec_ranker_v1.pid)"
```

Продолжение —та же команда запуска. Не меняйте код/config между остановкой
и resume: checkpoint закреплён их хешами. Завершённый output не перезаписывается.
Для остановки после первого полного backtest можно запустить:

```bash
./scripts/run_task15_sasrec_ranker.sh configs/task15_sasrec_ranker_v1.json --stop-after-phase rolling_validation
```

Продолжение всех этапов —та же команда без `--stop-after-phase`.
Checkpoint: `artifacts/.task15_sasrec_ranker_v1.work/checkpoint.json`.
Модели/оценки завершённых этапов доступны в `.work/completed/` ещё до CSV.
Weights, profiles и lookups публикуются через hard links в пределах `artifacts/`;
их следует считать неизменяемыми, как и исходные model artifacts.

Проверка готового результата без fit:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 \
  .venv/bin/python scripts/run_expanded_ranker.py \
  --verify-only artifacts/task15_sasrec_ranker_v1
```

CSV сохраняет подтверждённую в Task14 сериализацию: колонки `user_id,item_ids`,
где `item_ids` —строгий JSON-список20 чисел. Проверяются round trip CSV,
полнота target users, уникальность items/users, отсутствие null/seen/unknown
items, одинаковый порядок признаков и portable prediction сохранённого ранкера.

## Разработка

Код: `expanded_ranker.py`, `scripts/run_expanded_ranker.py`;
launcher: `scripts/run_task15_sasrec_ranker.sh`;
тесты: `tests/test_expanded_ranker.py`. Зависимости не добавлялись.
Smoke-конфиги имеют отдельные run IDs и не добавляются в experiment log
как полные эксперименты. Полный запуск в сессии Codex не выполняется.

GPU end-to-end smoke `task15_sasrec_ranker_gpu_smoke_v1` прошёл за~264s
с учётом публикации:16 target users, три ранкера по12 деревьев,
одна эпоха SASRec на32 training users при полном history catalog.
RSS peak по процессу6.95GiB, device VRAM peak6175MiB. Готовый CSV содержит
16 строк по20 items; semantic validation и отдельный `--verify-only` прошли.
Это только инженерная проверка; её P20 не сопоставим с reference на200152 users.
Unit tests дополнительно покрывают relevance, seen filtering, deduplication,
обе P20, UInt64 IDs, отсутствие позитивных потерь при sampling, сохранение
полных query ranks, SASRec cross scores вне native topK и временные границы.

CPU smoke `task15_sasrec_ranker_cpu_smoke_v5` также завершён (151.57s активного
времени): намеренная остановка после rolling validation, продолжение без повторного
fit этого этапа, завершение до CSV и отдельная проверка артефакта. Predictions
всех трёх ранкеров и финальные recommendations точно совпали с предыдущим
CPU запуском с тем же seed. Проверены7 новых unit tests и9 существующих
`test_ranker_backtest.py`; Ruff, `--help` и `bash -n` прошли.

Команды ограниченных smoke (завершённые run IDs повторно не перезаписываются):

```bash
./scripts/run_task15_sasrec_ranker.sh configs/task15_sasrec_ranker_gpu_smoke_v1.json
./scripts/run_task15_sasrec_ranker.sh configs/task15_sasrec_ranker_cpu_smoke_v5.json --stop-after-phase rolling_validation
./scripts/run_task15_sasrec_ranker.sh configs/task15_sasrec_ranker_cpu_smoke_v5.json
```

```bash
bash -n scripts/run_task15_sasrec_ranker.sh
./scripts/run_task15_sasrec_ranker.sh --help
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p test_expanded_ranker.py -v
```
