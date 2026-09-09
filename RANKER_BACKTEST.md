# Task 12: проверка обучения ранкера

Первый этап после аудита: проверить multi-fold обучение, явное квантование и
выбор числа деревьев по Precision@20. Кандидаты и 201 исходный признак берутся
из immutable `artifacts/task07_ranker_dataset_v1/`. Новые history features,
изменения retrieval и повторное создание submission — следующие этапы,
зависящие от результатов этой проверки.

## Запуск из корня репозитория

```bash
./scripts/run_task12_ranker_backtest.sh configs/task12_ranker_backtest_v1.json
```

Полный запуск выполнен пользователем 2026-09-05: 82.84 min, canonical прирост
+877 hits (+4.5753%) относительно Task08. Повтор исходной команды откажется
перезаписывать completed artifact; для повторного experiment нужен новый
run ID. Launcher использует `.venv`,
ограничивает CPU до 8 threads и предотвращает параллельные Task 12 runs.
Перед началом проверяются Python 3.12, CatBoost 1.2.10, минимум 140 GiB
свободного места в Linux, 40 GiB доступной RAM и 12,000 MiB свободной VRAM.
При resume часть требуемого дискового резерва уже занята собственными caches.
Для WSL также нужен запас на физическом диске с VHDX: этот runner проверяет
Linux filesystem, а не свободное место Windows. Последний сохранённый Task 11
отчёт показывал около 212 GiB свободного места на Windows G:; это не текущая
проверка диска.

Runner выполняет три обучения CatBoost: два сравниваемых варианта и refit выбранного.
На каждом этапе train budget — около 46 миллионов строк после sampling.
Фактическое число немного отличается из-за вероятностного отбора; все
позитивы сохраняются. Временный TSV ограничен 75 GiB. Из 82.84 min полного
run три quantization операции заняли 55.04 min, три fit — 10.86 min;
peak RSS составил 45.97 GiB. Терминал показывает прогресс, elapsed и ETA
для фаз, конфигураций, shards и деревьев. CPU quantization не предоставляет
поштучный прогресс: в этот момент отображается `quantize_cpu`, затем
`save_quantized_pool`.

## Что сравнивается

1. Train на `rolling_1 + rolling_2`, выбор на `rolling_3`.
   Исходный Task 08 также оценивается на тех же пользователях как reference.
2. `frozen_task08`: точные старые borders. Признаки без borders исключаются
   при квантовании, чтобы CatBoost не достраивал их неявно.
3. `fit_training`: новые borders, рассчитанные только по training examples;
   явный `border_count=32`, `GreedyLogSum` и seed 42.
4. Для каждого варианта один fit до 1,400 trees, затем staged inference для
   100/200/400/600/800/1,000/1,030/1,200/1,400 trees. Это позволяет отдельно
   сравнить квантование при 1,030 trees и результат выбора tree count.
5. Выбор по P@20 на 16,384 пользователях, отобранных только по хешу ID.
   У каждого пользователя сохраняется полный список кандидатов. Пользователи
   без GT остаются в all-target denominator. Logloss записывается как
   диагностика и не определяет победителя. При равных hits выбирается меньше
   деревьев, затем `frozen_task08`.
6. Победитель сохраняется до открытия canonical examples. Он оценивается на
   всех target users `rolling_3`, затем переобучается на
   `rolling_1 + rolling_2 + rolling_3` с выбранным числом деревьев и способом
   квантования. Выполняется одна canonical оценка на всех 200,152 targets.
   Рецепт балансирует ожидаемое число training rows по folds и учитывает
   полную вероятность попадания negatives в training через inverse weights.
7. Сохраняются обе P@20, candidate recall/coverage/oracle, hits, сравнение с
   Task 08, runtime, peak RSS, фактические borders и portable model.

Победитель здесь — лучший из двух новых вариантов; поле
`selection_beats_task08` отдельно показывает, выиграл ли он у старой модели.
Он не назначается автоматически production best. Полный `rolling_3` report
включает пользователей, участвовавших в выборе, поэтому это не независимый
holdout. Canonical уже использовался в предыдущих задачах: теперь он остаётся
известной диагностикой, хотя данный runner не использует его для выбора.

## Результаты, наблюдение и остановка

- Итог: `artifacts/task12_ranker_backtest_v1/`.
- Важные файлы: `metrics.json`, `selection_curves.csv`, `winner.json`,
  `training.json`, `quantization/`, `model/`, `canonical/recommendations.parquet`.
- Resume state: `artifacts/.task12_ranker_backtest_v1.work/checkpoint.json`.
- Лог: `logs/task12_ranker_backtest_v1.log`, ротация 5 MB × 4 файла,
  timestamp и stage/config/fold/operation в каждой строке, без ANSI sequences.
- Сводный experiment log: `experiments/results.csv`, append только для
  завершённого full run. Smoke в этот журнал не записывается.

```bash
tail -F logs/task12_ranker_backtest_v1.log
```

Остановка из другого терминала:

```bash
kill -TERM "$(cat logs/task12_ranker_backtest_v1.pid)"
```

В активном терминале также работает Ctrl-C. Во время native CatBoost call
обработка сигнала может задержаться. Resume запускается той же исходной
командой: готовые операции проверяются и пропускаются, незавершённая операция
повторяется; CatBoost использует snapshot, если успел его записать. Не менять
код и config между остановкой и resume: hashes защищают от смешивания runs.
SIGKILL не даёт сохранить текущую операцию. Уже опубликованный результат
перезаписать нельзя.

Можно запланированно остановиться после выбора, ещё до canonical refit:

```bash
./scripts/run_task12_ranker_backtest.sh configs/task12_ranker_backtest_v1.json --stop-after-selection
```

Затем повторить команду без `--stop-after-selection`.
Большие временные training caches удаляются после проверки и публикации.
Самостоятельная проверка готового результата:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 \
  .venv/bin/python scripts/run_ranker_backtest.py \
  --verify-only artifacts/task12_ranker_backtest_v1
```

Review сохранён в `artifacts/task12_review_20260905_v1/findings.md` и
`metrics.json`. Исходные `metrics.json` и `selection_curves.csv` не изменялись.
Отчёт также доступен в `research/12_ranker_training_backtest.ipynb`.

## Выполненные проверки

- Unit tests покрывают временные границы, сохранение positives, sampling
  weights, UInt64 IDs, оба denominator, выбор trees по hits, frozen/fresh
  borders, staged predictions и portable prefixes.
- Synthetic end-to-end проверяет pause/resume, запрет canonical на этапе
  выбора, контроль checksums, атомарность, запрет overwrite и final top-20.
- Real-data CPU smoke: 64 пользователя, максимум 12 trees, 12,058 sampled
  training rows на selection. Запуск остановлен после selection и продолжен.
  Canonical smoke: 3 hits; P@20 all/labeled
  `0.0023437500000000003 / 0.003488372093023256`. Время двух частей около
  7.42 s, peak RSS около 956 MiB. Эти метрики не сравнимы с full runs.
- Сохранённый `.cbm` повторяет inference детерминированно. Отдельные GPU fits
  при одинаковом seed могут различаться из-за floating-point reductions.
- Независимый повтор CPU smoke (`task12_ranker_backtest_cpu_repeat_v1`)
  воспроизвёл обе таблицы рекомендаций и все selection curves точно.
- GPU smoke (`task12_ranker_backtest_gpu_smoke_v1`): 64 users, depth 7,
  максимум 12 trees, выбран prefix 4. Canonical smoke P@20 all/labeled
  `0.0046875 / 0.006976744186046512`, 6 hits; 7.37 s, peak RSS 1,194 MiB.
  Portable `--verify-only` прошёл. Исходный sandbox attempt был остановлен
  preflight: NVML сообщил `GPU access blocked by the operating system`.
  Тот же bounded run успешно выполнен с разрешённым доступом к GPU.
- Для повторения этих дополнительных smokes нужны новые run IDs;
  сохранённые входные configs:
  `artifacts/task12_ranker_backtest_cpu_repeat_v1.input.json` и
  `artifacts/task12_ranker_backtest_gpu_smoke_v1.input.json`.
- Full artifact прошёл независимый `--verify-only`: checksums 27 files,
  обе canonical метрики, portable probe. Отдельно проверены полные top-20
  semantics и метрики двух моделей на rolling_3/canonical. 142 tests прошли.

## Вывод по полному запуску

Новый best validation baseline — Task12: P@20 all/labeled
`0.005007444342299852 / 0.006816357898745886`, 20,045 hits, против Task08
`0.004788360845757224 / 0.006518131614026497`, 19,168 hits.
Выбран `fit_training`, 1,030 trees. На rolling_3 вне selection users также
есть +242 hits (+0.95%). Canonical paired-user bootstrap interval delta all
`[0.0001743675, 0.0002642991]` положителен, но не учитывает temporal shift и
повторные GPU fits.

При fixed 1,030 trees fresh borders дали 2,223 selection hits против 2,195 у
frozen. Разница лучших точек policies — 11 hits, а 1,030 против 1,000 trees
отличаются всего на два hits. Выбор не следует трактовать как точный optimum.
Canonical gain относится ко всему recipe: изменились borders, folds и
sampling. Рост до 1,400 trees снижает P@20, хотя Logloss ещё улучшается.

User 72h shares получили 32 quantization borders, модель использует 11/12
из них вместо 1/1 у Task08. Это исправляет потерю разрешения, но не заменяет
добавление candidate-conditioned history features. Candidate recall остался
10.70%, oracle utilization вырос с 13.96% до 14.60%. Следующий этап —
history-feature ablations на том же union, затем retrieval-depth diagnostics.
Существующий Task11 submission не переобучался и не изменялся.
