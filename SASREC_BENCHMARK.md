# Task15: одно обучение SASRec для замера скорости

Обновление2026-09-07: пользовательский `task15_sasrec_benchmark_v1` завершён
и повторно проверен. Runtime164.83s, median epoch49.77s с подготовкой,
peakRSS2.89GiB. Этот run ID повторно не запускать. Следующий launcher и
разбор измерений: [SASREC_OPTUNA.md](SASREC_OPTUNA.md).
Ниже сохранена исходная инструкция подготовки и запуска benchmark.

Этап, согласованный 2026-09-07: сначала GPU benchmark, после его результата —
отдельный Optuna search на rolling_1 с target следующего дня, затем обучение
лучшего recipe на остальных rolling folds и анализ recall/overlap с ALS.

## Запуск

Из корня репозитория:

```bash
./scripts/run_task15_sasrec_benchmark.sh
```

Полная форма той же команды:

```bash
./scripts/run_task15_sasrec_benchmark.sh configs/task15_sasrec_benchmark_v1.json
```

Это **одно обучение одной модели, три эпохи** на всей history rolling_1.
Каждая эпоха использует один случайный history window на каждого пользователя
с минимум двумя positive daily rows и доступным negative item. Порядок users,
windows и negatives определяется seed42/epoch. Это не полный перебор всех
возможных prefixes длинной истории. Epoch/step counts и число supervised
targets сохраняются явно для будущего сравнения рецептов.

Модель: d64, 2 heads, 2 pre-norm blocks, length50, dropout0.1, tied embeddings,
final LayerNorm, normal initialization std0.02 и PAD=0. Right padding,
last-valid gather, masked padded states; пустая история обрабатывается отдельно.
AdamW, lr0.001, weight_decay0.0001, gradient_clip1.0, batch128, 64 uniform
negatives/valid target. Mixed precision BF16; loss reduction FP32.
Это стартовый recipe для скорости, гиперпараметры ещё не подобраны.

Последовательности строятся только из
`artifacts/task02_fold_20241129_v1/history_daily.parquet`:
cutoff `2024-11-29 08:56:28`. Целевой следующий день для будущего Optuna —
`[2024-11-29 08:56:28, 2024-11-30 08:56:28)`.
Ни raw events, ни validation labels benchmark не читает. Словарь — все
history items (по artifact:1,337,354), а seen — все history events, включая
weak watches и items вне последних50. Повторы positive items между днями
сохраняются. `dt=min` остаётся приблизительным порядком daily interactions.

## Что происходит в терминале

1. Resource preflight: RAM, GPU, свободный Linux disk и **реальный Windows G:**.
2. Проверка history checksum/cutoff, построение flat arrays/mappings/seen lists.
3. Три эпохи: отдельные progress bars для epochs, подготовки windows/negatives
   и training batches; elapsed/ETA, loss и время batch. Первая эпоха включает
   прогрев CUDA; в статистике steady steps первые пять batches исключены.
4. Небольшой exact retrieval probe:128 target users с positive history,
   выбранных по hash user_id без labels, top200 по полному history catalog.
   Он отдельно измеряет скорость inference; это не качество на полном fold.
5. Проверка portable inference на CPU и atomic publication модели/метрик.

Полные catalog logits `[batch,length,num_items]` не создаются. Во время
обучения рассчитываются только positive/negative scores. Scoring разбит на
query blocks, sampled embeddings извлекаются одним lookup, чтобы каждый block
не создавал отдельный dense gradient всей item table. Retrieval использует
users batches64 и item chunks32768, с фильтрацией seen до выбора top-k и
детерминированным item_id tie-break.

## Ресурсы

Лимиты учитывают50GiB RAM, RTX5070Ti16GiB и не более180GiB свободного G:.

| Ресурс | Ограничение benchmark |
|---|---|
| CPU | 8 threads, BLAS/MKL1 |
| RAM | RSS<=40GiB; старт available>=16GiB; остановка при available<4GiB |
| Epoch arrays | <=8GiB RAM; на диск negatives не записываются |
| GPU | один worker; allocator cap75%; старт free>=10000MiB; stop free<2048MiB |
| Собственные файлы run | максимум12GiB, включая work/checkpoint и временную запись |
| Linux free disk | старт>=25GiB; stop<10GiB |
| Windows G: free | старт>=70GiB; stop<50GiB |

Внешний supervisor проверяет ресурсы каждые2s, включая время нативных Torch
calls. Он останавливает worker через TERM, при необходимости KILL. Это не
kernel RSS cap: возможен короткий пик между проверками. CUDA allocator cap не
включает память driver/других процессов; поэтому дополнительно проверяется NVML.
Модель и checkpoints ожидаются существенно меньше12GiB; это верхний stop
budget, не предварительная резервация диска. Никакой старый artifact не удаляется.

В `.venv` уже установлен и проверен `torch==2.13.0+cu130` на этой RTX5070Ti.
Версия и CUDA wheel source записаны в requirements.txt; при подготовке ничего
не устанавливалось. Lightning/Optuna/FAISS на этом этапе не требуются.
GPU mode не переключается молча на CPU, если CUDA недоступна.

## Результаты и восстановление

- `artifacts/task15_sasrec_benchmark_v1/metrics.json`: epochs, train/data/step
  timing, memory peaks, retrieval throughput и оценка20epochs при том же recipe.
- `artifacts/task15_sasrec_benchmark_v1/config.json`: параметры, split, версии,
  hashes входов и implementation.
- `artifacts/task15_sasrec_benchmark_v1/model/`: portable weights/config/IDs.
- `artifacts/task15_sasrec_benchmark_v1/probe_candidates.parquet`: маленький
  retrieval probe, не submission.
- `artifacts/.task15_sasrec_benchmark_v1.work/`: sequence cache и последний
  optimizer/RNG checkpoint; после успешного run пока сохраняется для разбора.
- `logs/task15_sasrec_benchmark_v1.log`: plain-text log без ANSI, rotation
  1MB +2 backups. `*.resources.log`/`*.resources.json` содержат внешний monitor.

```bash
tail -F logs/task15_sasrec_benchmark_v1.log
cat logs/task15_sasrec_benchmark_v1.resources.json
kill -TERM "$(cat logs/task15_sasrec_benchmark_v1.pid)"
```

Ctrl-C также останавливает запуск. Повторить **ту же launch command** для resume:
завершённые epochs пропускаются; незавершённая epoch начинается с её исходного
RNG state. Сохраняются model/optimizer и Torch CPU/CUDA/Python RNG; window
sampling восстанавливается из seed/epoch. Partial files не помечаются готовыми.
При изменении config, implementation или input hashes нужен новый run ID.
Готовый published run защищён от overwrite, concurrent benchmark блокируется.

Проверка уже готового результата без обучения:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 \
  .venv/bin/python scripts/run_sasrec_benchmark.py \
  --verify-only artifacts/task15_sasrec_benchmark_v1
```

`estimated_20_epoch_training_seconds_same_shape` включает training и подготовку
epoch arrays, но исключает checkpoint I/O, validation retrieval и новые размеры
модели. Для smoke эта оценка null. `runtime_seconds_current_session` — время
текущей части запуска, а не сумма с паузами пользователя. Для Optuna отдельно
учтём стоимость оценок на next-day positives, размер user sample и pruning.

Обе `precision_at_20_*` и `candidate_recall` здесь **null**: будущие labels не
использовались. Benchmark не записывается как quality experiment в results.csv.
Лучший validation run остаётся Task13 D_all: P20all/labeled
`0.005077640992845437 / 0.006911912728855518`; это прежние метрики другого run.

## Проверки реализации

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_sasrec*.py' -v
bash -n scripts/run_task15_sasrec_benchmark.sh
./scripts/run_task15_sasrec_benchmark.sh --help
```

18 synthetic tests: causal prefix invariance, padding/empty/length validation,
last-valid state, finite loss/gradients и нулевой PAD, точная sampled loss,
uniform unseen negatives, UInt64 IDs, relevance60/61, seen/cold filtering,
dedup и обе P20, chunked vs brute-force retrieval, ties, portable restore,
resource limits, защита от смешивания mappings/folds, atomic epoch pause/resume
и independent CPU repeat. Ещё20существующих data/metrics/candidate tests прошли.

Limited GPU smoke (`task15_sasrec_gpu_smoke_final_v1`) на1024training users, полном
rolling_1 catalog и двух epochs по8batches прошёл. Steady step около15.7ms,
PyTorch peak allocated1.66GiB, reserved1.84GiB; external sampled peak device3692MiB,
RSS2.42GiB. Native top200 probe16users прошёл примерно за0.28s, seen pairs0.
Это стартовая калибровка; полный benchmark на всех users в сессии не запускался.
В раннем smoke_v1 поле20epoch estimate ещё относилось к его малому user sample;
в итоговом runner для smoke оно заменено на null.

Проверены real-data GPU pause после первой epoch/resume и независимые CPU/GPU
повторы: веса и candidates совпали точно. Это наблюдение на данном smoke,
а не обещание bitwise GPU reproducibility на другом hardware/version.
Все5smoke artifacts проверены повторным portable restore. Отчёт и воспроизводимый
read-only audit:`artifacts/task15_benchmark_preflight_v1/{config.json,metrics.json,review.py}`.

Предварительный ориентир для default benchmark — **3–6 минут**, на основании
короткого full-vocabulary smoke и верхней границы числа training users.
Это оценка: полная подготовка17.35m daily rows, длины histories, I/O и состояние
GPU ещё могут её изменить. После реальных трёх epochs ориентироваться на
`metrics.json`, а не на extrapolation маленького smoke.

## Файлы реализации

- `sasrec_data.py`, `sasrec_model.py`: sequence loader, модель, loss и retrieval.
- `scripts/run_sasrec_benchmark.py`, `scripts/task15_resources.py`,
  `scripts/run_task15_sasrec_benchmark.sh`: runner, monitor и исполняемый launcher.
- `configs/task15_sasrec_{benchmark,cpu_smoke,gpu_smoke}_v1.json`: три режима запуска.
- `tests/test_sasrec.py`, `tests/test_sasrec_benchmark.py`:18проверок.
- `requirements.txt`: закреплён существующий Torch CUDA build.
- `PROJECT_STATE.md`, `ROADMAP.md`, `SASREC_PLAN.md`, этот документ: handoff
  и согласованный порядок следующих этапов.
