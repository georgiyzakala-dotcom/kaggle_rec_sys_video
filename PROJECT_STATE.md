# PROJECT_STATE

Обновление2026-09-09: **Task15 expanded ranker завершён; подготовлена очистка диска**.
Artifact `artifacts/task15_sasrec_ranker_v1/`, submission200152x20:
`submission.csv` SHA256 `8c92e29077507dc87cf34fdcd0d925186c97b39a5b8e26a28c4c2bc4e8a4fe36`.
Новый лучший проверенный локальный результат (canonical уже открывался):

| Fold | P20 all targets | P20 labeled users | Hits | Delta all vs Task13 |
|---|---:|---:|---:|---:|
| rolling_3 | 0.007672918581877774 | 0.009515415499764554 | 30715 | +0.00040469243375035064 |
| canonical | 0.005333946200887327 | 0.007260806877227347 | 21352 | +0.00025630520804188935 |

Canonical +5.0477% relative, rolling_3 +5.5680%; future/leaderboard labels неизвестны.
Union recall прежний: r3=0.21576144264492378, canonical=0.2142813756669792.
Reproduction: `./scripts/run_task15_sasrec_ranker.sh` (completed не перезаписывается).
Verify: `.venv/bin/python scripts/run_expanded_ranker.py --verify-only artifacts/task15_sasrec_ranker_v1`.
Actual active runtime16480.28s (~4ч35мин), RSSpeak33.53GiB,
deviceVRAMpeak12186MiB. Обе full validation rows уже в experiments/results.csv.
Полный verifier повторно прошёл при dry-run cleanup: hashes, portable model,
P20 из сохранённых recommendations/GT, CSV round trip, seen/unknown/duplicate/
null/missing/extra=0. Новое обучение в этой сессии не запускалось.

Очистка: `./scripts/cleanup_task15_ranker.sh` (preview),
`./scripts/cleanup_task15_ranker.sh --apply` (только exact completed work).
Codex не удалял реальные пользовательские кэши и не останавливал WSL;
последующие пользовательские действия очистки здесь не перепроверялись.
Work содержит1689files,4.385GiBunique, но реально освобождаемыеfileblocks~0.254GiB:
4.133GiB связаны hard links с retained artifacts. Большие training pools/TSV
уже удалены runner; повторное rm не вернёт их размер ещё раз.
Output, raw data, остальные runs и parent fold/sequence caches сохраняются.
Cleanup проверяет completed artifact и owner/checkpoint, удерживает runner locks,
отказывает на symlinks/unknown root entries, имеет file progress и resume через
`artifacts/.task15_sasrec_ranker_v1.cleanup-trash` + audit receipt.
Отчёт применения: `artifacts/task15_sasrec_ranker_v1_cleanup_v1/`.

G: сейчас~114GiB free при~606GiB free внутри Linux; Windows registry подтвердил
Ubuntu/WSL2 и `G:\WSL\Ubuntu\ext4.vhdx` (~415.12GiB file length).
`scripts/compact_wsl_disk.ps1` по умолчанию preview; -Apply из локальной
Windows-копии в elevated PowerShell делает fstrim, shutdown всех WSL и compact
отключённого VHDX. Выполнять вручную после сохранения/закрытия Codex/WSL/Docker.
Пользовательский -Apply 2026-09-09 остановился на `execvpe(fstrim)` до trim,
shutdown и DiskPart. Исправлен вызов: `/usr/sbin/fstrim` вместо bare `fstrim`.
Добавлен `-CheckPrerequisites`: только запуск `--version` через WSL.
Из Windows реально проверены executable probe и fstrim --dry-run; оба прошли.
Нужно заново скопировать helper в Windows `%TEMP%` с `Copy-Item ... -Force`.
Команды copy/check/apply и monitoring: `DISK_CLEANUP.md`.
Native compaction не тестировалась на реальном диске: она остановила бы эту сессию.

Проверки helper:11synthetic tests, Ruff, bash syntax/help, реальный dry-run.
Disposable hard-link copy CPU artifact прошла cleanup --apply с полным verifier
до/после и повторным no-op, оригинальные artifacts сохранены;
отчёт `artifacts/task15_ranker_cleanup_smoke_v1/`.
Windows PowerShell Parser/реальный preview/ошибочный VHDX guard passed.
`tests/test_compact_wsl_disk.ps1` passed: preview не вызывает WSL,
probe использует абсолютный путь и --version, ошибка executable блокирует работу;
WSL/DiskPart в regression подменены. Настоящие trim/shutdown/compact не запускались.

Далее: (1) пользователь при необходимости удаляет exact work;
(2) закрывает WSL и выполняет Windows compact; (3) проверяет свободное место G;
(4) оценивает новый submission в leaderboard по собственному решению.

**Предыдущий handoff: подготовка runner до полного запуска.**

Обновление 2026-09-08: **готов runner ранкера на ALS600/SASRec600 и нового submission**.
Команда для пользователя: `./scripts/run_task15_sasrec_ranker.sh`.
Config: `configs/task15_sasrec_ranker_v1.json`; инструкция: `SASREC_RANKER.md`.
Output: `artifacts/task15_sasrec_ranker_v1/submission.csv` и `metrics.json`.
Полный запуск НЕ выполнялся; best validation остаётся Task13 D_all:
canonical P20all/labeled `0.005077640992845437 / 0.006911912728855518`.
Его reproduction: `./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.

Новый pipeline: global/recency/item2item200 + ALS600 + SASRec600, cap1800,
232features (D_all221 + SASRec11, включая cross scores для всех union pairs).
Три CatBoost fits по1030trees/depth7/seed42: r1+r2 -> r3;
r1+r2+r3 -> canonical; все4folds -> future. В каждом fit ожидаемые30Mrows,
все positive hits + deterministic uniform negatives, IPW1/p, свежие32borders
только по training rows. Full-query ranks считаются до negative sampling.
Новый sampler отличается от Task07 hard-negative sampler; сравнивается весь pipeline.
Future: четыре готовые full-history models/lookups/profiles Task14 переиспользуются
по SHA256, новый SASRec обучается12epochs на полной общей history Task14.
CV/GT не используются для fit candidates или выбора числа деревьев ранкера.
Обе P20 на r3/canonical сравниваются с Task13 на полном target universe;
CSV создаётся и при отрицательной delta, `improved_on_both_validation_folds`
фиксирует результат. Автоматической отправки в Kaggle/promotion нет.

Ресурсы: users_per_shard256, CPU8/BLAS1, RSS<=40GiB, GPU fractions0.7;
G start>=155GiB/stop<50GiB, TSV<=60GiB. Кэш каждого fit удаляется после
атомарного completed phase; портативные модели и оценки остаются.
Ожидание6–9часов, дополнительное пиковое место80–100GiB (экстраполяция).
VHDX не обязательно уменьшается после unlink; supervisor следит также за G.
Log: `logs/task15_sasrec_ranker_v1.log`; resource status: `.resources.json`.
Stop: `kill -TERM "$(cat logs/task15_sasrec_ranker_v1.pid)"`; resume той же
командой без изменения code/config. `--stop-after-phase rolling_validation`
позволяет сначала остановиться на завершённой оценке r3.
Verify: `.venv/bin/python scripts/run_expanded_ranker.py --verify-only artifacts/task15_sasrec_ranker_v1`.

Проверки: 7 новых synthetic tests +9ranker_backtest tests, Ruff, shell syntax/help.
GPU end-to-end smoke `task15_sasrec_ranker_gpu_smoke_v1`:16target users,
три модели по12trees, SASRec1epoch/32training users с full catalog,
263.96s including publication, RSS6.95GiB, deviceVRAM6175MiB.
CSV16x20, seen/unknown/duplicate/null/missing/extra=0; standalone verifier passed.
CPU repeated rolling smoke: features/scores/recommendations совпали точно;
resume completed phase после удаления training cache пропускает обучение.
CPU end-to-end `task15_sasrec_ranker_cpu_smoke_v5` (151.57s active): stop после
rolling_validation и resume до CSV, fit rolling_validation ровно1раз;
отдельный verifier passed. Scores всех3ранкеров и final recs точно совпали
с CPU v4. Smoke configs: `configs/task15_sasrec_ranker_{gpu_smoke_v1,cpu_smoke_v5}.json`.
Bounded timing без fit: `artifacts/task15_sasrec_ranker_timing_v1/` (256users):
train features1.61s, evaluation features1.29s, score1030trees0.49s,
production retrieval2.96s +cross scores0.62s +features0.78s, RSS6.98GiB.
Timing использует old221-feature ranker и smoke SASRec той же архитектуры;
полный232-feature quality/resource result остаётся неизвестным.
В ранних development smokes исправлены callback progress и legacy Task02
config layout; старые smoke artifact/log IDs не являются полными экспериментами.

Далее: (1) пользователь запускает full runner; (2) проверить P20/delta обоих
folds и runtime/resources; (3) принять решение о новом submission по результатам;
(4) при регрессии отдельно проверить ALS600+остальные200 и sampler/ranker tuning.

**Предыдущий handoff: проверка полного top600.**

Обновление 2026-09-08: **полный top600 выполнен пользователем; union recall проверен**.
Artifact: `artifacts/task15_sasrec_top600_v1/`; policy `als600_sasrec600_1800`:
ALS600 + SASRec600 + global/recency/item2item по200. Unique пары считаются один
раз, seen/cold GT удалены. Во всех4фолдах полные200152target IDs и coverage1.0.

| Fold | Positive hits / eligible pairs | Micro recall | Mean unique candidates |
|---|---:|---:|---:|
| rolling_1 | 292668 / 1839297 | 0.15911948967458764 | 1382.82996 |
| rolling_2 | 315415 / 1703659 | 0.18513974921037601 | 1429.57406 |
| rolling_3 | 346416 / 1605551 | 0.21576144264492378 | 1462.08955 |
| canonical | 275169 / 1284148 | 0.2142813756669792 | 1442.34763 |

Canonical оба200/оба300/оба600: recall0.13248862280671697 /
0.1572824939181465 / 0.2142813756669792. Прирост600vs200=105034positive hits,
8.179275п.п.,61.73568% relative; сумма caps1000 ->1800. При бюджете1200
ALS600-only recall0.16668172204449955 превосходит оба300; так во всех4folds.
«Only» относится к двум embedding sources; остальные3x200 включены.
Overlap600 canonical: доля списка SASRec в ALS0.2256369221, Jaccard0.1271155950;
shared hits73961, SASRec-only70566, ALS-only101989. Это overlap двух источников,
не уникальность SASRec относительно всех пяти.
Oracle P20 all/labeled0.06829884287941165/0.09297144916891102;
actual P20 нового ранкера=null: ranker не обучался, promotion=false.
Rolling_1 selection diagnostic, canonical previously opened, один seed/короткая timeline.

Runtime7822.41s (около2ч10мин), fit=0, RSSpeak4.55483GiB, VRAM4294MiB.
GPU SASRec/CPU8 ALS, прежние weights и daily histories; parent training не повторялся.
Финальные SHA256/агрегация прошли CLI verifier; independent sums per-user counts
совпали точно, target universe и denominator проверены по parent GT.
Повторный расчёт: `artifacts/task15_top600_recall_review_v1/review.py`;
рядом config.json, metrics.json, recall.csv, report.md и manifest.json.
Четыре full top600 строки уже в experiments/results.csv; verify не дублирует их.
Инструкция и сводная таблица: `SASREC_TOP600.md`.
Reproduction: `./scripts/run_task15_sasrec_top600.sh` (completed не перезаписывается).
Verify: `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 .venv/bin/python scripts/run_sasrec_top600.py --verify-only artifacts/task15_sasrec_top600_v1`.

Best ranker validation по-прежнему Task13 D_all: P20all/labeled
0.005077640992845437/0.006911912728855518; команда:
`./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.
Для следующего ranker backtest пользователь выбрал оба600+остальные200;
runner подготовлен, см. актуальный handoff выше. ALS остаётся более сильным
источником при проверенных равных бюджетах; это не мешает проверить расширенный union.

**Предыдущий handoff: подготовка top600 до пользовательского полного запуска.**

Обновление 2026-09-08: **полные SASRec folds проверены; top600 runner готов**.
Пользователь попросил остановить top300 и самостоятельно запустить расчёт
ALS@600/SASRec@600. Top300 остановлен штатно (status=paused):97atomic shards
rolling_1 в `artifacts/.task15_sasrec_top300_v1.work/`, полного результата нет.
Полный top600 в сессии НЕ запускался. Следующая команда для пользователя:
`./scripts/run_task15_sasrec_top600.sh`.
Конфигурация: `configs/task15_sasrec_top600_v1.json`; инструкция: `SASREC_TOP600.md`.

Новый top600 runner использует готовые SASRec GPU/ALS CPU models, fit=0,
все200152target users каждого rolling_1/2/3/canonical. Первые200 SASRec и400 ALS
сверяет точно с saved predictions; старые per-user metrics пересчитывает точно.
Global/recency/item2item остаются200. Сохраняет source depths50/100/150/200/300/
400/600, union policies в том числе оба600, оба300 и ALS600-only control при
том же бюджете1200, raw/positive overlap200/300/400/600, обе oracle P20 и
прежние standalone P20. Ранкер не обучается, primary unranked union P20=null.
Новые `sasrec_top600.py`, `scripts/run_sasrec_top600.py`, launcher/configs/tests
используют проверенный `sasrec_top300.py` для GPU retrieval и старых метрик.
Старые model/runner файлы не менялись при подготовке top300/top600.

Ориентир2,5–3часа на текущем железе: GPU smoke2048users/fold дал21,12/22,45/
23,54/24,48s на блок, экстраполяция392блоков2,49часа. Smoke120,40s,
RSSpeak3,88GiB/external VRAM4262MiB; projected predictions+stats5,3GiB.
CPU8/BLAS1, RSSmax40GiB, CUDA allocator75%, run filesmax30GiB;
G: free start80/stop50GiB. Atomic shards/folds + checkpoint identity, общий
Task15 lock, compact logs/ресурсы и progress/ETA fold/users/SASRec/ALS batches.
Output: `artifacts/task15_sasrec_top600_v1/{metrics.json,comparison.csv,
source_depth_metrics.csv,overlap_metrics.csv}`; parquet candidates в
`folds/<fold>/parts/part-*/{sasrec600.parquet,als600.parquet}`.
Мониторинг: `tail -F logs/task15_sasrec_top600_v1.log`.
Stop: Ctrl-C или `kill -TERM "$(cat logs/task15_sasrec_top600_v1.pid)"`;
resume той же командой. Completed artifact не перезаписывается.
Verify: `.venv/bin/python scripts/run_sasrec_top600.py --verify-only artifacts/task15_sasrec_top600_v1`.

Прошли28tests (top300/top600/data/metrics/ALS), Ruff, --help, bash syntax,
GPU smoke на всех4фолдах и отдельный portable CLI verifier. Tests запрещают
fit обеих моделей и проверяют pause/resume/точный repeat, UInt64/empty GT,
relevance/dedup/seen/cold и оба знаменателя P20. Исходный GPU predict600 для
64users/fold точно совпал с optimized saved candidates по IDs/ranks/scores. Preflight:
`artifacts/task15_top600_preflight_v1/{review.py,config.json,metrics.json}`.
Smokes не пишутся в experiments/results.csv; полный top600 добавит4rows.

Пользовательский `artifacts/task15_sasrec_folds_v1` прошёл verifier;
runtime11846,36s (3ч17мин),4full targets folds. Frozen Optuna trial12/epoch12:
d128/L100/1block/2heads/dropout0.2/batch128/negatives32/BF16/AdamW/seed42,
lr0.0014935660033749737, weight_decay0.0002979293642858734.
Rolling_1 reuse winner, остальные3folds fresh12epochs. Optuna выбирала recipe
по8192target IDs rolling_1, поэтому rolling_1 selection diagnostic;
canonical previously opened, one seed/short timeline, promotion=false.

Canonical micro recalls: SASRec@200=0.053030491812470215,
ALS@200=0.0674260287754994, baseline800=0.107002463890455,
blend150/50=0.1045284499917455, replaceALS=0.09966608210268599,
both200/1000caps=0.13248862280671697, ALS400/1000caps=0.1400352607331865.
ALS400 превосходит оба200 при том же бюджете1000 во всех4folds;
frozen blend150/50 ухудшает baseline, mean delta rolling_2/3=-0.0019218956.
Canonical SASRec добавляет32728hits сверх всехстарых sources; у ALS остаётся
42149hits сверх других+SASRec. Overlap200: средняя доля SASRec в ALS0.16997318,
micro Jaccard0.09284515; shared27658, SASRec-only40441, ALS-only58927.
Standalone SASRec P20all/labeled0.0030479335704864303/0.0041489839223047414;
ALS0.003859816539430033/0.0052541554449250525. Итоги: `SASREC_FOLDS.md`.

Best ranker validation остаётся Task13 D_all: P20all/labeled
0.005077640992845437/0.006911912728855518; воспроизведение:
`./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.
Следующие шаги: пользователь запускает top600; проверить full artifact и
recall300/600/overlap; сравнить оба300 с ALS600 при1200; для перспективного
состава отдельно подготовить matched Task13 ranker backtest до promotion.

**Предыдущий handoff: исправление pruning до завершения пользовательской Optuna.**

Обновление 2026-09-07: **исправлен сбой Optuna при pruning, пользовательская study восстановлена**.
Продолжить: `./scripts/run_task15_sasrec_optuna.sh` с прежними config/run_id.
В `artifacts/.task15_sasrec_optuna_v1.work/study.sqlite3` сохранены 10 COMPLETE
trials и trial №10 PRUNED; следующий trial №11 (нумерация с нуля).
Причина сбоя: runner передавал `values` в `study.tell(..., state=PRUNED)`;
Optuna запрещает значения для PRUNED/FAIL. Теперь COMPLETE получает objective,
а PRUNED использует последний `trial.report`. Добавлены regression tests для
всех трёх состояний, принудительного pruning и восстановления без повторного fit.

Trial №10 восстановлен из его атомарного result/checkpoint и intermediate values.
Проверены SHA256 136 защищённых файлов: веса, candidates, метрики, frozen config
и timing не изменились; active time 9313.956607696004s сохранено. Fit не запускался.
SQLite до/после, dry run и отчёт: `artifacts/task15_optuna_pruning_fix_v1/`.
Разрешён только проверенный переход SHA256 самого runner через
`configs/task15_sasrec_optuna_resume_compatibility.json`; остальные config/input/
implementation hashes обязаны совпадать. Исходный config digest сохранён,
совместимая правка записана в `runtime_patch.json` и попадёт в final artifact.
Проверены 48 tests, Ruff, launcher `--help` и `bash -n`.

Промежуточный лучший результат: trial №8, epoch12, policy blend_150_50,
objective -0.0010798709422532443; улучшение union recall пока не получено.
На selection sample rolling_1 (8192 target IDs) native SASRec recall@200
0.03542503456903931; P20all/labeled 0.0033630371093750005/0.004139744552967694.
Это незавершённый подбор, не canonical/ranker validation. Best validation
остаётся Task13 D_all: P20all/labeled 0.005077640992845437/0.006911912728855518;
команда воспроизведения: `./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.
Следующие шаги: пользователь продолжает search; проверить опубликованный best
recipe/метрики; подготовить frozen training на остальных rolling и сравнение с ALS.

**Предыдущий handoff: Optuna preflight до пользовательского полного запуска.**

Обновление 2026-09-07: **пользователь выполнил SASRec benchmark; Optuna runner готов**.
Следующая команда: `./scripts/run_task15_sasrec_optuna.sh`.
Конфигурация:`configs/task15_sasrec_optuna_v1.json`; инструкция:`SASREC_OPTUNA.md`.
Full Optuna в сессии НЕ запускался. После его результата подготовить отдельный
скрипт обучения frozen recipe на остальных rolling и сравнения с ALS.

Benchmark `artifacts/task15_sasrec_benchmark_v1` повторно проверен:3epochs,
346,947training users, catalog1,337,354; runtime164.83s, median epoch с windows
49.77s, train42.37–43.28s +data7.20–7.34s, steady15.4ms/batch128.
Peak RSS2.89GiB, torch allocated1.66/reserved1.84GiB, device sampled3700MiB.
Retrieval128users0.604s; loss0.902/0.593/0.477. Quality metrics=null.

Optuna:24sequential trials, максимум12epochs, evaluate каждые2epochs на
фиксированных8192target IDs по hash(seed42), все empty/unknown-history users
сохраняются. Fit — все eligible history users, full catalog; fresh weights
seed42 каждого trial, reuse только history-only sequence cache benchmark.
TPE перебирает d32/64/128, L25/50/100, blocks1/2, dropout0/.1/.2/.3,
negatives32/64, log learning_rate[1e-4,3e-3]/weight_decay[1e-6,1e-3].
MedianPruner после6completed trials, warmup4epochs; patience3evaluations;
soft total limit8active hours, ориентир4–8h. Best epoch фиксируется по labels.

Objective: прирост micro recall union при сумме source caps800. Сравниваются
replace_als и blends ALS/SASRec150/50,100/100,50/150; остальные3sources по200.
Add_source1000 только diagnostic. Сохранены native recall/coverage/counts,
oracle P20, raw/positive overlap со всеми4sources и top20 P20all/labeled с
независимым global fallback. Это selection sample, не canonical/ranker quality.
Labels только rolling_1 `[2024-11-29T08:56:28,2024-11-30T08:56:28)`;
source tables reuse `artifacts/task06_candidate_datasets_v1/folds/rolling_1`.

Новые `sasrec_selection.py`, `scripts/run_sasrec_optuna.py`, launcher/configs/tests.
Установлена необходимая `optuna==4.9.0`, записана в requirements.txt; существующие
Torch/NumPy/Polars не обновлялись, pip check прошёл. Проверены45tests, Ruff,
help/bash syntax. CPU smoke:2trials/2epochs,32training users,8eval users;
epoch pause/resume. GPU smoke:2trials/2epochs,512training users,16eval users,
d128/L100/full catalog; pause после trial0/resume не повторяет его fit.
GPU torch peak3.28GiB, external device5550MiB/RSS2.76GiB. Smokes не доказывают
качество, их обе P20=0 и standalone recall=0; запись results.csv не создавалась.
Artifact review:`artifacts/task15_optuna_preflight_v1/{review.py,config.json,metrics.json}`.
Все3Optuna smoke artifacts и пользовательский benchmark прошли audit.
Independent real-data CPU repeat (`task15_sasrec_optuna_cpu_final_v1`) точно
совпал с resume run по params/weights/candidates. Oracle policy tie-break
использует сумму целых clipped hits: устранено влияние1ULP parallel means.

Ресурсы:CPU8/BLAS1, RSS40GiB/available stop4, GPU allocator75%,
freeVRAM start10000/stop2048MiB; собственные run files<=30GiB,
Linux free start40/stop10GiB, Windows G:free start80/stop50GiB.
SQLite study +atomic optimizer/RNG epoch checkpoints; завершённые trials
не повторяются; сохраняется один best model/trial и активный optimizer.
Результат:`artifacts/task15_sasrec_optuna_v1/{best_recipe.json,metrics.json,model/}`.
Мониторинг:`tail -F logs/task15_sasrec_optuna_v1.log`; остановка Ctrl-C или
`kill -TERM "$(cat logs/task15_sasrec_optuna_v1.pid)"`; resume той же командой.
Verify:`.venv/bin/python scripts/run_sasrec_optuna.py --verify-only artifacts/task15_sasrec_optuna_v1`.
Best validation по-прежнему Task13 D_all, P20all/labeled
0.005077640992845437/0.006911912728855518; Task14 submission не менялся.

**Предыдущий handoff: подготовка benchmark, до пользовательского запуска.**

Обновление 2026-09-07: **Task15 GPU benchmark готов к пользовательскому запуску**.
Команда: `./scripts/run_task15_sasrec_benchmark.sh` (конфигурация
`configs/task15_sasrec_benchmark_v1.json`). Это одно обучение,3epochs на всей
history rolling_1, d64/2blocks/2heads/L50, batch128/64negatives, BF16/AdamW.
Full benchmark в сессии НЕ запускался. По запросу пользователя Optuna готовится
отдельно после его замера, selection только rolling_1/target следующего дня;
после freeze — обучение на остальных rolling и сравнение recall/ALS overlap.
Подробная инструкция: `SASREC_BENCHMARK.md`; первоначальный план и критерии
сравнения: `SASREC_PLAN.md`. Статус Task15 `[~]`, качество ещё не измерено.

Новые `sasrec_data.py`/`sasrec_model.py` используют общие Candidate interfaces,
immutable daily history, deterministic UInt64/Int32 mappings, right padding,
last-valid selection, PAD=0, pre-norm/final LayerNorm, tied embeddings и
sampled BCE. Полные catalog logits при обучении не создаются. Epoch — один
случайный endpoint/window на eligible user, а не полный перебор всех prefixes.
Негативы исключают все seen history pairs; retrieval exact/chunked, tie-break
по item_id. Portable model хранит history SHA, loader отвергает другой fold.
Обычный PyTorch достаточен; уже установленный torch2.13.0+cu130 закреплён в
requirements.txt и проверен на RTX5070Ti. Зависимости не устанавливались.

Runner: `scripts/run_sasrec_benchmark.py`, внешний supervisor
`scripts/task15_resources.py`. Progress/ETA epochs/windows/batches, plain rotating
logs, atomic optimizer/RNG checkpoints и resume на границе epoch. Лимиты:
CPU8/BLAS1, RSS40GiB/available stop4, CUDA allocator75%/freeVRAM stop2048MiB,
run files12GiB, Linux free start25/stop10GiB, Windows G:start70/stop50GiB.
Epoch negatives находятся в RAM, budget8GiB; history/validation не изменяются.
Вход:`artifacts/task02_fold_20241129_v1/history_daily.parquet`, cutoff
2024-11-29T08:56:28, next-day end2024-11-30T08:56:28. Labels не читаются.

Проверены38tests (18SASRec +20data/metrics/candidate), Ruff, --help и bash syntax.
Итоговый real-data GPU smoke:`artifacts/task15_sasrec_gpu_smoke_final_v1`,
1024training users/2048context, весь catalog1,337,354items,2epochs по8batches.
Steady step15–16ms, torch peak allocated1.66GiB, external sampled RSS2.42GiB /
device3692MiB. Independent CPU/GPU repeats и GPU pause/resume после epoch1
дали точное совпадение weights/candidates. Все5smoke artifacts прошли portable
verify; полная RAM/скорость всего training universe пока не измерены.
Audit:`artifacts/task15_benchmark_preflight_v1/{review.py,config.json,metrics.json}`.
Smokes не являются quality experiments и не записаны в results.csv.
Обе precision_at_20_* и candidate_recall=null; best остаётся Task13 D_all ниже.

После запуска читать `artifacts/task15_sasrec_benchmark_v1/metrics.json` и
`logs/task15_sasrec_benchmark_v1.log`; внешний monitor — одноимённые
`.resources.json`/`.resources.log`. Остановка Ctrl-C или
`kill -TERM "$(cat logs/task15_sasrec_benchmark_v1.pid)"`; resume той же командой.
Проверка без fit: `.venv/bin/python scripts/run_sasrec_benchmark.py --verify-only artifacts/task15_sasrec_benchmark_v1`.
Ориентир3–6min для3epochs предварительный; после полного benchmark использовать
его измерения для оценки Optuna. Полные fold quality metrics в этот этап не входят.

**Исторический planning review, до реализации:**

Bounded CPU review: `artifacts/task15_sasrec_review_v1/review.py`, `config.json`,
`metrics.json`. В присланных blocks left padding создаёт NaN на pad queries,
второй block распространяет их на valid states; Xavier перезаписывает PAD row,
right padding требует last-valid gather. Полный logits tensor B256/L50 при
canonical catalog1,810,005 занимает86.308GiB FP32; нужны sampled loss и chunked
retrieval. Проверены metadata и доступность всех16Task06source/model paths,
без повторного fit или полного пересчёта candidate metrics. На том этапе GPU
не проверялся из-за sandbox NVML restrictions; позже bounded GPU smokes выше
выполнены с разрешённым доступом. Два CPU review дали идентичный JSON;
20 существующих data/metrics/candidate tests прошли. Это planning review,
не запись results.csv. Команды проверок сохранены в конце `SASREC_PLAN.md`.

**Уточнение устаревшего Task14 handoff ниже:** full artifact
`artifacts/task14_full_fit_v1/` уже опубликован, submission существует,
его SHA256 совпал с metrics:
`9df8f9e554a8f69fabc7c234c87c88ae81c7cfc243be6074b446c27eeddc844d`.
Есть строка results.csv; runtime3678.74s, обе P20/recall=null, validation_reference
указывает Task13. Повторный полный semantic verifier и leaderboard не проверялись
в этой задаче. Task14 full-history weights не подходят для rolling evaluation.
Лучший validation результат по-прежнему Task13 D_all:
P20all/labeled `0.005077640992845437 / 0.006911912728855518`,20326hits.
ALS используется generator/cross-score/history profiles; решение убрать
ALS candidates не равнозначно отказу от обучения ALS.

Обновление 2026-09-05: **Task14 подготовлен для внешнего полного запуска**
по запросу пользователя: новый submission с лучшим рецептом Task13, с обучением
всех production моделей, а не только ранкера. Full run в сессии НЕ запускался.
Команда: `./scripts/run_task14_full_fit.sh configs/task14_full_fit_v1.json`.
Ожидаемый файл: `artifacts/task14_full_fit_v1/submission.csv`; пока его нет.
Инструкция: `PROFILE_FULL_FIT.md`, notebook
`research/14_full_fit_history_profiles.ipynb`.

Новый `scripts/run_profile_full_fit.py` переиспользует проверенные Task12/13
sampling/profile/quantization компоненты и Task11 production pipeline. Новый
CatBoost D_all:221features/1030trees/fresh32borders, примерно30m training rows
с IPW и всеми positives, теперь из rolling_1+rolling_2+rolling_3+canonical.
Historical Task06/07 fold models/candidates/features переиспользуются;
profile scalars пересчитываются строго из ALS/history своего фолда.
Для production все raw events агрегируются в новый full-history snapshot;
global/recency/item2item/ALS обучаются с нуля, пересчитываются201features и20
profile scalars из нового полного ALS. Full-history ALS в training folds
не применяется. Кандидаты200/source,total800; inference2048users/shard.

Лимиты прежние: CPU8, BLAS1, RAM RSS40GiB, availableRAM start38/stop4GiB,
GPU0/gpu_ram_part0.70, freeVRAM start13000/stop2048MiB;
Linux free start110/stop15GiB, G:start160/stop50GiB. Внешний supervisor2s,
progress/ETA, native heartbeat30s, rotating logs, checkpoints и snapshots.
Оценка времени1.5–2.5h, прирост диска60–100GiB. G: после подготовки около207GiB
free; дополнительная очистка или diskpart для старта сейчас не требуется.
Fused candidate/features/inference сохраняет top20/diagnostics по шардам;
полная production feature matrix на диске не создаётся.

Проверены165unit tests, Ruff/help/bash syntax, real-data CPU smoke64users/12trees,
pause/resume после первого inference shard (ranker и все4candidate fits skipped),
независимый CPU repeat с точным совпадением CSV SHA256
`f064596bff1e05e3e3379929a6fc3998415cc15961f449a998062f97a15a9f23`.
Artifacts:`task14_full_fit_cpu_smoke_v2`, `task14_full_fit_cpu_repeat_v2`.
GPU smoke64users/12trees/depth7 также прошёл с прежним supervisor:
`task14_full_fit_gpu_v2`, sampled peak device VRAM3935MiB. Все три artifacts
прошли full top20 semantics, source-fit provenance и portable/profile verify.
На CPU repeat искусственно остановлена публикация после полного staging;
CLI `--publish-only` успешно завершил её без fit/predict, сохранив training hashes.
Начальный smoke_v1 поймал ошибку записи списка в dict-only JSON writer;
исправлено до передачи full runner. Промежуточная проверка конфигурации окон
также исправлена: используются действительные поля HistoryFeatureConfig.
Smokes не являются quality experiments и не записываются в results.csv.
Отчёт проверок:`artifacts/task14_preflight_review_20260905_v1/metrics.json`.
После verify удалены только созданные в этой задаче восстановимые work caches
четырёх smoke attempts:12.38GiB. Published CPU/GPU artifacts сохранены;
config/checkpoint/timing/operation metadata архивированы в review. Логи не удалены.

После внешнего запуска: проверить artifact через `--verify-only`, сравнить CSV
с Task11 и просмотреть full-history/shard/resource diagnostics; затем пользователь
сможет загрузить новый CSV на Kaggle. Рецепт имеет canonical P20all/labeled
0.005077640992845437/0.006911912728855518, но у full refit нет будущих labels:
его P20 и candidate_recall записываются как null, метрики Task13 отдельно в
validation_reference. Измерение leaderboard и устойчивости по дням остаётся
следующим шагом. User Makefile не изменён; зависимости не добавлены.

Обновление 2026-09-05: **Task13 завершён и проверен**, новый best canonical run —
`artifacts/task13_history_profiles_v1`, вариант `D_all`, 1,030 trees, 221 features.
P20 all/labeled `0.005077640992845437 / 0.006911912728855518`, 20,326 hits:
+281 (+1.4018%) к Task12 и +543 (+2.7448%) к matched A_base control30m.
Исходная команда пользовательского full run:
`./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.
Описание: `HISTORY_PROFILES.md`; команды Windows/DiskPart: `DISK_CLEANUP.md`.
Runtime 4,464.56s (74.41min), peak RSS31.62GiB. Прирост измерен на прежнем
canonical дне; устойчивость по дням/повторным GPU fits ещё не проверена.
Submission Task11 пока остаётся прежним.

На публикации full run упал из-за точного сравнения float: labeled score
0.006911912728855518 против 0.0069119127288555186, разница1ULP (8.67e-19).
Исправлен verifier: integer hits/target/labeled counts проверяются точно,
P20 сверяется с `hits/(20*users)` с допуском32ULP. Training/features не изменены.
Добавлен CLI `--publish-only`, который проверяет полностью готовый staging,
его связь с исходными checkpoints/frozen winner и атомарно завершает публикацию.
Восстановление выполнено за1.56s **без fit и без regeneration predictions**:
`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 .venv/bin/python scripts/run_history_profiles.py --config configs/task13_history_profiles_v1.json --publish-only`.
Исходные config/implementation hashes и метрики сохранены, новый verifier
отдельно записан в artifact/publication_recovery.json. CSV содержит одну строку.
Проверка готового результата:
`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 .venv/bin/python scripts/run_history_profiles.py --verify-only artifacts/task13_history_profiles_v1`.
Повторять full command для этого run ID не нужно: artifact уже опубликован.

Recovery review: `artifacts/task13_publication_review_20260905_v1/metrics.json`.
Независимо пересчитаны обе P20 и counts всех четырёх rolling_3 вариантов и
canonical winner/control; проверены checksums, frozen selection, exact top20,
известные unseen items и portable repeat. Пройдено158tests, включая округление,
отклонение неверных counts/метрик, no-fit publication recovery и отказ при
corrupted model; три прежних real-data smoke artifacts прошли новый verifier.
Rolling hits A/B/C/D:28035/29077/29069/29095. Все profile variants лучше A;
преимущество D над B — только18hits, отдельный вклад каждого event-profile пока
не установлен. Canonical matched A:19783hits, P20all/labeled
0.0049419940844957835/0.006727264071383879. Full work cache сохранён для анализа.

Task13 добавляет 20 скаляров из четырёх ALS-профилей (positive/like/favorite/
long_watch), mean unit item vectors по уникальным items, без передачи координат
в CatBoost. A/B/C/D сравниваются на всех target users rolling_3 при фиксированных
1,030 trees. Общие строки/IPW/train-only borders=32 и единый Pool на 221 feature;
группы переключаются через ignored_features. После freeze winner: canonical
refit и отдельный matched A_base control. Full training budget 30m вместо 46m
из-за памяти; сравнение с Task12 включает изменение sampling.

Новых численных лимитов пользователя в доступных файлах не найдено; вопрос о
лимитах был отправлен, ответа на момент подготовки нет. Приняты параметры:
CPU8/BLAS1, GPU0 gpu_ram_part=0.70; preflight available RAM>=38GiB,
VRAM>=13000MiB, Linux free>=110GiB, G: free>=160GiB; внешний supervisor
останавливает worker при RSS>40GiB, available RAM<4GiB, free VRAM<2048MiB,
Linux free<15GiB или G: free<50GiB. Poll2s; TERM/KILL работает даже во время
нативных CatBoost calls. Возможен короткий пик между опросами; kernel cap нет.
Первоначальная оценка full runtime была2–3h, резерв4h; фактически74.41min.

Очистка: штатный Task11 verify+cleanup подтвердил все checksums, portable model,
submission roundtrip и full top20, затем удалил `.task11_full_fit_v1.work`
(70.31GiB). Published Task11 и все входы сохранены. Отчёт:
`artifacts/task13_disk_cleanup_20260905_v1/`. Windows G: до compaction имеет
109.01GiB free, Linux deletion не уменьшил VHDX. TRIM выполнен, WSL не остановлен;
команды compaction оставлены пользователю, иначе завершится Codex-сессия.
**Не удалять `.task05_implicit_als_v1.best-model`: Task06 rolling_3 ссылается на
модель в versions/ этой директории.**

До полного запуска Task13 был проверен: 152 tests, Ruff, shell syntax/CLI help, CPU pause/resume,
независимый CPU repeat (точное равенство всех четырёх rolling и canonical top20),
GPU smoke 64 users/12 trees/depth7. Artifacts:
`task13_history_profiles_smoke_v1`, `task13_history_profiles_cpu_repeat_v1`,
`task13_history_profiles_gpu_smoke_v1`; все три прошли portable verify и full
top20 semantics для своего smoke universe. CPU P20 all/labeled
0.0023437500000000003/0.003488372093023256, GPU
0.0015625/0.002325581395348837; это не quality experiments и не CSV full runs.
Диагностика: `artifacts/task13_preflight_review_20260905_v1/` — покрытие history
positive 99.93%, long_watch 99.48%, like15.74%, favorite9.42%; benchmark200k
candidate rows, ~257650rows/s. Candidate recall по существующему union10.70%.
Пользовательский `Makefile` не изменён. Новые зависимости не добавлялись.
После проверки удалены только восстановимые work caches трёх Task13 smoke
запусков (~9.3GiB); их published artifacts и verification.json сохранены.
После подготовки free был Linux710.36GiB, G:109.01GiB; в конце full run
Linux677.15GiB, G:207.61GiB. TRIM отметил458.3GiB свободных extents;
это не объём дополнительно удалённых данных и не обещание прироста места G:.

Исторический результат: 2026-09-05 после проверки full Task12. Тогда лучшим
canonical run стал `task12_ranker_backtest_v1`: P@20 all/labeled
`0.005007444342299852 / 0.006816357898745886`, 20,045 hits. Прирост над Task08
+877 hits (+4.5753%). Победитель `fit_training`, 1,030 trees; это validation
baseline дальнейшей работы, а текущий competition submission остаётся Task11.
Исходная команда выполненного пользователем run:
`./scripts/run_task12_ranker_backtest.sh configs/task12_ranker_backtest_v1.json`.
Runner сравнивает frozen/fresh quantization при явных 32 borders и выбирает
tree count по P@20 на полных candidate lists 16,384 ID-sampled users rolling_3.
Selection train = rolling_1+rolling_2; после фиксации winner refit на
rolling_1+rolling_2+rolling_3 и одна canonical оценка. Candidates и 201 features
переиспользуются из Task07. Full runtime 4,970.11 s (82.84 min), peak RSS
47,074.45 MiB (45.97 GiB). Task12 status `[x]`; запись в results.csv уже
создана runner, review не дублирует её. Полный artifact защищён от overwrite.
Полная инструкция, ресурсы, log/output/stop/resume: `RANKER_BACKTEST.md`.
Независимый review: `artifacts/task12_review_20260905_v1/findings.md`,
`metrics.json`, `review.py`. Проверены 27 file checksums, обе метрики двух
folds, все top-20 semantics, portable probe и implementation hashes; 142 tests
прошли повторно. Canonical paired-user bootstrap 95% interval delta all:
`[0.0001743675, 0.0002642991]`. На 183,768 rolling_3 users вне selection
+242 hits (+0.95%). Bootstrap не учитывает смену дня, зависимость users и
повторные GPU fits; canonical уже использовался ранее.

Вклад исправлений полностью не разделён: fresh/frozen при 1,030 trees дали
2,223/2,195 selection hits, лучшие точки policies — 2,223/2,212. Помимо
квантования изменились folds и sampling. Старые 1,030 trees не оказались
ошибкой: P@20 выбрал их снова, против 1,000 trees выигрыш всего два hits.
User 72h share features в canonical refit получили 32 borders, деревья
используют 11/12 вместо 1/1 у Task08. Candidate recall остался 10.70%; доля
использованных oracle hits выросла до 14.6003%. Следующий этап — отдельные
ablations candidate-conditioned history features на том же candidate union.

Проверки Task12: 142 tests, включая synthetic end-to-end pause/resume и
контроль отсутствия canonical при selection. Real-data CPU smoke на 64 users,
максимум 12 trees, опубликован в `artifacts/task12_ranker_backtest_smoke_v1/`.
Resume пропустил 16 готовых операций. Независимый `--verify-only` подтвердил
checksums, portable inference, exact top-20 и обе метрики: smoke P@20
all/labeled `0.0023437500000000003 / 0.003488372093023256`, 3 hits, 43 labeled
из 64 target users. Runtime двух частей 7.42 s, peak RSS 956 MiB.
Smoke не добавлен в `experiments/results.csv` и не сравним с full runs.
Независимый CPU repeat `task12_ranker_backtest_cpu_repeat_v1` точно повторил
rolling/canonical recommendations и selection curves. Ограниченный GPU smoke
`task12_ranker_backtest_gpu_smoke_v1` (64 users, depth 7, максимум 12 trees)
прошёл с выбранными 4 trees: P@20 all/labeled
`0.0046875 / 0.006976744186046512`, 6 hits, 7.37 s, peak RSS 1,194 MiB.
Его первый preflight внутри sandbox получил NVML access denied; тот же
bounded run с доступом к GPU успешно завершён. Оба дополнительных artifacts
прошли независимый `--verify-only`. Shell syntax, CLI help, Ruff и code cells
нового notebook проверены. Эти проверки предшествовали полному Task12.

Final artifact
`artifacts/task11_full_fit_v1/` уже опубликован: четыре training folds,
201 features, fixed 1,030 trees, новый full-history fit candidate models.
Пользователь сообщил второе место на соревновании; точный внешний score пока
не сохранён. Task08 — прежний canonical baseline с P@20 all/labeled
`0.004788360845757224 / 0.006518131614026497`; теперь его превзошёл Task12.
Task11 нельзя оценивать на canonical: эти labels вошли в final training.
Исходная команда final run:
`./scripts/run_task11_full_fit.sh configs/task11_full_fit_v1.json`;
существующий artifact защищён от overwrite. Исторический статус Task11 в
ROADMAP/results.csv ещё не сверялся заново полным `--verify-only`; наличие
его финальных файлов и метрик подтверждено аудитом.

Исторический аудит до Task12: `artifacts/repository_audit_20260904_v1/findings.md`,
`config.json`, `metrics.json`, воспроизводимый `audit.py`. В рамках аудита
model fit не было; последующая проверка обучения описана выше.
133 unit tests прошли; timestamps проверены на 1,024 sampled users,
Task08/09 hits независимо пересчитаны из saved predictions. Основные findings:

- Candidate recall только `0.107002463890455`; у 54.714% labeled users нет
  позитивов в union. Task08 использует 13.9615% доступных oracle hits.
- Final quantization не полностью frozen: отсутствующие в input_borders
  признаки достраиваются, `pipeline.py` не передаёт border_count. В production
  item_daily_row_share_72h имеет 36 используемых borders. Две user 72h share
  features сохраняют только threshold 0.5, хотя full-history target lookups
  содержат 16,342 / 17,855 различных значений. Влияние исправления на score
  ещё не измерено.
- dt=min сохраняет daily contract, но искажает last-event/6h features и seeds:
  80/1,024 users имеют неточный last interaction, у 47 меняется top-5 seed set.
  Потеря positive daily groups через 6h boundary в sample около 0.91%.
- Pointwise early stopping выбирает sampled Logloss, не P@20. LTR не передаёт
  object IPW при том же неравномерном sampling; проигрыш Task10 нельзя
  приписать исключительно objective. Multi-fold final recipe отдельно
  walk-forward не оценивалась.
- Выбор Task08 вместо rolling winner Task09 использует открытый canonical;
  разница 67 hits неотличима от нуля по приближённому paired-user 95% interval.
  Canonical остаётся известной диагностикой, не новым нетронутым holdout.

Историческая сводка на 2026-09-03: Задача 10 завершена full LTR selection и ровно одной
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

`task13_history_profiles_v1`, вариант `D_all`, 221features,1030trees,
fresh train-only32borders, около30m sampled training rows с positives/IPW.
Canonical P20all/labeled `0.005077640992845437 / 0.006911912728855518`,20326hits;
candidate recall `0.107002463890455`, union source200/total800 не изменён.
Selection train rolling_1+rolling_2, evaluation rolling_3; canonical refit
train rolling_1+rolling_2+rolling_3. Команда исходного run:
`./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json`.
Проверка готового artifact:
`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 .venv/bin/python scripts/run_history_profiles.py --verify-only artifacts/task13_history_profiles_v1`.
Full-history counterpart уже существует в Task14; его будущие metrics неизвестны.

### Исторический comparator Task12

`task12_ranker_backtest_v1`: fresh train-only quantization, border_count=32,
1,030 trees, depth 7, lr 0.08, Logloss, seed 42, 201 features. Selection train
rolling_1+rolling_2, выбор на 16,384 ID-sampled users rolling_3; canonical refit
rolling_1+rolling_2+rolling_3, 46,008,034 training rows и 488,670 positives,
полные inverse sampling weights. Canonical P@20 all/labeled
`0.005007444342299852 / 0.006816357898745886`, 20,045 hits.
Candidate recall/coverage/oracle не изменились относительно Task08 ниже.
Модель: `artifacts/task12_ranker_backtest_v1/model/`, SHA-256
`e9522e6536068a55618fafd862ba02c0c93e8fa795c7b6ea94e255615a0ca83e`.
Команда исходного run:
`./scripts/run_task12_ranker_backtest.sh configs/task12_ranker_backtest_v1.json`.
Проверка:
`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 .venv/bin/python scripts/run_ranker_backtest.py --verify-only artifacts/task12_ranker_backtest_v1`.
Для следующего full-history fit нужен отдельный run с четырьмя training
folds и проверенной quantization policy; текущий Task11 не изменён.

### Исторический comparator Task08

`task08_catboost_pointwise_v1` — предыдущий best run. Это первый fixed
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
- Final-fit aggregate и submission уже созданы в `artifacts/task11_full_fit_v1/`;
  см. `MODEL_EXPLAIN.md` и актуальную сводку аудита выше.
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
- Submission Task11 использует подтверждённую schema `user_id,item_ids` с
  bracketed list. Published metrics содержат 200,152 users, ровно 20 items,
  нулевые seen/unknown/null/duplicate counts и portable_repeat_equal=true.
  SHA-256: `100613900ba1bb2660a5c32e3245790be6ccd9c4aa9dab382fb625c389aeea46`.
- Git доступен. Перед аудитом единственным untracked файлом был пользовательский
  `Makefile`; он не изменён.
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

1. Пользователь запускает `./scripts/run_task15_sasrec_optuna.sh`; после результата
   проверить `best_recipe.json`, `trials.json`, обе P20, recall/ALS overlap,
   положителен ли equal-budget gain и не упирается ли best epoch в budget12.
2. Freeze выбранные параметры/epoch/policy. Selection sample не использовать
   как доказательство улучшения на других днях; сначала проверить перенос.
3. После Optuna freeze recipe/epochs/budget, подготовить отдельный launcher
   обучения на остальных rolling и canonical diagnostic; reuse Task06
   models/candidates без повторного fit. Не выбирать параметры по canonical.
4. Сравнить ALS/SASRec positive overlap и incremental union recall при одинаковом
   budget; для перспективного варианта сделать matched Task13 ranker backtest.
   Разделить отказ от ALS generator и от всех ALS-derived features.
5. По итогам freeze решения подготовить отдельный новый full-history run.
   Task13 остаётся validation baseline, Task14 submission уже существует;
   текущие artifacts не перезаписывать, новые long runs в сессии не запускать
   без точного запроса пользователя.
