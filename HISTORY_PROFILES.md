# Task13: ALS-профили истории для CatBoost

Полный эксперимент `task13_history_profiles_v1` завершён и проверен.
Новые модели кандидатов не обучаются: используются неизменные Task06/Task07 и
сохранённые ALS-модели каждого fold. Новый validation baseline — Task13 D_all:
`precision_at_20_all_targets=0.005077640992845437`,
`precision_at_20_labeled_users=0.006911912728855518` (20,326 hits).
Это +281 hits/+1.4018% к Task12 и +543/+2.7448% к matched A_base30m.
Фактически полный запуск занял74.41min, peak RSS31.62GiB.

## Исправление публикации после полного запуска

На `publish_and_verify` обнаружилось расхождение labeled score на1ULP:
`0.006911912728855518` против `0.0069119127288555186` — всего `8.67e-19`.
Причина — точное сравнение Float64 после параллельного усреднения.
Verifier теперь проверяет integer hits и оба знаменателя точно, а метрики —
относительно `hits/(20*users)` с допуском32ULP. Неверные hits, знаменатели,
сдвиг метрики на одно попадание и NaN/Inf отклоняются тестами.

Добавлена отдельная команда для **полностью подготовленного** staging:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 \
  .venv/bin/python scripts/run_history_profiles.py \
  --config configs/task13_history_profiles_v1.json --publish-only
```

Эта команда уже выполнена; повторять её для завершённого run не нужно.
Она не вызывает training, quantization или shard inference, не заменяет
training config/hashes новыми. Проверяются staging checksums, исходные
checkpoints, frozen winner, portable inference и top20 semantics, затем artifact
публикуется атомарно. Данные восстановления сохранены в
`artifacts/task13_history_profiles_v1/publication_recovery.json`, CSV дополнен
одной строкой. Неполный staging эта команда восстанавливать не пытается.

Проверены158tests и все три сохранённые smoke artifacts с новым verifier.
Review: `artifacts/task13_publication_review_20260905_v1/metrics.json`.
Rolling_3 hits A/B/C/D:28035/29077/29069/29095; D лучше B всего на18hits.
Отдельный вклад event profiles и перенос улучшения на другие дни пока не доказаны.

## Гипотеза и признаки

Кандидат должен соответствовать истории позитивных событий пользователя.
Для каждого временного fold берутся только его `history_daily.parquet` и ALS,
обученный на этой истории. Новое чтение raw и изменение daily contract не нужны.
Сначала история сворачивается до уникальных `(user_id,item_id)`, чтобы повторения
в разные дни не давали итему случайный дополнительный вес.

Строятся четыре профиля: `positive`, `like`, `favorite`, `long_watch`.
Последний использует строго `watch_time > 60`; событие может входить в несколько
профилей. В этой первой ablation время и сила события дополнительно не взвешиваются.

```text
item_direction = als_item_vector / norm(als_item_vector)
centroid = mean(item_direction for unique valid items in the event profile)
concentration = norm(centroid)
cosine = dot(candidate_direction, centroid) / concentration
```

Нормализация каждого итема перед усреднением — заранее фиксированное решение:
его ALS-норма не задаёт вес в профиле. Оно ещё не сравнено с усреднением исходных
ненормированных факторов. Существующие ALS dot/cosine/norm features остаются.

Для каждого профиля передаются пять скаляров:

- `cosine`: близость кандидата к профилю;
- `concentration`: согласованность направлений истории, от 0 до 1;
- `log_count`: `log(1 + number_of_unique_profile_items)`;
- `embedding_coverage`: доля итемов истории с ненулевым доступным ALS-вектором;
- `available`: доступны и кандидат, и ненулевое направление профиля.

При отсутствии близости её значение равно 0, а `available=0` отличает этот случай
от настоящей нулевой близости. Координаты векторов в CatBoost не передаются.
Ортогональный поворот факторов не меняет эти признаки; это проверяется тестом.
Всего 201 старый и 20 новых признаков, без внешних metadata.

## Сравнение

| Вариант | Признаки сверх старых 201 |
|---|---|
| A_base | Нет |
| B_positive | Общий профиль позитивов |
| C_events | Лайки, избранное, длинные просмотры |
| D_all | Все четыре профиля |

Selection: обучение на `rolling_1+rolling_2`, оценка на всех 200,152 target users
`rolling_3` с полными списками кандидатов. Число деревьев фиксировано: 1,030.
Победитель — максимальный macro P@20 all targets; при равенстве предпочтение
более раннему варианту в таблице. Деревья и winner не выбираются по canonical.

Все четыре модели используют один набор строк, одинаковые IPW и один
квантованный Pool со всеми 221 признаками. Ненужные группы исключаются через
CatBoost `ignored_features`. Borders обучаются только на training rows,
`border_count=32`. Квантование выполняется два раза за весь эксперимент:
selection и canonical refit.

После фиксации winner выполняется refit на `rolling_1+rolling_2+rolling_3`,
затем canonical evaluation. На том же canonical Pool заранее предусмотрен
отдельный A_base control. Он позволяет отделить эффект профилей от изменения
обучающей выборки; canonical результаты не меняют сохранённого rolling winner.
Итого пять fits, если победил A_base, иначе шесть.

Бюджет уменьшен с примерно 46 до 30 млн строк: Task12 достигал 45.97 GiB RSS,
а текущий WSL видит около 49 GiB RAM. Сохраняются все positives, применяется
детерминированный дополнительный sampling negatives и обратные полные вероятности
весов. 30 млн — ожидаемое, а не точное число строк. Сравнение с Task12 поэтому
включает изменение sampling; эффект самих профилей показывают matched controls.

## Подготовка диска и ресурсы

Перед full run выполните [DISK_CLEANUP.md](DISK_CLEANUP.md). Старый временный
Task11 уже удалён после полной проверки его результата: освобождено 70.31 GiB
внутри Linux. Windows G: до сжатия VHDX всё ещё показывал 109.01 GiB свободного.

Новых численных лимитов RAM/GPU в переданных файлах не найдено. Пока приняты
следующие консервативные настройки в `configs/task13_history_profiles_v1.json`:

| Ресурс | Перед стартом | Остановка работающего процесса |
|---|---|---|
| Доступная RAM | >= 38 GiB | < 4 GiB |
| RSS вычислительного процесса | — | > 40 GiB |
| Свободная VRAM GPU 0 | >= 13,000 MiB | < 2,048 MiB |
| Свободное место Linux | >= 110 GiB | < 15 GiB |
| Свободное место Windows G: | >= 160 GiB | < 50 GiB |

CPU: 8 threads, BLAS: 1 thread. CatBoost GPU: `gpu_ram_part=0.70` на RTX 5070 Ti
с 16,303 MiB VRAM. Это параметр использования памяти CatBoost, а не ограничение
загрузки GPU в процентах. Для одновременной игры или другой GPU-модели этот запуск
не рассчитан. Объём временного TSV ограничен 60 GiB. До старта нужно именно
160 GiB **свободного места на G:**, а не большой виртуальный размер ext4.

Предварительная оценка RSS квантования была30–36GiB; измеренный peak31.62GiB.
Внешний процесс опрашивает ресурсы каждые 2 секунды, включая время нативных
CatBoost-вызовов. При превышении порога посылает TERM, затем KILL через 10 секунд,
если процесс не завершился. Это наблюдение с интервалом, не строгий kernel memory
limit: короткий пик между опросами возможен. Дополнительно проверяется измеренный
`ru_maxrss` между операциями. Swap не учитывается как доступная RAM.

## Запуск, наблюдение и остановка

Из корня репозитория после сжатия VHDX:

```bash
./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json
```

```bash
# Во втором терминале:
tail -f logs/task13_history_profiles_v1.log
tail -f logs/task13_history_profiles_v1.resources.log
cat logs/task13_history_profiles_v1.resources.json
nvidia-smi

# Остановка, либо Ctrl-C в терминале запуска:
kill -TERM "$(cat logs/task13_history_profiles_v1.pid)"
```

Повторение исходной команды продолжает checkpoint с тем же config и исходниками.
Готовые операции проверяются по SHA-256 и пропускаются. Незавершённый sampling
fold/feature shard/Pool перестраивается; CatBoost использует отдельный snapshot
каждого fit. После изменения code/config нужен новый `run_id`; старые checkpoints
с другой конфигурацией не принимаются.

Один lock защищает Task13 от одновременных запусков; launcher также защищает run ID.
Другие старые runners собственных задач используют свои locks — одновременно с
Task13 запускать их не следует. Опциональная остановка после selection:

```bash
./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json --stop-after-selection
# Продолжить canonical этап: та же команда без --stop-after-selection.
```

Есть вложенные progress bars: run, этап сравнения вариантов, fold/shard/batch и
деревья. Измеримые циклы показывают elapsed/ETA. У непрозрачного нативного
квантования нет достоверного внутреннего процента: runner показывает операцию,
а resource log — heartbeat каждые 30 секунд. Основной plain-text log ограничен
4 файлами по 5 MB; resource log — 3 файлами по 1 MB. В логах нет ANSI-кодов.

## Результаты и проверка

- Итог: `artifacts/task13_history_profiles_v1/` — config, metrics, модель,
  рекомендации, matched-control recommendations, feature importance,
  quantization metadata/borders, profile coverage и переносимый probe.
- Resume/cache: `artifacts/.task13_history_profiles_v1.work/`.
- Текущий лучший вариант: `artifacts/.task13_history_profiles_v1.work/best/best_model.json`.
  Он указывает на атомарно сохранённую portable model внутри work.
- Атомарно зафиксированный выбор: `artifacts/.task13_history_profiles_v1.work/frozen_selection/`.
- `experiments/results.csv` пополняется только после полного результата.

```bash
.venv/bin/python scripts/run_history_profiles.py --verify-only artifacts/task13_history_profiles_v1

# После анализа результатов: удалить восстановимый рабочий кэш этого run.
.venv/bin/python scripts/run_history_profiles.py --verify-only artifacts/task13_history_profiles_v1 --cleanup-recoverable
```

Проверяются SHA-256, два одинаковых portable predicts, точные target users,
20 уникальных известных unseen items, отсутствие null и обе метрики.
Готовый artifact никогда не перезаписывается. Если процесс прервался между
публикацией artifact и записью CSV, `--verify-only` восстановит недостающую строку.
Обучение на полном train и Kaggle submission в Task13 не входят.

## Что уже посчитано и проверено

На всей canonical target аудитории по сохранённым history-only user lookups:

| История | Пользователи | Доля |
|---|---:|---:|
| Любые позитивы | 200,014 | 99.93% |
| Просмотр > 60 s | 199,119 | 99.48% |
| Лайки | 31,501 | 15.74% |
| Избранное | 18,858 | 9.42% |

Медиана числа уникальных позитивных итемов — 41, p90 — 78.
Отдельные лайки/избранное полезно проверять по соответствующим сегментам после
full run: у большинства пользователей этих событий нет. Это покрытие событиями,
а не гарантия ненулевого ALS-профиля. Последнее считает сам полный runner.

Candidate recall остаётся 10.70%, oracle P@20 all/labeled
`0.03429693432990927 / 0.04668652574879622`. Кандидаты ещё далеки от покрытия всех
позитивов; текущий рангер реализует 14.60% доступных oracle hits. Первая ablation
сохраняет candidates для чистого сравнения, затем нужен отдельный recall/depth audit.

Проверки: 152 unit tests; CPU smoke с pause/resume; независимый CPU repeat
совпал точно по всем четырём rolling top-20 и canonical top-20; GPU smoke
на 64 users, 12 trees, depth 7. Все три portable artifacts прошли verify.
CPU smoke P@20 all/labeled `0.0023437500000000003 / 0.003488372093023256`,
GPU smoke `0.0015625 / 0.002325581395348837`. Это проверки исполнения на 64 users,
не результаты сравнения качества и не строки full experiment log.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
ruff check history_profiles.py scripts/run_history_profiles.py scripts/task13_resources.py tests/test_history_profiles.py
bash -n scripts/run_task13_history_profiles.sh
./scripts/run_task13_history_profiles.sh --help
```

Диагностика/benchmark/verification:
`artifacts/task13_preflight_review_20260905_v1/`.
Smoke artifacts: `task13_history_profiles_smoke_v1`,
`task13_history_profiles_cpu_repeat_v1`, `task13_history_profiles_gpu_smoke_v1`.
Их восстановимые work caches после проверок удалены (~9.3 GiB); опубликованные
модели, метрики и verification report сохранены. Это также проверило штатный
путь `--cleanup-recoverable`.

## Оценка времени

Фактически full run занял **74.41 минуты**. Ниже сохранена предварительная
оценка **2–3 часа**, с резервом до **4 часов**. Низкая загрузка GPU в
preparation/quantization нормальна: эти этапы CPU/I/O.

| Работа | Оценка |
|---|---:|
| Проверка входов, профили и новые scalar shards четырёх folds | 35–65 min |
| Два sampled набора и два квантования | 30–45 min |
| Пять-шесть GPU fits | 12–25 min |
| Полные оценки вариантов, проверки и публикация | 20–40 min |

Основание: Task12 выполнялся 82.84 min, из них три квантования — около 55 min.
Грубая линейная оценка новой квантованной матрицы:
`55 min * (2 / 3) * (30 / 46) * (221 / 201) = 26.3 min`, плюс sampling и I/O.
Ограниченный замер новых признаков на 200,000 реальных кандидатах дал
около 257,650 rows/s; для примерно 0.5 млрд строк четырёх folds это около
32 min только на расчёт скаляров. Замер одного префикса shard с тёплым файловым
кэшем не гарантирует скорость полного прохода. Полный runtime и peak RSS/VRAM
сохранит runner. Сжатие VHDX в эти 2–3 часа не включено.
