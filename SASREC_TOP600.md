# Task15: рекомендации ALS@600 и SASRec@600

Обновление 2026-09-08: **полный пользовательский запуск завершён и проверен**.
Runtime7822,41s (около2ч10мин), model_fits=0; peak RSS4,55GiB/VRAM4294MiB.
CLI verifier проверил SHA256 и агрегацию всех фолдов. Recall ниже также
независимо пересчитан из целых per-user counts с проверкой полного target universe.

Global popularity, recency popularity и item2item везде по200; micro recall в процентах:

| Fold | ALS200 + SASRec200 | ALS300 + SASRec300 | ALS600 + SASRec600 | Hits @600 | Eligible positives |
|---|---:|---:|---:|---:|---:|
| rolling_1 | 9.9115 | 11.6766 | 15.9119 | 292668 | 1839297 |
| rolling_2 | 11.4667 | 13.5556 | 18.5140 | 315415 | 1703659 |
| rolling_3 | 13.6827 | 16.0698 | 21.5761 | 346416 | 1605551 |
| canonical | 13.2489 | 15.7282 | 21.4281 | 275169 | 1284148 |

Coverage=100%, по200152target users в каждом фолде. Canonical union в среднем
содержит1442,35unique candidates при сумме caps1800. Относительно обоих200:
+105034positive hits, +8,1793п.п. recall (+61,74% относительно прежнего значения).
Это увеличение бюджета1000 ->1800. При одинаковом бюджете1200 на canonical
ALS600-only даёт16,6682%, оба300 —15,7282%; ALS600-only сильнее во всех4folds.
Здесь и выше «only» относится только к ALS/SASRec: остальные3x200 остаются.

Canonical oracle P20 all/labeled: `0.06829884287941165 / 0.09297144916891102`.
Обе actual P20 нового ранкера неизвестны: ранкер на этом union не обучался.
Пересечение SASRec600/ALS600: mean доля SASRec items в ALS22,5637%, micro
Jaccard12,7116%; shared positive hits73961, SASRec-only70566, ALS-only101989.

Повторный расчёт и таблица без округления:
`artifacts/task15_top600_recall_review_v1/{review.py,config.json,metrics.json,recall.csv,report.md}`.
Четыре full-run строки уже находятся в `experiments/results.csv`.
Инструкции ниже описывают воспроизведение; готовый artifact повторно не перезаписывается.

Текущий расчёт top300 остановлен по запросу пользователя 2026-09-08.
В `artifacts/.task15_sasrec_top300_v1.work/` сохранены 97 завершённых блоков
rolling_1; полного top300 результата нет. Новый запуск независим от этих блоков.

## Запуск

```bash
./scripts/run_task15_sasrec_top600.sh
```

Явная конфигурация:

```bash
./scripts/run_task15_sasrec_top600.sh configs/task15_sasrec_top600_v1.json
```

Скрипт использует уже обученные fold models из
`artifacts/task15_sasrec_folds_v1/` и прежние ALS artifacts. Обучения и Optuna
нет. SASRec выдаёт кандидатов на RTX 5070 Ti, ALS использует тот же CPU backend,
что и исходные предикты, с лимитом 8 потоков. У каждой модели сохраняются до
600 уникальных непросмотренных items на пользователя. При отсутствии допустимой
SASRec history native список пуст; такие пользователи сохраняются в знаменателе
метрик, а остальные источники обеспечивают fallback.

Фолды: rolling_1, rolling_2, rolling_3 и canonical; все 200152 target users
каждого фолда. Значения cutoffs и universe берутся из исходных manifests.
Никакие raw/prepared snapshots и веса существующих моделей не меняются.

Ожидаемое время на текущем компьютере: **2,5–3 часа**. Реальный GPU smoke
на одном полном блоке2048users каждого фолда дал21,12 / 22,45 / 23,54 /
24,48s на блок соответственно. Экстраполяция на98блоков каждого фолда —
2,49часа, плюс проверка входов и финальная публикация. На занятом компьютере
возможен больший срок. SASRec занимал4,0–4,4s/блок, ALS с loader10,4–13,8s;
остальное — чтение, метрики и сохранение. Новое обучение в эту оценку не входит.

Измеренный smoke peak: RSS3,88GiB, VRAM4262MiB. Новые candidates и per-user
counts займут ориентировочно5–6GiB; разумно оставить запас10GiB. Финальная
публикация использует hard links внутри того же artifacts filesystem.

Прогресс виден на трёх уровнях: весь запуск по фолдам, блоки пользователей
текущего фолда, batches SASRec/ALS. Для каждого уровня показываются elapsed/ETA.
Блок = до 2048 пользователей. В начале каждого фолда проверяются SHA256
исходной истории, моделей и source artifacts.

## Метрики и сравнения

Из top600 автоматически получаются prefix-списки @200, @300 и @400, поэтому
отдельно повторять остановленный top300 запуск не требуется.

- Native recall обоих источников @50/100/150/200/300/400/600, coverage,
  среднее число кандидатов и quantiles, macro recall по labeled users.
- Дедуплицированный union recall: старый baseline с ALS200; оба по200;
  ALS300/SASRec200, ALS200/SASRec300, оба по300; оба по600 и другие caps.
- Контроль при одинаковой сумме caps1200: ALS600, SASRec600, ALS400/SASRec200,
  ALS200/SASRec400 и ALS300/SASRec300. Global popularity, recency popularity
  и item2item везде остаются по200.
- Пересечение SASRec и ALS @200/300/400/600: raw Jaccard, доля общих items,
  общие и уникальные positive hits.
- Обе oracle Precision@20 и исходные standalone P20 с global fallback.
  P20 нового обученного ранкера не вычисляется: такого ранкера пока нет.

`recall = unique_union_positive_hits / eligible_positive_pairs` — основной
micro recall. Каждый positive pair учитывается один раз после фильтрации
seen/cold. Macro recall и обе P20 имеют отдельные поля и знаменатели.
Сумма caps1800 для двух моделей по600 больше бюджета1200 у двух моделей по300;
это расширение бюджета, а не доказательство преимущества при равных затратах.

## Выходные файлы

`artifacts/task15_sasrec_top600_v1/` публикуется атомарно после всех фолдов:

- `config.json`, `metrics.json`, `manifest.json`;
- `comparison.csv` — union policies, recall, hits, фактические unique counts,
  обе oracle P20;
- `source_depth_metrics.csv`, `overlap_metrics.csv`;
- `folds/<fold>/parts/part-00000/` и последующие блоки:
  `sasrec600.parquet`, `als600.parquet`, `per_user.parquet`, metrics/manifest;
- `folds/<fold>/per_user.parquet` и `metrics.json` — точные fold aggregates.

Схема candidates: `user_id UInt64`, `item_id Int32`, `score Float64`,
`rank UInt32`, `source String`. Для получения @300: фильтр `rank <= 300`.
Объединения не сохраняются как большие дополнительные таблицы: для них
сохраняются точные per-user counts и итоговые метрики.
Четыре full-run строки добавятся в `experiments/results.csv` после публикации.
Smokes и незавершённые фолды не записываются как результаты полного эксперимента.

## Мониторинг, остановка и продолжение

```bash
tail -F logs/task15_sasrec_top600_v1.log
cat logs/task15_sasrec_top600_v1.resources.json
```

Остановка — Ctrl-C либо:

```bash
kill -TERM "$(cat logs/task15_sasrec_top600_v1.pid)"
```

Повторить исходную команду для resume. Завершённые блоки и фолды не повторяются;
незавершённый блок пересчитывается целиком. Параметры/run ID/код между остановкой
и resume менять нельзя. Готовые результаты launcher не перезаписывает.
Один общий Task15 lock запрещает одновременные запуски.

Проверка опубликованного результата:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=8 \
  .venv/bin/python scripts/run_sasrec_top600.py \
  --verify-only artifacts/task15_sasrec_top600_v1
```

## Ограничения ресурсов и проверка кода

CPU8, BLAS1, RSS максимум40GiB, CUDA allocator75%; остановка при available
RAM<4GiB или free VRAM<2048MiB. Собственные файлы запуска <=30GiB. Свободное
место G: не менее80GiB на старте и50GiB во время работы; Linux40/10GiB.
Raw и derived data не копируются, модели не дублируются в выходном artifact.

Первые200 рекомендаций SASRec и первые400 ALS должны точно совпадать с ранее
сохранёнными item IDs/ranks. Старые per-user counts вычисляются заново и
сравниваются точно. Новые рекомендации проверяются на seen/unknown/null/duplicate
пары. Синтетические тесты проверяют resume, repeat и запрещают вызовы fit обеих
моделей. Progress hooks ALS не меняют его scoring path.

```bash
./scripts/run_task15_sasrec_top600.sh --help
bash -n scripts/run_task15_sasrec_top600.sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_sasrec_top600.py' -v
```

GPU preflight имеет отдельный ID `task15_sasrec_top600_gpu_smoke_v1` и ограничен
одним блоком2048users каждого фолда (8192 user-fold evaluations). Его метрики
проверяют код и не являются полной оценкой качества.

Проверки:28tests (top300/top600, metrics, daily data contract, ALS), Ruff,
shell syntax, launcher --help, полный ограниченный GPU smoke120,40s и
отдельный CLI verifier. Повторный исходный GPU `predict(k=600)` на64users
каждого фолда точно совпал с оптимизированным по IDs/ranks/scores.
Артефакт проверки и оценки времени:
`artifacts/task15_top600_preflight_v1/{review.py,config.json,metrics.json}`.
Полный top600 run впоследствии выполнен пользователем; результаты приведены выше.

Новые рабочие файлы: `sasrec_top600.py`, `scripts/run_sasrec_top600.py`,
`scripts/run_task15_sasrec_top600.sh`, два `configs/task15_sasrec_top600_*.json`,
`tests/test_sasrec_top600.py`. Общая оптимизированная GPU выдача и метрики @300
переиспользуются из `sasrec_top300.py`; текущие SASRec/ALS model implementations
не менялись. Handoff обновлён в `PROJECT_STATE.md`, `ROADMAP.md`, `SASREC_FOLDS.md`.
