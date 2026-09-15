# Освобождение G: и сжатие WSL

## После SASRec/ALS600 ranker: 2026-09-09

Полный `task15_sasrec_ranker_v1` завершён; опубликованные модели, обе validation
P20 и `submission.csv` проверены повторно. Очистка пользовательских артефактов
и сжатие VHDX в сессии Codex **не выполнялись**; подготовлены следующие команды.

В WSL, из корня репозитория:

```bash
./scripts/cleanup_task15_ranker.sh          # проверка результата и preview
./scripts/cleanup_task15_ranker.sh --apply  # удалить только рабочий кэш этого run
```

Удаляемый каталог: `artifacts/.task15_sasrec_ranker_v1.work`.
Сохраняются `artifacts/task15_sasrec_ranker_v1/` целиком (включая submission,
SASRec/ALS/CatBoost models, history, profiles, lookups, validation metrics),
родительские runs, общие fold artifacts/sequence caches, `data/`, `logs/`
и `experiments/results.csv`. Поэтому команда не ищет «лишнее» по глобальной
маске `.work` и не удаляет нужный `.task05_implicit_als_v1.best-model`.

В preview обнаружено1689 файлов и4.385GiB уникального содержимого рабочего
каталога, однако освобождаемые file blocks —только **0.254GiB**:
4.133GiB имеют hard links за его пределами. В оценке каждая inode учитывается
один раз; размер файла с оставшейся внешней ссылкой не считается освобождаемым.
Большие sampled features, TSV и quantized pools ранкер удалил самостоятельно
после завершения каждого fit. Повторное удаление кэша не вернёт их размер ещё раз.

Cleanup требует завершённый artifact с совпадающим config/checkpoint и успешный
полный verifier. Он берёт те же locks, что и launcher/training runners,
отказывает при активном pipeline, symlink, неизвестном файле в корне рабочего
каталога или несоответствии владельца. Есть progressbar по файлам.
Вначале записывается receipt, затем work переименовывается в
`artifacts/.task15_sasrec_ranker_v1.cleanup-trash` и удаляется.
После прерывания повторите `--apply`: оставшийся trash проверяется по inode
и receipt. Успешное повторное выполнение при уже удалённом work ничего не удаляет.
Отчёт: `artifacts/task15_sasrec_ranker_v1_cleanup_v1/{config,plan,metrics}.json`.
Logs/метрики финального эксперимента не изменяются.

## Возврат места на G: новый Windows helper

На момент проверки Linux показывает около606GiB свободного, G: —около114GiB.
В реестре Windows подтверждено: Ubuntu, WSL2, `G:\WSL\Ubuntu\ext4.vhdx`;
размер VHDX около415.12GiB. Динамический VHD не уменьшается автоматически
после удаления файлов; для возврата места используется `compact vdisk`.
[Документация Microsoft](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk).

Для этого подготовлен `scripts/compact_wsl_disk.ps1`:
он берёт путь именно из регистрации выбранного WSL2 distro, при явном `-VhdPath`
проверяет точное совпадение; без `-Apply` выполняет только read-only preview.
Текущий путь ASCII; helper формирует ASCII DiskPart script и отказывает для
не-ASCII пути вместо молчаливой подмены символов. Модели/файлы Linux при
компактации не удаляются. Точный прирост свободного места заранее неизвестен.

Сначала сохраните работу и закройте Codex, WSL-терминалы и Docker Desktop.
Откройте **Windows PowerShell от имени администратора**. Скопируйте helper
на Windows-диск: UNC-путь внутри WSL перестанет быть доступен после shutdown.

```powershell
Copy-Item '\\wsl.localhost\Ubuntu\home\gzakala\kaggle\scripts\compact_wsl_disk.ps1' "$env:TEMP\compact_wsl_disk.ps1" -Force
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\compact_wsl_disk.ps1" -Distro Ubuntu -CheckPrerequisites
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\compact_wsl_disk.ps1" -Distro Ubuntu -Apply
```

Вторая команда показывает план и проверяет запуск `/usr/sbin/fstrim --version`
через WSL. Она не выполняет trim, shutdown или compact и не требует прав
администратора; если Ubuntu остановлена, проверка запустит её. Третья выполняет:

1. Проверку executable, затем `/usr/sbin/fstrim -v /` от root только в выбранном distro.
2. `wsl --shutdown`, затем ожидание освобождения файла VHDX.
3. DiskPart: `select vdisk file="..."`, `compact vdisk`, `exit` для отключённого VHDX.

`wsl --shutdown` немедленно останавливает **все** WSL-дистрибутивы и их процессы,
включая эту сессию Codex. Это причина отдельного ручного Windows-запуска.
[Документация Microsoft](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#shutdown).
Компактация отключённого VHD поддерживается Microsoft; helper не делает
force-detach, unregister, format или удаления VHDX. Если Docker/WSL вновь занял
файл, скрипт прекращает работу. DiskPart запускается без `noerr` и его exit code
проверяется. [DiskPart scripts](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/diskpart-scripts-and-examples).

Терминал показывает этапы и собственный progress DiskPart. Transcript и точные
DiskPart commands сохраняются в `%TEMP%\wsl_compact_<timestamp>.log` и
`%TEMP%\wsl_compact_<timestamp>.diskpart.txt`; выводится фактическое свободное
место до/после. WSL не перезапускается автоматически. Затем можно выполнить:

```powershell
Get-PSDrive G
wsl -d Ubuntu
```

### Ошибка `execvpe(fstrim) failed: No such file or directory`

При пользовательском запуске 2026-09-09 старый helper остановился до trim,
shutdown и DiskPart: `wsl --exec fstrim` не находил программу по имени.
В Ubuntu она установлена в `/usr/sbin/fstrim`; абсолютный путь работает
при вызове из Windows PowerShell. Helper исправлен: и проверка, и trim
используют этот путь. Устанавливать пакет в текущей Ubuntu не требуется.
Перед повторным запуском обязательно выполните `Copy-Item ... -Force` из блока
выше: исправление репозитория само по себе не обновляет старую копию в `%TEMP%`.

## Проверки новых helper scripts

- `bash -n scripts/cleanup_task15_ranker.sh` и `--help` прошли.
- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_ranker_cleanup.py -v`:11tests passed.
- Ruff passed. Реальный run проверен только в preview; work остался на месте.
- Limited end-to-end smoke: disposable hard-link copy готового CPU smoke artifact,
  полный verifier до/после удаления тестового кэша, повторный no-op; оригинальные
  модели и CSV не изменились. Отчёт: `artifacts/task15_ranker_cleanup_smoke_v1/`.
- Windows PowerShell Parser, реальный preview и отказ при неправильном VHDX path
  прошли. Вызовы fstrim/shutdown/DiskPart с изменением реального диска не выполнялись.
- После исправления реальный `-CheckPrerequisites` из Windows прошёл;
  `/usr/sbin/fstrim --dry-run --verbose /` через Windows → WSL тоже прошёл
  без trim. Это проверка запуска, а не результат освобождения места.
- Windows regression: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  .\tests\test_compact_wsl_disk.ps1 -Distro Ubuntu` из корня репозитория.
  WSL/DiskPart подменены; проверяются preview без вызова WSL, абсолютный путь
  для `--version` и немедленный отказ при недоступности программы.

## Предыдущая очистка Task11: 2026-09-05

2026-09-05 полностью проверен `artifacts/task11_full_fit_v1`: checksums,
переносимая модель, submission round trip и корректность top-20 всех 200,152
пользователей. Затем штатной командой Task11 удалён только его восстановимый
рабочий кэш `artifacts/.task11_full_fit_v1.work`: **70.31 GiB**.

Отчёт: `artifacts/task13_disk_cleanup_20260905_v1/`.
Raw data, Task06/Task07, результаты экспериментов, submission и модели сохранены.
В частности, `.task05_implicit_als_v1.best-model` **нужен**: из него Task06 берёт
ALS rolling_3. По имени скрытой папки нельзя считать её лишним кэшем.

Удаление освободило место в Linux, но G: оставался с 109.01 GiB свободного.
Причина: `G:\WSL\Ubuntu\ext4.vhdx` занимает около 423.5 GiB и автоматически не
уменьшается вместе с Linux-файлами. [Microsoft: compact vdisk](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk).

## Команды в Windows

Сначала дождитесь завершения работы Codex, сохраните открытые файлы и закройте
Docker Desktop и окна/процессы WSL. `wsl --shutdown` завершит текущую сессию Codex;
поэтому этот шаг оставлен для ручного выполнения после получения инструкции.
Имя дистрибутива проверено: `Ubuntu`; второй дистрибутив — `docker-desktop`.

Откройте **PowerShell от имени администратора**:

```powershell
wsl -d Ubuntu -u root --exec /usr/sbin/fstrim -v /
wsl --shutdown
diskpart
```

В открывшемся DiskPart выполните:

```text
select vdisk file="G:\WSL\Ubuntu\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

Команда `compact vdisk` работает с динамическим VHD, который отсоединён либо
подключён только для чтения; здесь используется `readonly`.
[Microsoft: compact vdisk](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk).
Если файл занят, закройте автоматически перезапустивший WSL Docker/терминал,
повторите `wsl --shutdown`, затем команды DiskPart. Если Windows предлагает
инициализировать или форматировать подключённый диск, закройте это предложение.

Проверка в PowerShell после завершения:

```powershell
Get-PSDrive G | Select-Object Name, @{Name='FreeGiB';Expression={[math]::Round($_.Free / 1GB, 2)}}
wsl -d Ubuntu
```

Длительность и размер возврата места зависят от VHDX и SSD; удаление 70 GiB не
обещает точно такой же прирост счётчика G:. Возможно освободится и ранее удалённое
место. Перед Task13 нужно не менее **160 GiB** свободного на G:, иначе launcher
остановится на preflight. Не отключайте этот порог ради запуска при 109 GiB.

Затем в WSL из корня репозитория:

```bash
./scripts/run_task13_history_profiles.sh configs/task13_history_profiles_v1.json
```
