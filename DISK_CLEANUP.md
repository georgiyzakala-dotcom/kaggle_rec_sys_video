# Освобождение G: и сжатие WSL

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
wsl -d Ubuntu -u root -- fstrim -v /
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
