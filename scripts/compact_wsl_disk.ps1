<#
.SYNOPSIS
Preview, or trim and compact, one registered WSL2 distribution's VHDX.
.DESCRIPTION
Run a local Windows copy of this script from elevated PowerShell with -Apply.
Save all work and close WSL terminals, Codex and Docker Desktop first.
The default is a read-only preview. This script never deletes a VHDX.
Use -CheckPrerequisites to test the Linux executable without trimming or shutdown.
#>
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu',
    [string]$VhdPath,
    [switch]$CheckPrerequisites,
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# wsl --exec does not run the interactive shell that normally adds sbin to PATH.
# Use the Ubuntu executable's absolute path for BOTH the probe and actual trim.
$fstrimPath = '/usr/sbin/fstrim'
function Test-FstrimExecutable {
    & wsl.exe --distribution $Distro --user root --exec $fstrimPath --version
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot execute $fstrimPath in '$Distro' (exit $LASTEXITCODE). Check the util-linux installation; compaction cancelled."
    }
}

$registrations = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' |
    ForEach-Object { Get-ItemProperty $_.PSPath } |
    Where-Object { $_.DistributionName -eq $Distro })
if ($registrations.Count -ne 1 -or $registrations[0].Version -ne 2) {
    throw "Expected exactly one registered WSL2 distribution named '$Distro'."
}
$registration = $registrations[0]
$filename = 'ext4.vhdx'
if ($registration.PSObject.Properties['VhdFileName'] -and $registration.VhdFileName) {
    $filename = $registration.VhdFileName
}
$registeredPath = [IO.Path]::GetFullPath((Join-Path $registration.BasePath $filename))
if (-not $VhdPath) { $VhdPath = $registeredPath }
$VhdPath = [IO.Path]::GetFullPath($VhdPath)
if (-not [string]::Equals($VhdPath, $registeredPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Requested VHDX does not match the registered distribution. Refusing to proceed.'
}
if ($VhdPath -match '["\r\n]' -or [IO.Path]::GetExtension($VhdPath) -ne '.vhdx' -or
    -not (Test-Path -LiteralPath $VhdPath -PathType Leaf)) {
    throw 'Expected an existing VHDX file with a valid DiskPart path.'
}
if ($VhdPath -match '[^\x20-\x7E]') {
    throw 'This helper writes an ASCII DiskPart script and requires an ASCII VHDX path.'
}

$beforeLength = (Get-Item -LiteralPath $VhdPath).Length
$driveName = [IO.Path]::GetPathRoot($VhdPath).TrimEnd('\').TrimEnd(':')
$beforeFree = (Get-PSDrive -Name $driveName).Free
Write-Host "Distribution: $Distro"
Write-Host "Registered VHDX: $VhdPath"
Write-Host ('VHDX length: {0:N2} GiB; drive free: {1:N2} GiB' -f ($beforeLength / 1GB), ($beforeFree / 1GB))
Write-Host 'Plan: fstrim / in the selected distribution; wsl --shutdown; compact the detached VHDX.'
Write-Host 'Applying this plan stops ALL WSL distributions, including Codex and Docker WSL workloads.'
if (-not $Apply) {
    if ($CheckPrerequisites) {
        Test-FstrimExecutable
        Write-Host "Prerequisite passed: $fstrimPath can be executed through wsl --exec."
    }
    Write-Host 'PREVIEW ONLY. Nothing was trimmed, stopped or compacted. Add -Apply to execute.'
    return
}

$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open Windows PowerShell as Administrator before using -Apply.'
}
if ($PSScriptRoot.StartsWith('\\')) {
    throw 'Copy this script to a local Windows directory (for example $env:TEMP) before -Apply.'
}
Get-Command wsl.exe, diskpart.exe -ErrorAction Stop | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$logPath = Join-Path $env:TEMP "wsl_compact_$stamp.log"
$commandsPath = Join-Path $env:TEMP "wsl_compact_$stamp.diskpart.txt"
Start-Transcript -Path $logPath -NoClobber | Out-Null
try {
    Test-FstrimExecutable
    Write-Progress -Activity 'WSL disk compaction' -Status '1/3: discard unused Linux blocks' -PercentComplete 5
    & wsl.exe --distribution $Distro --user root --exec $fstrimPath -v /
    if ($LASTEXITCODE -ne 0) { throw "fstrim failed with exit code $LASTEXITCODE; compaction cancelled." }

    Write-Progress -Activity 'WSL disk compaction' -Status '2/3: stop WSL and wait for VHDX release' -PercentComplete 25
    & wsl.exe --shutdown
    if ($LASTEXITCODE -ne 0) { throw "WSL shutdown failed with exit code $LASTEXITCODE." }
    $released = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $handle = [IO.File]::Open($VhdPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
            $handle.Dispose()
            $released = $true
            break
        } catch [IO.IOException] {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $released) { throw 'VHDX is still in use. Close Docker/WSL applications, then retry. No force-detach is attempted.' }

    # Microsoft supports compact vdisk on a detached dynamic VHD(X).
    # Do not attach, format, force-detach or unregister any disk/distribution.
    @("select vdisk file=`"$VhdPath`"", 'compact vdisk', 'exit') |
        Set-Content -LiteralPath $commandsPath -Encoding Ascii
    Write-Progress -Activity 'WSL disk compaction' -Status '3/3: DiskPart compaction (see native progress below)' -PercentComplete 40
    & diskpart.exe /s $commandsPath
    if ($LASTEXITCODE -ne 0) { throw "DiskPart failed with exit code $LASTEXITCODE. See $logPath" }

    $afterLength = (Get-Item -LiteralPath $VhdPath).Length
    $afterFree = (Get-PSDrive -Name $driveName).Free
    Write-Host ('VHDX length: {0:N2} -> {1:N2} GiB' -f ($beforeLength / 1GB), ($afterLength / 1GB))
    Write-Host ('Drive free: {0:N2} -> {1:N2} GiB (change {2:N2} GiB)' -f ($beforeFree / 1GB), ($afterFree / 1GB), (($afterFree - $beforeFree) / 1GB))
    Write-Host 'Compaction completed. WSL remains stopped; restart it when ready.'
} finally {
    Write-Progress -Activity 'WSL disk compaction' -Completed
    Stop-Transcript | Out-Null
    Write-Host "Log: $logPath"
    Write-Host "DiskPart commands: $commandsPath"
}
