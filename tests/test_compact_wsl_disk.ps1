# Windows-only regression tests. Native WSL and DiskPart are mocked; no trimming.
[CmdletBinding()]
param([string]$Distro = 'Ubuntu')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$helper = Join-Path $PSScriptRoot '..\scripts\compact_wsl_disk.ps1'
$parseTokens = $null
$parseErrors = $null
[Management.Automation.Language.Parser]::ParseFile($helper, [ref]$parseTokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }

# Global test state is needed when the helper invokes the mock from its script scope.
$global:compactTestNativeCalls = [Collections.Generic.List[object]]::new()
$global:compactTestExitCode = 0
function wsl.exe {
    $global:compactTestNativeCalls.Add(@($args))
    $global:LASTEXITCODE = $global:compactTestExitCode
}
function diskpart.exe { throw 'A prerequisite check must never invoke DiskPart.' }

try {
    & $helper -Distro $Distro *> $null
    if ($global:compactTestNativeCalls.Count -ne 0) { throw 'Default preview unexpectedly invoked WSL.' }

    & $helper -Distro $Distro -CheckPrerequisites *> $null
    if ($global:compactTestNativeCalls.Count -ne 1) { throw 'Expected exactly one executable probe.' }
    $call = $global:compactTestNativeCalls[0]
    $execIndex = [Array]::IndexOf($call, '--exec')
    if ($execIndex -lt 0 -or -not $call[$execIndex + 1].StartsWith('/') -or $call[-1] -ne '--version') {
        throw 'Probe must use an absolute Linux path and --version, without trimming.'
    }

    $global:compactTestNativeCalls.Clear()
    $global:compactTestExitCode = 127
    $failed = $false
    try { & $helper -Distro $Distro -CheckPrerequisites *> $null }
    catch {
        if ($_.Exception.Message -notmatch 'Cannot execute') { throw }
        $failed = $true
    }
    if (-not $failed -or $global:compactTestNativeCalls.Count -ne 1) {
        throw 'Missing executable must stop prerequisite checking immediately.'
    }
    if ($global:compactTestNativeCalls[0][-1] -ne '--version') { throw 'Failed probe triggered another operation.' }
    Write-Output 'Passed: parser, preview isolation, absolute executable probe, prerequisite failure guard.'
} finally {
    Remove-Variable compactTestNativeCalls, compactTestExitCode -Scope Global
}
