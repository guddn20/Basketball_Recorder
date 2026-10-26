$ErrorActionPreference = 'Stop'
$taskPidPath = Join-Path $PSScriptRoot 'data\server.pid'
$taskPortPath = Join-Path $PSScriptRoot 'data\server.port'
if ((Test-Path -LiteralPath $taskPidPath) -and (Test-Path -LiteralPath $taskPortPath)) {
    $taskServerPid = [int](Get-Content -LiteralPath $taskPidPath)
    $taskServerPort = [int](Get-Content -LiteralPath $taskPortPath)
    try {
        $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$taskServerPort/api/health" -TimeoutSec 2
        $taskProcess = Get-Process -Id $taskServerPid -ErrorAction SilentlyContinue
        if ($taskHealth.app -eq 'courtside-local' -and $taskHealth.root -eq $PSScriptRoot -and $taskHealth.pid -eq $taskServerPid -and $taskProcess -and $taskProcess.ProcessName -like 'python*') {
            Stop-Process -Id $taskServerPid
            Write-Output 'COURTSIDE stopped. Saved records remain in data.'
        }
    } catch { Write-Output 'COURTSIDE is already stopped.' }
}
