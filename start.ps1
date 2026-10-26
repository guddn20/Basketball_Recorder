param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPort = 8765
$taskPython = $null
$taskCandidates = @(
    (Join-Path $taskRoot '.venv\Scripts\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
)
foreach ($taskCandidate in $taskCandidates) {
    if (Test-Path -LiteralPath $taskCandidate) {
        & $taskCandidate -c "import importlib.util,sys; from fastapi import FastAPI; import uvicorn,multipart,cv2,numpy,scipy; FastAPI(); sys.exit(0 if importlib.util.find_spec('ultralytics') else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $taskPython = $taskCandidate; break }
    }
}
if (-not $taskPython) { throw 'Python dependencies are missing. Run setup.ps1 first.' }
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'data') -Force | Out-Null
$taskLog = Join-Path $taskRoot 'data\server.log'
$taskOut = Join-Path $taskRoot 'data\server-output.log'
$taskAlreadyRunning = $false
foreach ($taskTryPort in 8765..8775) {
    try {
        $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$taskTryPort/api/health" -TimeoutSec 1
        if ($taskHealth.app -eq 'courtside-local' -and $taskHealth.root -eq $taskRoot) { $taskPort = $taskTryPort; $taskAlreadyRunning = $true; break }
    } catch {
        $taskListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$taskTryPort)
        try { $taskListener.Start(); $taskPort = $taskTryPort; break } catch { continue } finally { $taskListener.Stop() }
    }
}
if (-not $taskAlreadyRunning) {
    $env:COURTSIDE_PORT = [string]$taskPort
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('server.py') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardError $taskLog -RedirectStandardOutput $taskOut -PassThru
    $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRoot 'data\server.pid')
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 60; $taskAttempt++) {
        try {
            $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$taskPort/api/health" -TimeoutSec 1
            if ($taskHealth.app -eq 'courtside-local') { $taskReady = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
        if ($taskProcess.HasExited) { break }
    }
    if (-not $taskReady) { throw "Server could not start. See $taskLog" }
}
$taskHealth.pid | Set-Content -LiteralPath (Join-Path $taskRoot 'data\server.pid')
$taskPort | Set-Content -LiteralPath (Join-Path $taskRoot 'data\server.port')
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$taskPort" }
