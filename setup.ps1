$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = $null
foreach ($taskVersion in @('Python314','Python313','Python312','Python311')) {
    $taskCandidate = Join-Path $env:LOCALAPPDATA "Programs\Python\$taskVersion\python.exe"
    if (Test-Path -LiteralPath $taskCandidate) { $taskPython = $taskCandidate; break }
}
if (-not $taskPython) { throw 'Install Python 3.11 or newer from python.org, then run this file again.' }
& $taskPython -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create a Python environment.' }
$taskVenv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $taskVenv -m pip install --upgrade pip
& $taskVenv -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check internet connectivity and pip output.' }
New-Item -ItemType Directory -Path models -Force | Out-Null
if (-not (Test-Path -LiteralPath 'models\yolo11n.pt')) {
    Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt' -OutFile 'models\yolo11n.pt'
}
Write-Output 'Ready. Double-click start.cmd.'
