param(
    [string]$Run = 'runs/ppo-normal-v1',
    [switch]$Resume,
    [switch]$NoMonitor,
    [ValidateRange(1024,65535)][int]$DashboardPort = 8765
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$pythonPath = (py -3.10 -c 'import sys; print(sys.executable)').Trim()
$bridgeListener = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort 9999 -State Listen -ErrorAction SilentlyContinue
if ($bridgeListener) { throw "Port 9999 is already owned by PID $($bridgeListener.OwningProcess). Leave the running trainer/evaluator in place." }
$runPath = [System.IO.Path]::GetFullPath((Join-Path $workspace $Run))
if (Test-Path -LiteralPath (Join-Path $runPath 'stop.request')) { throw "A stop.request exists in $runPath. Remove it before resuming." }
if ((Test-Path -LiteralPath (Join-Path $runPath 'status.json')) -and -not $Resume) { throw 'Run already exists. Use -Resume or choose a new -Run.' }
$gamePath = Join-Path $workspace 'runtime\game\isaac-ng.exe'
$runningGame = Get-CimInstance Win32_Process -Filter "Name = 'isaac-ng.exe'" |
    Where-Object { $_.ExecutablePath -eq $gamePath }
if (-not $runningGame) {
    & $pythonPath scripts/prepare_runtime.py
    if ($LASTEXITCODE -ne 0) { throw 'Runtime preparation failed' }
    $game = Start-Process -FilePath $gamePath -ArgumentList '--luadebug' -WorkingDirectory (Split-Path -Parent $gamePath) -WindowStyle Hidden -PassThru
}
$trainArgs = @('-u','-m','isaac_rl.train','--run',('"' + $runPath + '"'))
if ($Resume) { $trainArgs += @('--resume',('"' + (Join-Path $runPath 'latest.pt') + '"')) }
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$trainer = Start-Process -FilePath $pythonPath -ArgumentList $trainArgs -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath 'stdout.log') -RedirectStandardError (Join-Path $runPath 'stderr.log') -PassThru
Write-Output "Training PID $($trainer.Id); run $runPath"
if (-not $NoMonitor) {
    try { & (Join-Path $PSScriptRoot 'launch_dashboard.ps1') -Run $Run -Port $DashboardPort }
    catch { Write-Warning "Training remains running; dashboard could not start: $_" }
}
if (-not $runningGame) {
    # Normal menus are essential. --set-stage enables a debug run with unlocked
    # items and makes the game ignore requested seed resets.
    & $pythonPath scripts/game_window.py --start
}
