param(
    [string]$Run = 'runs/ppo-damage-v4',
    [ValidateRange(1024,65535)][int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$runPath = (Resolve-Path -LiteralPath (Join-Path $workspace $Run)).Path
$url = "http://127.0.0.1:$Port/"
$listener = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    try { $existing = Invoke-RestMethod -Uri ($url + 'api/history') -TimeoutSec 5 }
    catch { throw "Port $Port is occupied by another service. Choose a different dashboard port." }
    if ($existing.run_path -ne $runPath) { throw "Port $Port serves another run. Choose a different dashboard port." }
    Write-Output "Dashboard already running: $url (PID $($existing.server.pid))"
    return
}
$pythonPath = (py -3.10 -c 'import sys; print(sys.executable)').Trim()
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$display = Start-Process -FilePath $pythonPath -ArgumentList '-u','-m','isaac_rl.dashboard',('"' + $runPath + '"'),'--port',([string]$Port) -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath "dashboard-$stamp.log") -RedirectStandardError (Join-Path $runPath "dashboard-$stamp-error.log") -PassThru
Write-Output "Dashboard starting: $url (PID $($display.Id)); read-only, remains available after learner exit."
