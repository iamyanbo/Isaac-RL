param(
    [string]$Run = 'runs/ppo-damage-v4',
    [int]$Games = 3,
    [int[]]$Instances = @(),
    [ValidateRange(0,2147483647)][int]$Steps = 0,
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [ValidateRange(0,1)][double]$EntropyCoef = 0.02,
    [ValidateRange(1,30)][int]$Frames = 8,
    [ValidateSet('legacy_v1','physical_v1')][string]$TimingProfile = 'legacy_v1',
    [switch]$Resume,
    [string]$Initialize = '',
    [ValidateSet('', 'legacy_v1', 'balanced_v2', 'confirmed_v3')][string]$RewardProfile = '',
    [ValidateSet('', 'legacy_v1', 'terrain_v2')][string]$ObservationProfile = '',
    [switch]$NoMonitor,
    [ValidateRange(1024,65535)][int]$DashboardPort = 8765
)
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('Games') -and $PSBoundParameters.ContainsKey('Instances')) { throw 'Choose -Games or -Instances, not both' }
if ($Games -lt 1 -or $Games -gt 8) { throw 'Use between 1 and 8 game instances' }
if ($Resume -and $Initialize) { throw '-Resume and -Initialize are mutually exclusive' }
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$pythonPath = (py -3.10 -c 'import sys; print(sys.executable)').Trim()
$runPath = [System.IO.Path]::GetFullPath((Join-Path $workspace $Run))
if (Test-Path -LiteralPath (Join-Path $runPath 'stop.request')) { throw 'Remove the stop.request before resuming' }
if ((Test-Path -LiteralPath (Join-Path $runPath 'status.json')) -and -not $Resume) { throw 'Run exists; resume it or choose another run' }
if (-not $PSBoundParameters.ContainsKey('Instances')) {
    if ($Resume -and -not $PSBoundParameters.ContainsKey('Games') -and (Test-Path -LiteralPath (Join-Path $runPath 'config.json'))) {
        $previousConfig = Get-Content -LiteralPath (Join-Path $runPath 'config.json') -Raw | ConvertFrom-Json
        $Instances = @($previousConfig.ports | ForEach-Object { [int]$_ - 9999 })
    } else { $Instances = @(0..($Games - 1)) }
}
if ($Instances.Count -lt 1 -or $Instances.Count -gt 8 -or @($Instances | Sort-Object -Unique).Count -ne $Instances.Count -or @($Instances | Where-Object { $_ -lt 0 -or $_ -gt 16 }).Count -gt 0) {
    throw 'Provide 1 to 8 unique instance IDs between 0 and 16'
}
$ports = @($Instances | ForEach-Object { 9999 + $_ })
foreach ($port in $ports) {
    $listener = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($listener) { throw "Port $port is in use by PID $($listener.OwningProcess). Preserve the live trainer/evaluator." }
}
foreach ($instance in $Instances) {
    & $pythonPath scripts/launch_worker.py $instance
    if ($LASTEXITCODE -ne 0) { throw "Game instance $instance failed to start" }
}
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$trainArgs = @('-u','-m','isaac_rl.train_vector','--run',('"' + $runPath + '"'),'--ports') + @($ports | ForEach-Object { [string]$_ })
$trainArgs += @('--steps',[string]$Steps)
$trainArgs += @('--device',$Device)
if ($PSBoundParameters.ContainsKey('Frames')) { $trainArgs += @('--frames',[string]$Frames) }
if ($PSBoundParameters.ContainsKey('TimingProfile')) { $trainArgs += @('--timing-profile',$TimingProfile) }
if ($PSBoundParameters.ContainsKey('EntropyCoef')) {
    $trainArgs += @('--entropy-coef',$EntropyCoef.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
if ($Resume) { $trainArgs += @('--resume',('"' + (Join-Path $runPath 'latest.pt') + '"')) }
if ($RewardProfile) { $trainArgs += @('--reward-profile',$RewardProfile) }
if ($ObservationProfile) { $trainArgs += @('--observation-profile',$ObservationProfile) }
if ($Initialize) {
    $initialPath = (Resolve-Path -LiteralPath $Initialize).Path
    $parentPath = Join-Path $runPath 'parent.pt'
    if (Test-Path -LiteralPath $parentPath) { throw 'Parent snapshot already exists' }
    Copy-Item -LiteralPath $initialPath -Destination $parentPath
    $trainArgs += @('--resume',('"' + $parentPath + '"'))
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$trainer = Start-Process -FilePath $pythonPath -ArgumentList $trainArgs -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath "stdout-$stamp.log") -RedirectStandardError (Join-Path $runPath "stderr-$stamp.log") -PassThru
Write-Output "Parallel training PID $($trainer.Id); run $runPath; ports $($ports -join ', ')"
if (-not $NoMonitor) {
    try { & (Join-Path $PSScriptRoot 'launch_dashboard.ps1') -Run $Run -Port $DashboardPort }
    catch { Write-Warning "Training remains running; dashboard could not start: $_" }
}
