# M1-7 sequential training runner. Run as its own process:
#   pwsh -NoProfile -File run_m17_sweep.ps1 [-Strengths 0,0.5,1,2] ...
# It pins ITSELF (affinity + priority); conda/python/DataLoader workers inherit both.
# Stops at the first failure (non-zero exit or incomplete results.csv).
param(
    [double[]]$Strengths = @(0, 0.5, 1, 2),
    [int]$Epochs = 30,
    [int]$Workers = 2,
    [int]$Seed = 0,
    [string]$Prefix = 'm17',
    [string]$Suffix = '',
    [string]$Config = '',
    [int]$Batch = 0,
    [string]$ProjectRoot = 'C:\workspace\gpr_detection',
    [int]$Mask = 0xFF,
    [ValidateSet('Idle','BelowNormal','Normal')][string]$Priority = 'Normal'
)
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$inv = [System.Globalization.CultureInfo]::InvariantCulture

$self = [System.Diagnostics.Process]::GetCurrentProcess()
$self.ProcessorAffinity = [IntPtr]$Mask
$self.PriorityClass = $Priority
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $ProjectRoot
$logDir = Join-Path $ProjectRoot 'runs\logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$conda = 'C:\ProgramData\miniforge3\Scripts\conda.exe'

foreach ($s in $Strengths) {
    $sTxt = $s.ToString('g', $inv)
    $name = "${Prefix}_aug${sTxt}${Suffix}"
    $runDir = Join-Path $ProjectRoot "runs\detect\$name"
    $csv = Join-Path $runDir 'results.csv'
    $log = Join-Path $logDir "$name.log"
    $gpuLog = Join-Path $logDir "$name.gpu.csv"

    if ((Test-Path $csv) -and ((Get-Content $csv).Count -ge ($Epochs + 1))) {
        Write-Output "[skip] $name already complete"
        continue
    }
    if (Test-Path $runDir) {
        Write-Output "[stop] $runDir exists but is incomplete; Ultralytics would create '${name}2'. Move or delete it first."
        exit 2
    }

    $cmdArgs = @('run', '--no-capture-output', '-n', 'gpr-dev', 'python', '-I',
        'scripts/train_detector.py', '--fold', '0', '--epochs', $Epochs,
        '--augment-strength', $sTxt, '--seed', $Seed, '--workers', $Workers,
        '--name', $name)
    if ($Config) { $cmdArgs += @('--config', $Config) }
    if ($Batch -gt 0) { $cmdArgs += @('--batch', $Batch) }

    $t0 = Get-Date
    "[start] $name $($t0.ToString('s')) mask=0x{0:X} workers=$Workers prio=$Priority" -f $Mask | Tee-Object -FilePath $log -Append
    $smi = Start-Process nvidia-smi -ArgumentList '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw', '--format=csv', '-l', '5' `
        -RedirectStandardOutput $gpuLog -NoNewWindow -PassThru
    & $conda @cmdArgs 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    Stop-Process -Id $smi.Id -ErrorAction SilentlyContinue
    $wall = [int]((Get-Date) - $t0).TotalSeconds
    "[end] $name exit=$code wall_s=$wall" | Tee-Object -FilePath $log -Append

    $rows = if (Test-Path $csv) { (Get-Content $csv).Count - 1 } else { 0 }
    if ($code -ne 0 -or $rows -lt $Epochs) {
        Write-Output "[stop] $name failed (exit=$code, epochs_in_results=$rows/$Epochs). Remaining runs NOT started."
        exit 1
    }
}
Write-Output '[done] all runs complete'
exit 0
