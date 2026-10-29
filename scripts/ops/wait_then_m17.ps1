# Wait until CPU/GPU resources are free, then start the M1-7 augmentation strength sweep.
# Free = logical CPUs 0,2,4,6 (the physical cores of the training mask 0xFF, used by other-session Chrono workers) average < $PMax %, GPU utilization < 10 %,
#        free RAM > 8 GB, for $Need consecutive checks spaced $IntervalSec apart.
param(
    [int]$PMax = 20,
    [int]$Need = 5,
    [int]$IntervalSec = 60,
    [int]$MaxWaitHours = 12
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $projectRoot 'runs\logs\m17_wait.log'
$run = Join-Path $PSScriptRoot 'run_m17_sweep.ps1'
function Write-Log([string]$m) { "$((Get-Date).ToString('s')) $m" | Tee-Object -FilePath $log -Append }

$deadline = (Get-Date).AddHours($MaxWaitHours)
$ok = 0
Write-Log "[wait] start PMax=$PMax% need=$Need x ${IntervalSec}s"
while ($ok -lt $Need) {
    if ((Get-Date) -gt $deadline) { Write-Log "[wait] timeout after $MaxWaitHours h - not started"; exit 3 }
    $s = (Get-Counter '\Processor(*)\% Processor Time' -SampleInterval 5 -MaxSamples 2).CounterSamples |
        Where-Object { $_.InstanceName -match '^\d+$' -and [int]$_.InstanceName -in 0,2,4,6 }
    $p = ($s.CookedValue | Measure-Object -Average).Average
    $gpu = [int]((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits).Trim())
    $ramGb = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $free = ($p -lt $PMax) -and ($gpu -lt 10) -and ($ramGb -gt 8)
    if ($free) { $ok++ } else { $ok = 0 }
    Write-Log ("[wait] P0246={0:N0}% gpu={1}% ram={2:N1}GB free={3} streak={4}/{5}" -f $p, $gpu, $ramGb, $free, $ok, $Need)
    if ($ok -lt $Need) { Start-Sleep -Seconds $IntervalSec }
}
Write-Log "[wait] resources free - starting sweep (strength 0,0.5,1,2; batch 64; workers 2; mask 0xFF)"
pwsh -NoProfile -File $run -Strengths 0,0.5,1,2 -Epochs 30 -Batch 64 -Workers 2 -Mask 255 -Prefix m17
$code = $LASTEXITCODE
Write-Log "[wait] sweep finished exit=$code"
exit $code
