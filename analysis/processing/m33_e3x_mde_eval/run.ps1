param([string[]]$Args2)
# Run python-dev at BelowNormal on a single idle core; args passed through.
$env:MKL_THREADING_LAYER = 'TBB'; $env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
$d = 'C:\Users\jinbeom\AppData\Local\Temp\claude\8d33c416c891208f\f5e47223-b6d7-4da2-b289-9fbecf2c5aa0\scratchpad\stats-e3x-mde'
$py = 'C:\ProgramData\miniforge3\envs\python-dev\python.exe'
$tag = [IO.Path]::GetFileNameWithoutExtension($Args2[1])
$p = Start-Process -FilePath $py -ArgumentList (@('-I', '-B') + $Args2) -PassThru -NoNewWindow -RedirectStandardOutput "$d\log_$tag.out" -RedirectStandardError "$d\log_$tag.err"
$p.PriorityClass = 'BelowNormal'; $p.ProcessorAffinity = [IntPtr]0x8000
$p.WaitForExit(590000) | Out-Null
if (-not $p.HasExited) { "still running pid=$($p.Id)"; exit 9 }
"rc=$($p.ExitCode) pid=$($p.Id)"; Get-Content "$d\log_$tag.out"; Get-Content "$d\log_$tag.err" | Select-Object -Last 10
