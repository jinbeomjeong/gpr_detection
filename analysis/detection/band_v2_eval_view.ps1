# Band merge v2 comparison (docs/experiments/band_merge_v2_compare.md):
# evaluate one training run against one ground truth (v1 or v2 labels)
# without touching the run's own eval/ folder.
#
# scripts/eval_detector.py always writes <run-dir>/eval/<weights>_<unit>/, so
# evaluating the same run against two label sets would overwrite one with the
# other. This helper makes a "view" folder per (run, ground truth):
#   <ViewRoot>\<run name>__gt_<Gt>\gpr_train_settings.yaml   (copy)
#   <ViewRoot>\<run name>__gt_<Gt>\weights\last.pt           (hard link)
# and runs eval_detector.py on the view with the matching dataset folder and
# manifest. The fold is passed explicitly (0); the test hold-out stays sealed.
#
# Usage:
#   pwsh -NoProfile -File analysis\detection\band_v2_eval_view.ps1 `
#       -RunDir runs\detect\m17_aug0 -Gt v2 [-Units crop,image] [-ViewRoot runs\detect\band_v2_eval]
# Exit codes: eval_detector.py's (0 done, 1 input error, 2 test refused);
# 1 also when the source run or its last.pt is missing.
param(
    [Parameter(Mandatory)][string]$RunDir,
    [Parameter(Mandatory)][ValidateSet('v1', 'v2')][string]$Gt,
    [string]$Units = 'crop,image',
    [string]$ViewRoot = 'runs\detect\band_v2_eval',
    [string]$ProjectRoot = 'C:\workspace\gpr_detection',
    [double]$ConfusionConf = 0.25
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-Location $ProjectRoot

$datasets = @{
    v1 = @{ Dir = 'data\aihub_yolo_v1'; Manifest = 'data\splits\aihub_yolo_v1.csv' }
    v2 = @{ Dir = 'data\aihub_yolo_v2_auto32'; Manifest = 'data\splits\aihub_yolo_v2_auto32.csv' }
}
$src = (Resolve-Path -LiteralPath $RunDir -ErrorAction SilentlyContinue)
if ($null -eq $src) { Write-Output "[error] run folder not found: $RunDir"; exit 1 }
$src = $src.Path
$srcWeights = Join-Path $src 'weights\last.pt'
$srcSettings = Join-Path $src 'gpr_train_settings.yaml'
foreach ($f in @($srcWeights, $srcSettings)) {
    if (-not (Test-Path -LiteralPath $f)) { Write-Output "[error] missing $f"; exit 1 }
}

$viewRootFull = if ([IO.Path]::IsPathRooted($ViewRoot)) { $ViewRoot } else { Join-Path $ProjectRoot $ViewRoot }
$view = Join-Path $viewRootFull ("{0}__gt_{1}" -f (Split-Path $src -Leaf), $Gt)
New-Item -ItemType Directory -Force -Path (Join-Path $view 'weights') | Out-Null
Copy-Item -LiteralPath $srcSettings -Destination (Join-Path $view 'gpr_train_settings.yaml') -Force
$viewWeights = Join-Path $view 'weights\last.pt'
if (Test-Path -LiteralPath $viewWeights) { Remove-Item -LiteralPath $viewWeights -Force }
New-Item -ItemType HardLink -Path $viewWeights -Target $srcWeights | Out-Null
"source_run: $src`nground_truth: $Gt`ndataset_dir: $($datasets[$Gt].Dir)`nmanifest: $($datasets[$Gt].Manifest)`n" |
    Set-Content -LiteralPath (Join-Path $view 'view_info.txt') -Encoding utf8

$conda = 'C:\ProgramData\miniforge3\Scripts\conda.exe'
$evalArgs = @('run', '--no-capture-output', '-n', 'gpr-dev', 'python', '-I',
    (Join-Path $ProjectRoot 'scripts\eval_detector.py'),
    '--run-dir', $view, '--weights', 'last', '--unit', $Units, '--fold', '0',
    '--dataset-dir', (Join-Path $ProjectRoot $datasets[$Gt].Dir),
    '--manifest', (Join-Path $ProjectRoot $datasets[$Gt].Manifest),
    '--confusion-conf', $ConfusionConf.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--device', '0')
Write-Output "[eval] $view"
$ErrorActionPreference = 'Continue'
& $conda @evalArgs 2>&1 | ForEach-Object { "$_" }
exit $LASTEXITCODE
