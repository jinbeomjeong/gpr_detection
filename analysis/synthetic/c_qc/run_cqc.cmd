@echo off
rem M2-10 (c) P3 gprMax run for one input (gprMax 4.0.1, 2D TMz, GPU 0).
rem Usage:  cmd /c analysis\synthetic\c_qc\run_cqc.cmd <name> <ntraces> [pilot]
rem   <name> = Z_HSTEP / Z_HWAVE with 161, Z_REF_H / Z_FS_H with 21 (per-height libraries)
rem   "pilot" writes to data\synthetic_gprmax_c_qc\pilot\ instead of ...\h5\ (same inputs; spec 6 pilots whole lines)
rem The inputs set the antenna height per A-scan in a #python block (current_model_run), so gprMax runs
rem WITHOUT --geometry-fixed (the input is re-parsed for every A-scan).
rem Output: <out>\<name>_merged.h5 and <name>.log. Affinity/priority are set by the caller.
setlocal EnableDelayedExpansion
if "%~2"=="" (echo usage: run_cqc.cmd ^<name^> ^<ntraces^> [pilot] & exit /b 2)
set M=%~1
set N=%~2
set ROOT=%~dp0..\..\..
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate gprmax-acceleration >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set IN=%ROOT%\analysis\synthetic\c_qc\inputs\%M%.in
if /i "%~3"=="pilot" (
  set OUT=%ROOT%\data\synthetic_gprmax_c_qc\pilot
) else (
  set OUT=%ROOT%\data\synthetic_gprmax_c_qc\h5
)
if not exist "!IN!" (echo missing input !IN! & exit /b 2)
if not exist "!OUT!" mkdir "!OUT!"
cd /d "!OUT!"
echo start %M% n=%N% !DATE! !TIME!
python -m gprMax "!IN!" -gpu 0 -n %N% -o "!OUT!\%M%.h5" > "!OUT!\%M%.log" 2>&1 < nul
set RC=!ERRORLEVEL!
echo gprMax %M% exit !RC! !DATE! !TIME!
if not "!RC!"=="0" exit /b !RC!
python -m gprMax.toolboxes.Utilities.outputfiles_merge "!OUT!\%M%" --remove-files < nul
set RC=!ERRORLEVEL!
echo merge %M% exit !RC! !DATE! !TIME!
exit /b !RC!
