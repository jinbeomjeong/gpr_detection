@echo off
rem M2-10 K7 gprMax run for one input (gprMax 4.0.1, 2D TMz, GPU 0).
rem Usage:  cmd /c analysis\synthetic\k7\run_k7.cmd <name> <ntraces> [pilot]
rem   full models : <name> = K7_* (models_k7.json) with 161; references K7_REF_S001 / S002 / S003 with 3
rem   pilot       : third argument "pilot"; <name> = P_K7_* (pilot\, traces per models_k7.json pilot.traces_x_m)
rem                 or K7_REF_* with 3 (input from inputs\, output to the pilot folder)
rem Inputs: analysis\synthetic\k7\inputs\ or ...\pilot\ (pilot\<name>.in is used first when "pilot" is given)
rem Output: data\synthetic_gprmax_k7\h5\ (or ...\pilot\) <name>_merged.h5 (n > 1) or <name>.h5 (n = 1), plus <name>.log
rem Affinity/priority are set by the caller (compute-scheduler-expert plan), not here.
setlocal EnableDelayedExpansion
if "%~2"=="" (echo usage: run_k7.cmd ^<name^> ^<ntraces^> [pilot] & exit /b 2)
set M=%~1
set N=%~2
set ROOT=%~dp0..\..\..
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate gprmax-acceleration >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set IN=%ROOT%\analysis\synthetic\k7\inputs\%M%.in
if /i "%~3"=="pilot" (
  if exist "%ROOT%\analysis\synthetic\k7\pilot\%M%.in" set IN=%ROOT%\analysis\synthetic\k7\pilot\%M%.in
  set OUT=%ROOT%\data\synthetic_gprmax_k7\pilot
) else (
  set OUT=%ROOT%\data\synthetic_gprmax_k7\h5
)
if not exist "!IN!" (echo missing input !IN! & exit /b 2)
if not exist "!OUT!" mkdir "!OUT!"
cd /d "!OUT!"
echo start %M% n=%N% !DATE! !TIME!
if "%N%"=="1" (
  python -m gprMax "!IN!" -gpu 0 -o "!OUT!\%M%.h5" > "!OUT!\%M%.log" 2>&1 < nul
) else (
  python -m gprMax "!IN!" -gpu 0 -n %N% -o "!OUT!\%M%.h5" --geometry-fixed > "!OUT!\%M%.log" 2>&1 < nul
)
set RC=!ERRORLEVEL!
echo gprMax %M% exit !RC! !DATE! !TIME!
if not "!RC!"=="0" exit /b !RC!
if "%N%"=="1" exit /b 0
python -m gprMax.toolboxes.Utilities.outputfiles_merge "!OUT!\%M%" --remove-files < nul
set RC=!ERRORLEVEL!
echo merge %M% exit !RC! !DATE! !TIME!
exit /b !RC!
