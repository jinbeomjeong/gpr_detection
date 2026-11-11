@echo off
rem S4/S6 gprMax run for one input (gprMax 4.0.1, 2D TMz, GPU 0).
rem Usage:  cmd /c analysis\synthetic\s4s6\run_s4s6.cmd <name> <ntraces> [pilot]
rem   full models : <name> = S4N_* / S4S_* / S6P_* with 161; references BG_* with 3, FS with 1
rem   pilot       : third argument "pilot"; <name> = P_S6P_D15_S030 or P_S4N_D15_S030 with 2,
rem                 P_S6P_D15_S040, P_S6P_NP_S040 or P_S4N_D15_S040 with 1, L_S6P_NP_S040_T15 with 161 (15 ns Lambda line)
rem Output: data\synthetic_gprmax_s4s6\h5\ (or ...\pilot\) <name>_merged.h5 (n > 1) or <name>.h5 (n = 1), plus <name>.log
rem Affinity/priority are set by the caller (compute-scheduler-expert plan), not here.
setlocal EnableDelayedExpansion
if "%~2"=="" (echo usage: run_s4s6.cmd ^<name^> ^<ntraces^> [pilot] & exit /b 2)
set M=%~1
set N=%~2
set ROOT=%~dp0..\..\..
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate gprmax-acceleration >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if /i "%~3"=="pilot" (
  set IN=%ROOT%\analysis\synthetic\s4s6\pilot\%M%.in
  set OUT=%ROOT%\data\synthetic_gprmax_s4s6\pilot
) else (
  set IN=%ROOT%\analysis\synthetic\s4s6\inputs\%M%.in
  set OUT=%ROOT%\data\synthetic_gprmax_s4s6\h5
)
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
