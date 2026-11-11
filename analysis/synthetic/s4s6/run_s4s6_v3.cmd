@echo off
rem M2-10 (b) v3 gprMax run for one input (gprMax 4.0.1, 2D TMz, GPU 0). Batch-1 and v2 runners are unchanged.
rem Usage:  cmd /c analysis\synthetic\s4s6\run_s4s6_v3.cmd <name> <ntraces> [pilot]
rem   full models : <name> = S6P_*_J* / S4N_* (models_s4s6_v3.json) with 161; references BG_N_* / BG_PF_* (S038, S042, S048, S012) with 3
rem   pilot       : third argument "pilot"; <name> = L_S6P_NP_<sigma>_<J>_T15 with 161 (15 ns Lambda line),
rem                 P_S6P_<model>, P_S6P_NP_<sigma>_<J>, P_S4N_<model> with 1, P_S4N_<model>_X5 with 5
rem Inputs: analysis\synthetic\s4s6\inputs_v3\ (full, references) or ...\pilot_v3\ (pilot)
rem Output: data\synthetic_gprmax_s4s6_v3\h5\ (or ...\pilot\) <name>_merged.h5 (n > 1) or <name>.h5 (n = 1), plus <name>.log
rem Affinity/priority are set by the caller (compute-scheduler-expert plan), not here.
setlocal EnableDelayedExpansion
if "%~2"=="" (echo usage: run_s4s6_v3.cmd ^<name^> ^<ntraces^> [pilot] & exit /b 2)
set M=%~1
set N=%~2
set ROOT=%~dp0..\..\..
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate gprmax-acceleration >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if /i "%~3"=="pilot" (
  set IN=%ROOT%\analysis\synthetic\s4s6\pilot_v3\%M%.in
  set OUT=%ROOT%\data\synthetic_gprmax_s4s6_v3\pilot
) else (
  set IN=%ROOT%\analysis\synthetic\s4s6\inputs_v3\%M%.in
  set OUT=%ROOT%\data\synthetic_gprmax_s4s6_v3\h5
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
