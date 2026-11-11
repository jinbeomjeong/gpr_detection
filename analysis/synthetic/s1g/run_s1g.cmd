@echo off
rem S1g full B-scan run for one model (gprMax 4.0.1, 2D TMz, GPU 0).
rem Usage:  cmd /c analysis\synthetic\s1g\run_s1g.cmd S1G_D060
rem         (models: S1G_D060 S1G_D100 S1G_C1 S1G_BGL; 161 traces each, x = 0.40 .. 3.60 m step 0.02 m;
rem          S1G_BGL is the no-pipe background line for the G4 null test)
rem Output: data\synthetic_gprmax_s1g\h5\<model>_merged.h5 and <model>.log
rem Background (S1G_bg_merged.h5, 3 traces) and free-space reference (S1G_fs.h5) were produced by the pilot.
rem Affinity/priority are set by the caller (compute-scheduler-expert plan), not here.
setlocal EnableDelayedExpansion
if "%~1"=="" (echo usage: run_s1g.cmd ^<model^> & exit /b 2)
set M=%~1
set ROOT=%~dp0..\..\..
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate gprmax-acceleration >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set IN=%ROOT%\analysis\synthetic\s1g\inputs\%M%.in
set OUT=%ROOT%\data\synthetic_gprmax_s1g\h5
if not exist "%OUT%" mkdir "%OUT%"
cd /d "%OUT%"
echo start %M% !DATE! !TIME!
python -m gprMax "%IN%" -gpu 0 -n 161 -o "%OUT%\%M%.h5" --geometry-fixed > "%OUT%\%M%.log" 2>&1 < nul
set RC=!ERRORLEVEL!
echo gprMax %M% exit !RC! !DATE! !TIME!
if not "!RC!"=="0" exit /b !RC!
python -m gprMax.toolboxes.Utilities.outputfiles_merge "%OUT%\%M%" --remove-files < nul
set RC=!ERRORLEVEL!
echo merge %M% exit !RC! !DATE! !TIME!
exit /b !RC!
