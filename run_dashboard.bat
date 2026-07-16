@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM run_dashboard.bat
REM Double-click this file (from Explorer, cmd, or Anaconda Prompt)
REM to launch the Streamlit dashboard.
REM Assumes you already ran the notebooks (01-06) at least once,
REM so models\clustering_pipeline.joblib exists.
REM ============================================================

REM Move to the folder this .bat file lives in, regardless of
REM where it was double-clicked from.
cd /d "%~dp0"

echo ============================================================
echo  E-Commerce Customer Intelligence Dashboard
echo ============================================================
echo.

REM --- Check the trained model exists before trying to launch ---
if not exist "models\clustering_pipeline.joblib" (
    echo [ERROR] models\clustering_pipeline.joblib not found.
    echo Run notebooks 01 through 06 in the notebooks\ folder first,
    echo then re-run this file.
    echo.
    pause
    exit /b 1
)

REM --- Find conda even when this is opened from plain cmd/Explorer,
REM     where conda usually isn't on PATH. We try, in order:
REM     1. conda already on PATH (works if opened from Anaconda Prompt)
REM     2. common per-user and system-wide Anaconda/Miniconda install folders
set "CONDA_ROOT="

where conda >nul 2>nul
if not errorlevel 1 goto :activate

if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\anaconda3" & goto :activate
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\miniconda3" & goto :activate
if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=%LOCALAPPDATA%\anaconda3" & goto :activate
if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=%LOCALAPPDATA%\miniconda3" & goto :activate
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\anaconda3" & goto :activate
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\miniconda3" & goto :activate
if exist "C:\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\anaconda3" & goto :activate
if exist "C:\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\miniconda3" & goto :activate

echo [ERROR] Could not find a conda installation automatically.
echo Either open this file from an Anaconda Prompt instead, or
echo edit this .bat file and add your conda install folder to the
echo list of paths it checks near the top of the file.
echo.
pause
exit /b 1

:activate
echo Activating conda environment: ecomm-clustering
if defined CONDA_ROOT (
    call "%CONDA_ROOT%\Scripts\activate.bat" ecomm-clustering
) else (
    call conda activate ecomm-clustering
)

if errorlevel 1 (
    echo [ERROR] Could not activate environment "ecomm-clustering".
    echo Create it first from an Anaconda Prompt:
    echo     conda create -n ecomm-clustering python=3.11
    echo     conda activate ecomm-clustering
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM --- Launch Streamlit ---
echo.
echo Starting Streamlit... your browser will open automatically.
echo Close this window (or press Ctrl+C) to stop the dashboard.
echo.
cd dashboard
streamlit run Home.py

pause
