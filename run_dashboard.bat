@echo off
REM ============================================================
REM run_dashboard.bat
REM Double-click this file to launch the Streamlit dashboard.
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

REM --- Check conda is available ---
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] "conda" was not found on your PATH.
    echo Open this file from an Anaconda Prompt instead, or add
    echo conda to your PATH via Anaconda Navigator's environment settings.
    echo.
    pause
    exit /b 1
)

REM --- Activate the project environment ---
echo Activating conda environment: ecomm-clustering
call conda activate ecomm-clustering
if errorlevel 1 (
    echo [ERROR] Could not activate environment "ecomm-clustering".
    echo Create it first:
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
