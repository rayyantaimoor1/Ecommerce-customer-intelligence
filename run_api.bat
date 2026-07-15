@echo off
REM ============================================================
REM run_api.bat
REM Double-click this file to launch the FastAPI prediction server.
REM Assumes you already ran the notebooks (01-06) at least once,
REM so models\clustering_pipeline.joblib exists.
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo  E-Commerce Customer Segmentation API
echo ============================================================
echo.

if not exist "models\clustering_pipeline.joblib" (
    echo [ERROR] models\clustering_pipeline.joblib not found.
    echo Run notebooks 01 through 06 in the notebooks\ folder first,
    echo then re-run this file.
    echo.
    pause
    exit /b 1
)

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] "conda" was not found on your PATH.
    echo Open this file from an Anaconda Prompt instead.
    echo.
    pause
    exit /b 1
)

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

echo.
echo Starting FastAPI on http://127.0.0.1:8000
echo Swagger docs at http://127.0.0.1:8000/docs
echo Close this window (or press Ctrl+C) to stop the server.
echo.
cd app
uvicorn main:app --reload

pause
