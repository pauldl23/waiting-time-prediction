@echo off
:: Use pushd to handle the directory safely (avoids trailing backslash issues)
pushd "%~dp0"

echo ===================================================
echo   WaitStudio | Waiting Time Prediction System
echo ===================================================
echo.

:: Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b
)

:: Check if requirements are installed
echo [INFO] Verifying dependencies...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

echo.
echo [INFO] Starting WaitStudio Dashboard...
echo [INFO] The application will open in your default browser.
echo.

python -m streamlit run app.py

pause
