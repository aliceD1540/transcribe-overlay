@echo off
REM TranscribeOverlay - Windows launcher script
REM Creates virtual environment, installs dependencies, and starts the application

setlocal enabledelayedexpansion

echo.
echo ================================
echo TranscribeOverlay - Launcher
echo ================================
echo.

REM Check if Python is available
REM Try py.exe first (Windows Python Launcher)
py -3 --version >nul 2>&1
if errorlevel 1 (
    REM Fallback to python command
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found.
        echo Please install Python 3.8 or later.
        echo https://www.python.org/
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
) else (
    set PYTHON_CMD=py -3
)

echo [OK] Python found
%PYTHON_CMD% --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo [*] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo.
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Upgrade pip
echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip -q
if errorlevel 1 (
    echo [WARNING] pip upgrade failed, continuing anyway...
)

REM Install dependencies
echo.
echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully

REM Start application
echo.
echo [*] Starting TranscribeOverlay...
echo.
python main.py

REM Error handling
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code: !errorlevel!
    pause
    exit /b 1
)

endlocal
