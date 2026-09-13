@echo off
REM TranscribeOverlay - Windows実行スクリプト
REM 仮想環境の作成、依存パッケージのインストール、アプリケーションの起動

setlocal enabledelayedexpansion

echo.
echo ================================
echo TranscribeOverlay - Launcher
echo ================================
echo.

REM Python が利用可能か確認
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python が見つかりません。
    echo Python 3.8 以上をインストールしてください。
    echo https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM 仮想環境がなければ作成
if not exist "venv" (
    echo.
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM 仮想環境を有効化
echo.
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM pip をアップグレード
echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip -q
if errorlevel 1 (
    echo [WARNING] pip upgrade failed, continuing anyway...
)

REM 依存パッケージをインストール
echo.
echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully

REM アプリケーションを起動
echo.
echo [*] Starting TranscribeOverlay...
echo.
python main.py

REM エラー処理
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code: !errorlevel!
    pause
    exit /b 1
)

endlocal
