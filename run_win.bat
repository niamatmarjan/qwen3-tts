@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  Starting Qwen3-TTS for Windows (NVIDIA CUDA / CPU)
echo ========================================================

cd /d "%~dp0"

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] uv detected.
    if not exist ".venv" (
        echo [+] Creating Python 3.12 virtual environment...
        uv venv .venv --python 3.12
    )
    call .venv\Scripts\activate.bat
    echo [+] Installing/updating dependencies...
    uv pip install -e .
) else (
    echo [!] uv not found. Falling back to standard Python...
    if not exist ".venv" (
        echo [+] Creating Python virtual environment...
        python -m venv .venv
    )
    call .venv\Scripts\activate.bat
    echo [+] Installing/updating dependencies...
    pip install -e .
)

echo.
echo [+] Launching Unified Web UI...
python app.py %*

pause
