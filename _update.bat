@echo off
setlocal enabledelayedexpansion

echo.
echo === StereoCrafter Modern Update Script ===
echo.

:: Change to the directory of this batch file
cd /d "%~dp0"

:: 1. Git Pull
echo [1/3] Pulling latest changes from GitHub...
git pull
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Git pull failed. You may have local changes that conflict.
    echo Please commit or stash your changes and try again.
    pause
    exit /b %errorlevel%
)

:: 2. Update Submodules
:: Since StereoCrafter uses submodules, we MUST update them too 
:: in case the developer updated a submodule link.
echo [2/3] Updating submodules...
git submodule update --init --recursive

:: 3. UV Sync
echo [3/3] Syncing dependencies with uv...

:: Verify uv is installed (just in case they moved to a new machine)
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

:: Run the sync
uv sync

if %errorlevel% equ 0 (
    echo.
    echo =================================================
    echo UPDATE SUCCESSFUL
    echo Project and environment are now up to date.
    echo =================================================
    
    :: Check for the old redundant venv folder
    if exist "venv" (
        echo.
        echo [NOTE] A legacy 'venv' folder was detected. 
        echo This project now uses '.venv' (with a dot) via UV.
        echo The 'venv' folder is now REDUNDANT and can be safely deleted 
        echo to save several GBs of disk space.
    )
) else (
    echo.
    echo [WARNING] uv sync failed. Your environment might be inconsistent.
)

echo.
pause