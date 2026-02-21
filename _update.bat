@echo off
setlocal enabledelayedexpansion

echo.
echo =================================================
echo === StereoCrafter Modern Update Script ===
echo =================================================
echo.

:: Change to the directory of this batch file
cd /d "%~dp0"

:: --- Detect Remotes ---
set "ORIGIN_URL=Not Found"
for /f "tokens=*" %%i in ('git remote get-url origin 2^>nul') do set "ORIGIN_URL=%%i"

set "UPSTREAM_URL=Not Found"
for /f "tokens=*" %%i in ('git remote get-url upstream 2^>nul') do set "UPSTREAM_URL=%%i"

:: --- Configuration Selection ---
echo [1/4] Please Select Update Source:
echo 1. origin    - !ORIGIN_URL!
if not "!UPSTREAM_URL!"=="Not Found" (
    echo 2. upstream  - !UPSTREAM_URL!
) else (
    echo 2. [Not Configured]
)
echo.

set "REMOTE=origin"
set /p choice="Select source (1 or 2) [Default=1]: "

if "%choice%"=="2" (
    if "!UPSTREAM_URL!"=="Not Found" (
        echo [ERROR] Upstream remote is not configured. Falling back to origin.
        set "REMOTE=origin"
    ) else (
        set "REMOTE=upstream"
    )
)

:: Safety check for TencentARC
set "CHECK_URL=!ORIGIN_URL!"
if "!REMOTE!"=="upstream" set "CHECK_URL=!UPSTREAM_URL!"
echo !CHECK_URL! | findstr /i "TencentARC" >nul
if %errorlevel% equ 0 (
    echo [NOTE] You are updating from the official TencentARC repository.
)

echo.
echo [2/4] Pulling latest changes from !REMOTE!...
:: Fetch first to see if branch exists
git fetch !REMOTE! main --quiet 2>nul
if %errorlevel% equ 0 (
    set "BRANCH=main"
) else (
    set "BRANCH=master"
)

git pull !REMOTE! !BRANCH!

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Git pull failed. You may have local changes that conflict.
    echo Please commit or stash your changes and try again.
    pause
    exit /b %errorlevel%
)

:: --- Update Submodules ---
echo.
echo [3/4] Updating submodules...
git submodule update --init --recursive

:: --- UV Sync ---
echo.
echo [4/4] Syncing dependencies with uv...

:: Verify uv is installed
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
    echo Source: !REMOTE! (!BRANCH!)
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