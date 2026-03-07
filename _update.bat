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
set "Bill8_URL=https://github.com/Billynom8/StereoCrafter.git"
set "ENOKY_URL=https://github.com/enoky/StereoCrafter.git"

:: Check if origin points to either repo, if not set it
set "ORIGIN_URL=Not Found"
for /f "tokens=*" %%i in ('git remote get-url origin 2^>nul') do set "ORIGIN_URL=%%i"

:: Check if upstream points to either repo, if not set it
set "UPSTREAM_URL=Not Found"
for /f "tokens=*" %%i in ('git remote get-url upstream 2^>nul') do set "UPSTREAM_URL=%%i"

:: Configure origin if not set or pointing to wrong repo
if "!ORIGIN_URL!"=="Not Found" (
    git remote add origin !Bill8_URL! 2>nul
    set "ORIGIN_URL=!Bill8_URL!"
) else (
    echo !ORIGIN_URL! | findstr /i "Billynom8" >nul
    if !errorlevel! neq 0 (
        git remote set-url origin !Bill8_URL! 2>nul
        set "ORIGIN_URL=!Bill8_URL!"
    )
)

:: Configure upstream if not set or pointing to wrong repo
if "!UPSTREAM_URL!"=="Not Found" (
    git remote add upstream !ENOKY_URL! 2>nul
    set "UPSTREAM_URL=!ENOKY_URL!"
) else (
    echo !UPSTREAM_URL! | findstr /i "enoky" >nul
    if !errorlevel! neq 0 (
        git remote set-url upstream !ENOKY_URL! 2>nul
        set "UPSTREAM_URL=!ENOKY_URL!"
    )
)

:: --- Configuration Selection ---
echo [1/4] Please Select Update Source:
echo 1. Billynom8 - !Bill8_URL!
echo 2. enoky    - !ENOKY_URL!
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

:: Safety check - warn if pulling from unexpected repo
set "CHECK_URL=!Bill8_URL!"
if "!REMOTE!"=="upstream" set "CHECK_URL=!ENOKY_URL!"

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
    echo [WARNING] Git pull failed. This usually means you have local changes that conflict.
    echo.
    set /p force_pull="Would you like to DISCARD your local changes and force update? (Y/N): "
    if /i "!force_pull!"=="Y" (
        echo [INFO] Forcing update (reset --hard)...
        git reset --hard !REMOTE!/!BRANCH!
        git pull !REMOTE! !BRANCH!
    ) else (
        echo.
        echo [ERROR] Update aborted. Please commit or stash your changes manually.
        pause
        exit /b 1
    )
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