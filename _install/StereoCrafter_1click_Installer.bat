@echo off
setlocal enabledelayedexpansion
REM StereoCrafter Universal Smart Installer (v4.2 - Fixed Log Pathing)

:: --- LOG INITIALIZATION ---
set "LOGFILE=%~dp0install_log.txt"

:: --- HEALTH CHECK (Pre-flight Permissions) ---
echo [0/7] Verifying Environment...

:: Check if running from a ZIP
echo %~dp0 | findstr /i "Temp" >nul
if !errorlevel! equ 0 (
    echo [WARNING] It looks like you are running this from a temporary folder or a ZIP.
    echo Please EXTRACT the folder to your Desktop or a permanent location first.
)

:: Check Write Permissions
echo test > "%~dp0write_test.txt" 2>nul
if !errorlevel! neq 0 (
    echo [ERROR] Access Denied! Cannot write to this folder.
    echo Please move the StereoCrafter folder to your Desktop or Documents.
    echo Or try right-clicking this script and selecting 'Run as Administrator'.
    pause && exit /b 1
)
del "%~dp0write_test.txt"

:: Check for Admin (Required for persistent pathing/winget)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Note: Not running as Administrator. Some features (winget) might prompt for permission.
)

:: Clear the log and start the session
echo [%date% %time%] --- NEW INSTALLATION/UPDATE SESSION --- > "%LOGFILE%"


call :log "[1/7] Checking for Git..."
where git >nul 2>&1
if %errorlevel% neq 0 (
    call :log "[INFO] Git not found. Attempting install via Winget..."
    winget install --id Git.Git -e --source winget >> "%LOGFILE%" 2>&1
    if %errorlevel% neq 0 (
        call :log "[ERROR] Git install failed. Please install manually."
        pause && exit /b 1
    )
    set "PATH=%PATH%;C:\Program Files\Git\cmd"
)

call :log "[2/7] Checking for UV..."
where uv >nul 2>&1
if %errorlevel% neq 0 (
    call :log "[INFO] uv not found. Installing..."
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex" >> "%LOGFILE%" 2>&1
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

call :log "[3/7] Analyzing Path..."
for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"

if /i "!CUR_NAME!"=="_install" (
    call :log "[INFO] Moving out of _install folder..."
    cd ..
    for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"
)

set "PREFIX=!CUR_NAME:~0,13!"
if /i "!PREFIX!"=="StereoCrafter" (
    if exist "pyproject.toml" (
        call :log "[INFO] Project folder verified: !CUR_NAME!"
        set "ALREADY_HOME=true"
    ) else (
        call :log "[INFO] Shell folder detected. Moving up..."
        cd ..
        set "ALREADY_HOME=false"
    )
) else (
    set "ALREADY_HOME=false"
)

call :log "[4/7] Repository Selection..."

:: Define URLs
set "Bill8_URL=https://github.com/Billynom8/StereoCrafter.git"
set "ENOKY_URL=https://github.com/enoky/StereoCrafter.git"

echo.
echo =========================================================
echo REPOSITORY SELECTION
echo =========================================================
echo 1. Billynom8 - !Bill8_URL! [DEFAULT]
echo 2. enoky    - !ENOKY_URL!
echo.
set "repo_choice=1"
set /p repo_choice="Select source (1 or 2): "

set "SELECTED_URL=!Bill8_URL!"
set "PULL_REMOTE=origin"
if "!repo_choice!"=="2" (
    set "SELECTED_URL=!ENOKY_URL!"
    set "PULL_REMOTE=upstream"
)

if "!ALREADY_HOME!"=="false" (
    set "FOUND_SUB="
    for /d %%D in (StereoCrafter*) do (
        if exist "%%D\pyproject.toml" set "FOUND_SUB=%%D"
    )

    if defined FOUND_SUB (
        call :log "[INFO] Found project in subfolder: !FOUND_SUB!"
        cd "!FOUND_SUB!"
        set "ALREADY_HOME=true"
    ) else (
        call :log "[INFO] Cloning fresh repository from !SELECTED_URL!..."
        git clone --recurse-submodules !SELECTED_URL! >> "%LOGFILE%" 2>&1
        if !errorlevel! neq 0 (
            call :log "[ERROR] Git clone failed. Check log for details: %LOGFILE%"
            pause && exit /b 1
        )
        cd StereoCrafter
        set "ALREADY_HOME=true"
    )
)

:: Standardize Remotes (matching _update.bat structure)
where git >nul 2>&1
if %errorlevel% equ 0 (
    if exist ".git" (
        call :log "[INFO] Standardizing remotes (origin=Billynom8, upstream=enoky)..."
        git remote set-url origin !Bill8_URL! 2>nul || git remote add origin !Bill8_URL! 2>nul
        git remote set-url upstream !ENOKY_URL! 2>nul || git remote add upstream !ENOKY_URL! 2>nul
    )
)

:: Optional Update Pull
if "!ALREADY_HOME!"=="true" (
    echo.
    set /p do_pull="Detected existing repo. Pull latest code from !PULL_REMOTE!? (Y/N) [Default=N]: "
    if /i "!do_pull!"=="Y" (
        call :log "[INFO] Pulling updates from !PULL_REMOTE!..."
        
        :: Detect branch (main or master)
        git fetch !PULL_REMOTE! main --quiet 2>nul
        if !errorlevel! equ 0 ( set "BR=main" ) else ( set "BR=master" )
        
        git pull !PULL_REMOTE! !BR! >> "%LOGFILE%" 2>&1
        if !errorlevel! neq 0 (
            echo.
            echo [WARNING] Git pull failed. This usually means you have local changes.
            set /p force_pull="Would you like to DISCARD your local changes and force update? (Y/N): "
            if /i "!force_pull!"=="Y" (
                call :log "[INFO] Forcing update (reset --hard)..."
                git reset --hard !PULL_REMOTE!/!BR! >> "%LOGFILE%" 2>&1
                git pull !PULL_REMOTE! !BR! >> "%LOGFILE%" 2>&1
            ) else (
                call :log "[SKIP] User declined force update. Manual resolution required."
            )
        )
    )
)


call :log "[5/7] Setting up Environment..."
if exist "venv" (
    call :log "[INFO] Legacy 'venv' detected."
    echo.
    echo This project now uses UV and '.venv' [with a dot].
    set /p del_venv="Would you like to delete the old 'venv' to save space? (Y/N): "
    if /i "!del_venv!"=="Y" (
        call :log "[INFO] Removing legacy venv..."
        rmdir /s /q venv >> "%LOGFILE%" 2>&1
    )
)

call :log "[INFO] Pinning Python 3.12 and syncing..."
uv python pin 3.12 >> "%LOGFILE%" 2>&1
uv sync >> "%LOGFILE%" 2>&1

REM --- [6/7] WEIGHTS SECTION (Parenthesis Safe) ---
echo.
echo =========================================================
echo MODEL WEIGHTS DOWNLOAD
echo =========================================================
set /p get_weights="Would you like to download weights now? (Y/N): "

if /i "!get_weights!"=="Y" (
    call :log "[INFO] Starting weight downloads..."
    if not exist "weights" mkdir weights

    set /p hf_login="Do you need to log in to Hugging Face? (Y/N): "
    if /i "!hf_login!"=="Y" (
        echo [PROMPT] Please paste your Hugging Face Access Token below:
        uv run hf auth login
    )

    call :log "[INFO] Downloading Models..."
    uv run hf download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir weights/stable-video-diffusion-img2vid-xt-1-1
    uv run hf download tencent/DepthCrafter --local-dir weights/DepthCrafter
    uv run hf download TencentARC/StereoCrafter --local-dir weights/StereoCrafter
    
    call :log "[SUCCESS] Weights downloaded."
) else (
    call :log "[SKIP] User skipped weights."
)

call :log "[7/7] Finalizing..."
echo.
echo INSTALLATION SUCCESSFUL
echo Log location: %LOGFILE%
call :log "[FINISH] Session complete."

pause
exit /b

:: --- THE LOGGING FUNCTION ---
:log
echo %~1
:: Use double quotes around the logfile path to handle spaces in folder names
echo [%time%] %~1 >> "%LOGFILE%"
exit /b