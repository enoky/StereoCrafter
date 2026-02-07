@echo off
setlocal enabledelayedexpansion
REM StereoCrafter Universal Smart Installer (v4.2 - Fixed Log Pathing)

:: --- LOG INITIALIZATION (The "Absolute" Fix) ---
:: %~dp0 is the folder where THIS script is located.
:: We set the log to always live next to the script, no matter where we 'cd' to.
set "LOGFILE=%~dp0install_log.txt"

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

call :log "[4/7] Repository Check..."
if "!ALREADY_HOME!"=="false" (
    set "FOUND_SUB="
    for /d %%D in (StereoCrafter*) do (
        if exist "%%D\pyproject.toml" set "FOUND_SUB=%%D"
    )

    if defined FOUND_SUB (
        call :log "[INFO] Found project in subfolder: !FOUND_SUB!"
        cd "!FOUND_SUB!"
    ) else (
        call :log "[INFO] Cloning fresh repository..."
        git clone --recurse-submodules https://github.com/enoky/StereoCrafter.git >> "%LOGFILE%" 2>&1
        cd StereoCrafter
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