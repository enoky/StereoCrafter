@echo off
setlocal enabledelayedexpansion
REM StereoCrafter Universal Smart Installer
REM Handles: Git Clone, GitHub ZIPs (branch-names), and Nested Folders

echo [1/7] Checking for Git...
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo Git not found. Attempting to install via Winget...
    winget install --id Git.Git -e --source winget
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install Git. Please install manually: https://git-scm.com/
        pause && exit /b 1
    )
    set "PATH=%PATH%;C:\Program Files\Git\cmd"
)

echo [2/7] Checking for UV...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

echo [3/7] Analyzing Path...
:: Get current folder name
for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"

:: A. If in '_install', move up once
if /i "!CUR_NAME!"=="_install" (
    echo [INFO] Moving out of _install...
    cd ..
    for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"
)

:: B. Check if current folder is a 'StereoCrafter' variant (ZIP or Clone)
:: 'StereoCrafter' is 13 characters. We check the first 13.
set "PREFIX=!CUR_NAME:~0,13!"
if /i "!PREFIX!"=="StereoCrafter" (
    if exist "pyproject.toml" (
        echo [INFO] Detected valid StereoCrafter project folder: !CUR_NAME!
        set "ALREADY_HOME=true"
    ) else (
        echo [INFO] Inside an empty StereoCrafter folder. Moving up to allow clean clone...
        cd ..
        set "ALREADY_HOME=false"
    )
) else (
    set "ALREADY_HOME=false"
)

echo [4/7] Repository Check...
if "!ALREADY_HOME!"=="false" (
    :: Check if a subfolder exists that starts with StereoCrafter and has a toml
    set "FOUND_SUB="
    for /d %%D in (StereoCrafter*) do (
        if exist "%%D\pyproject.toml" (
            set "FOUND_SUB=%%D"
        )
    )

    if defined FOUND_SUB (
        echo [INFO] Found project in subfolder: !FOUND_SUB!
        cd "!FOUND_SUB!"
    ) else (
        echo [INFO] No project found. Cloning fresh...
        git clone --recurse-submodules https://github.com/enoky/StereoCrafter.git
        cd StereoCrafter
    )
)

echo [5/7] Setting up Environment...

:: --- NEW CLEANUP LOGIC ---
if exist "venv" (
    echo [INFO] Legacy 'venv' folder detected.
    :: We use ^) to tell Batch this is text, not the end of the IF block
    echo This project now uses UV and '.venv' ^(with a dot^).
    set /p del_venv="Would you like to delete the old 'venv' to save space? (Y/N): "
    if /i "!del_venv!"=="Y" (
        echo [INFO] Removing legacy venv...
        rmdir /s /q venv
    )
)
:: -------------------------

:: Ensure we use the pinned Python version
uv python pin 3.12

:: This one command handles everything: creates .venv, installs Python, 
:: and installs all CUDA-enabled libraries from your pyproject.toml
echo [INFO] Running uv sync...
uv sync

REM -----------------------------------------------------------------
REM [6/7] Optional Model Weights Download
REM -----------------------------------------------------------------
echo.
echo =========================================================
echo MODEL WEIGHTS DOWNLOAD
echo =========================================================
echo These models are very large (~22GB total). 
set /p get_weights="Would you like to download them now? (Y/N): "

if /i "!get_weights!"=="Y" (
    echo [INFO] Creating weights folder...
    if not exist "weights" mkdir weights

    echo [INFO] Downloading SVD img2vid XT 1.1...
    uv run huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir weights/stable-video-diffusion-img2vid-xt-1-1 --local-dir-use-symlinks False

    echo [INFO] Downloading DepthCrafter...
    uv run huggingface-cli download tencent/DepthCrafter --local-dir weights/DepthCrafter --local-dir-use-symlinks False

    echo [INFO] Downloading StereoCrafter...
    uv run huggingface-cli download TencentARC/StereoCrafter --local-dir weights/StereoCrafter --local-dir-use-symlinks False
    
    echo [SUCCESS] All weights downloaded to the /weights folder.
) else (
    echo [SKIP] Skipping model downloads. Ensure you place them in /weights manually.
)

echo [7/7] Finalizing...
echo.
echo =========================================================
echo INSTALLATION SUCCESSFUL
echo =========================================================
echo.
echo Location: %CD%
echo =========================================================

pause