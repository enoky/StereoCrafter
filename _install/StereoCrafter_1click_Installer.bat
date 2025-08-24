@echo off
REM StereoCrafter Installer Script
REM Version: 1.3
REM Logs output to install_log.txt for debugging

REM Initialize log file
echo [%date% %time%] Starting installation > install_log.txt

REM Check if git is installed
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo git is not installed or not available in PATH. >> install_log.txt
    echo git is not installed or not available in PATH.
    echo Please install git from https://git-scm.com/ and ensure it is in your PATH.
    pause
    exit /b 1
)

REM Check if Python is installed and verify version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH. >> install_log.txt
    echo Python is not installed or not added to PATH.
    echo Please install Python 3.12 from https://www.python.org/.
    pause
    exit /b 1
)

REM Check Python version (require exactly 3.12)
for /f "tokens=2 delims= " %%v in ('python --version') do set py_version=%%v
echo Python version: %py_version% >> install_log.txt
for /f "tokens=1,2 delims=." %%a in ("%py_version%") do (
    set major=%%a
    set minor=%%b
)
if %major% neq 3 (
    echo Python version %py_version% is not supported. Requires Python 3.12. >> install_log.txt
    echo Python version %py_version% is not supported. Requires Python 3.12.
    pause
    exit /b 1
)
if %minor% neq 12 (
    echo Python version %py_version% is not supported. Requires Python 3.12. >> install_log.txt
    echo Python version %py_version% is not supported. Requires Python 3.12.
    pause
    exit /b 1
)

REM Check for CUDA Toolkit and version 12.8 or 12.9
echo Checking for CUDA 12.8 or 12.9 Toolkit... >> install_log.txt
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo NVIDIA CUDA Toolkit [nvcc] not found in PATH. >> install_log.txt
    echo NVIDIA CUDA Toolkit [nvcc] not found in PATH.
    echo Please install CUDA Toolkit 12.8 or 12.9 from https://developer.nvidia.com/cuda-toolkit and ensure it is in your PATH.
    pause
    exit /b 1
)

REM Log raw nvcc --version output for debugging
echo Raw nvcc --version output: >> install_log.txt
nvcc --version >> install_log.txt

REM Parse the output of nvcc --version to get the version number
set "CUDA_VERSION="
for /f "tokens=5 delims= " %%v in ('nvcc --version ^| findstr "release"') do (
    for /f "tokens=1 delims=," %%c in ("%%v") do set CUDA_VERSION=%%c
)

REM Check if CUDA_VERSION was successfully set
if not defined CUDA_VERSION (
    echo Failed to determine CUDA version. Check install_log.txt for nvcc output. >> install_log.txt
    echo Failed to determine CUDA version.
    echo Please ensure CUDA Toolkit 12.8 or 12.9 is correctly installed and nvcc is functioning.
    pause
    exit /b 1
)

echo Found CUDA version: %CUDA_VERSION% >> install_log.txt
echo Found CUDA version: %CUDA_VERSION%

if not "%CUDA_VERSION%"=="12.8" if not "%CUDA_VERSION%"=="12.9" (
    echo Incorrect CUDA version detected. This script requires version 12.8 or 12.9, but found %CUDA_VERSION%. >> install_log.txt
    echo Incorrect CUDA version detected. This script requires version 12.8 or 12.9, but found %CUDA_VERSION%.
    pause
    exit /b 1
)

echo CUDA %CUDA_VERSION% Toolkit found. >> install_log.txt

REM If the StereoCrafter directory exists, prompt user
if exist "StereoCrafter" (
    echo The StereoCrafter directory already exists. >> install_log.txt
    echo The StereoCrafter directory already exists.
    set /p user_choice="Do you want to remove it and continue? (Y/N): "
    if /i "%user_choice%"=="Y" (
        rmdir /s /q StereoCrafter
        if %errorlevel% neq 0 (
            echo Failed to remove existing StereoCrafter directory. >> install_log.txt
            echo Failed to remove existing StereoCrafter directory.
            pause
            exit /b %errorlevel%
        )
    ) else (
        echo Aborting installation. >> install_log.txt
        echo Aborting installation.
        pause
        exit /b 0
    )
)

REM Set environment variable before cloning
set GIT_CLONE_PROTECTION_ACTIVE=false

REM Clone the StereoCrafter repository with submodules
echo Cloning repository with submodules... >> install_log.txt
git clone --recursive https://github.com/enoky/StereoCrafter.git
if %errorlevel% neq 0 (
    echo Failed to clone the StereoCrafter repository. >> install_log.txt
    echo Failed to clone the StereoCrafter repository.
    pause
    exit /b %errorlevel%
)

REM Verify directory exists before changing
if not exist "StereoCrafter" (
    echo Cloned directory StereoCrafter not found. >> install_log.txt
    echo Cloned directory StereoCrafter not found.
    pause
    exit /b 1
)

cd StereoCrafter
if %errorlevel% neq 0 (
    echo Failed to change directory into StereoCrafter. >> install_log.txt
    echo Failed to change directory into StereoCrafter.
    pause
    exit /b %errorlevel%
)

REM Check for requirements.txt
if not exist "requirements.txt" (
    echo requirements.txt not found in StereoCrafter directory. >> install_log.txt
    echo requirements.txt not found in StereoCrafter directory.
    pause
    exit /b 1
)

REM Create a virtual environment
echo Creating virtual environment... >> install_log.txt
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. >> install_log.txt
    echo Failed to create virtual environment.
    pause
    exit /b %errorlevel%
)

REM Activate the virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment activation script not found. >> install_log.txt
    echo Virtual environment activation script not found.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. >> install_log.txt
    echo Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

REM Upgrade pip
echo Upgrading pip... >> install_log.txt
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip. >> install_log.txt
    echo Failed to upgrade pip.
    pause
    exit /b %errorlevel%
)

REM Install dependencies from requirements.txt
echo Installing dependencies from requirements.txt... >> install_log.txt
python -m pip install --upgrade -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies from requirements.txt. >> install_log.txt
    echo Failed to install dependencies from requirements.txt.
    pause
    exit /b %errorlevel%
)

REM Final verification that PyTorch can see the GPU
echo Verifying PyTorch can access CUDA... >> install_log.txt
python -c "import torch; exit(0 if torch.cuda.is_available() and torch.version.cuda in ['12.8', '12.9'] else 1)"
if %errorlevel% neq 0 (
    echo Verification failed: PyTorch cannot access CUDA 12.8 or 12.9. >> install_log.txt
    echo Verification failed: PyTorch cannot access CUDA 12.8 or 12.9.
    echo Please check your NVIDIA driver, PyTorch, and CUDA Toolkit installation compatibility.
    pause
    exit /b 1
)

echo PyTorch CUDA verification successful. >> install_log.txt
echo All dependencies installed successfully. >> install_log.txt
echo All dependencies installed successfully.
echo Installation log saved to install_log.txt
pause
