@echo off
setlocal enabledelayedexpansion

:: Set the CUDA_PATH to your 12.8 installation directory
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

:: Prepend CUDA 12.8 bin and libnvvp folders to PATH so they take precedence
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Count how many GPUs are present
set GPU_COUNT=0
for /f %%i in ('nvidia-smi -L ^| find /c "GPU"') do set GPU_COUNT=%%i

:: If more than 1 GPU, ask the user
if %GPU_COUNT% GTR 1 (
    echo =================================================
    echo Multiple GPUs detected:
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    echo =================================================
    set /p GPU_ID="Enter the ID of the GPU you want to use (e.g. 0 or 1): "
    
    :: Set the environment variable for this session
    set CUDA_VISIBLE_DEVICES=!GPU_ID!
    echo Script will now run using ONLY GPU !GPU_ID!.
) else (
    echo Only %GPU_COUNT% GPU detected. Running on default.
)

:: 4. Run your python script
python depthcrafter_gui_seg.py