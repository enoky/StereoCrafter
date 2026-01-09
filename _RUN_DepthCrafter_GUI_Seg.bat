@echo off
:: Set the CUDA_PATH to your 12.8 installation directory
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

:: Prepend CUDA 12.8 bin and libnvvp folders to PATH so they take precedence
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

call venv\scripts\activate.bat
python depthcrafter_gui_seg.py