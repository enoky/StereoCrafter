@echo off
setlocal enabledelayedexpansion

REM Define output folder name
set temp_folder=resized_videos

REM Create a temporary folder to store scaled videos
if not exist %temp_folder% mkdir %temp_folder%

REM Loop through all MP4 files in the current directory
echo Scaling videos...
set counter=0
for %%f in (*.mp4) do (
    set /a counter+=1
    ffmpeg -i "%%f" -vf "scale=2688:1152:flags=spline" -c:v h264_nvenc -cq 16 -preset medium "%temp_folder%\scaled_%%~nf.mp4"
)

pause