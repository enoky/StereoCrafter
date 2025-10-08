@echo off
REM --- StereoCrafter Repository Update Script ---
REM This script is designed to be run from the root directory of a local Git repository.

echo.
echo Navigating to the script's directory...
REM Change the current directory to the directory where this batch file is located.
cd /d "%~dp0"

echo.
echo Attempting to pull the latest changes from the remote GitHub repository (StereoCrafter)...
REM Execute the git pull command to fetch and merge changes.
git pull

REM Check the exit code of the last command (git pull)
if %errorlevel% equ 0 (
echo.
echo Successfully updated the repository.
) else (
echo.
echo ERROR: Git pull failed [Error Code: %errorlevel%].
echo Please check the error message above. You may need to commit or stash local changes first.
)

echo.
pause