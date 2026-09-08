@echo off
setlocal
cd /d "%~dp0"

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    call conda activate tfm_unified
)

python code\tfm_GUI\GUI.py
pause
