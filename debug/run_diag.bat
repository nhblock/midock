@echo off
setlocal

if not defined PYTHON_ARM64 (
    set "PYTHON_ARM64=%LOCALAPPDATA%\Python\Python311-arm64\python.exe"
)

"%PYTHON_ARM64%" "%~dp0diag_diarize.py" %1 %2 %3 2>nul
