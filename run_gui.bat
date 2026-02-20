@echo off
setlocal

:: ARM64 Python is required for NPU/QNN. Set PYTHON_ARM64 to override.
if not defined PYTHON_ARM64 (
    set "PYTHON_ARM64=%LOCALAPPDATA%\Python\Python311-arm64\python.exe"
)

if not exist "%PYTHON_ARM64%" (
    echo ERROR: ARM64 Python not found at %PYTHON_ARM64%
    echo Set PYTHON_ARM64 environment variable to the correct path.
    exit /b 1
)

"%PYTHON_ARM64%" "%~dp0hidock_gui.py" %*
