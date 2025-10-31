@echo off
REM Format code script for Windows

echo Formatting code...
echo.

uv run python scripts\format.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Formatting failed!
    exit /b 1
)

echo.
echo Code formatted successfully!
