@echo off
REM Quick quality check script for Windows

echo Running code quality checks...
echo.

uv run python scripts\lint.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Quality checks failed! Run 'scripts\format.cmd' to fix formatting issues.
    exit /b 1
)

echo.
echo All quality checks passed!
