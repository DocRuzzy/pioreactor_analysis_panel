@echo off
REM Quick launcher for Pioreactor Analysis Panel
REM Double-click this file to start the application

echo ========================================
echo Pioreactor Analysis Panel Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the application.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the application
echo Starting application...
echo.
echo The application will open in your default browser at:
echo http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py

pause
