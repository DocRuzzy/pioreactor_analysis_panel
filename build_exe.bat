@echo off
REM Build standalone executable using PyInstaller
REM Run this after installing PyInstaller: pip install pyinstaller

echo ========================================
echo Pioreactor Analysis Panel - EXE Builder
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo.
echo Building executable...
echo This may take several minutes...
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build the executable
pyinstaller pyinstaller.spec
if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo The executable can be found at:
echo   dist\PioreactorAnalysisPanel.exe
echo.
echo You can now distribute this single file.
echo No Python installation required to run it!
echo.
pause
