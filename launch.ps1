# PowerShell launcher for Pioreactor Analysis Panel
# Run this with: .\launch.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pioreactor Analysis Panel Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run install.bat first to set up the application." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"

# Start the application
Write-Host ""
Write-Host "Starting application..." -ForegroundColor Green
Write-Host ""
Write-Host "The application will open in your default browser at:" -ForegroundColor Yellow
Write-Host "http://localhost:7860" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run the app
python app.py

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
