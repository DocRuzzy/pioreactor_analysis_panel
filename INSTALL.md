# Installation Guide

This guide provides multiple ways to install and run the Pioreactor Analysis Panel.

## Quick Start (Recommended for Windows Users)

### Option 1: Automated Installation (Easiest)

1. **Download** or clone this repository
2. **Double-click** `install.bat` in the project folder
3. Wait for installation to complete
4. **Double-click** `launch.bat` to start the application
5. Your browser will open to `http://localhost:7860`

That's it! The installer creates a virtual environment and installs all dependencies automatically.

---

## Option 2: Standalone Executable (No Python Required)

If you don't want to install Python or deal with dependencies:

1. Download the pre-built executable from the [Releases page](https://github.com/DocRuzzy/pioreactor_analysis_panel/releases)
2. Double-click `PioreactorAnalysisPanel.exe`
3. Your browser opens automatically

**To build the executable yourself:**
```batch
install.bat          # First-time setup
build_exe.bat        # Build the EXE
```
The executable will be in `dist\PioreactorAnalysisPanel.exe`

---

## Option 3: Manual Installation (All Platforms)

### Requirements
- Python 3.8 or higher
- pip (included with Python)

### Steps

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open your browser to `http://localhost:7860`

---

## Option 4: Install as Python Package

Install the application as a Python package with a command-line launcher:

```bash
# Install in development mode
pip install -e .

# Or install from source
pip install .

# Run from anywhere
pioreactor-analysis
```

This creates a `pioreactor-analysis` command you can run from any terminal.

---

## Troubleshooting

### Windows Execution Policy Error
If you see an error about script execution being disabled:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Python not found"
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Restart your terminal/command prompt

### Port Already in Use
If port 7860 is busy, edit `app.py` and change:
```python
port = int(os.environ.get("PORT", 7860))  # Change 7860 to another port
```

### Missing Dependencies
If you get import errors:
```bash
pip install -r requirements.txt --force-reinstall
```

### Firewall Warning
Windows may ask to allow network access. Click "Allow" - the app only runs locally on your computer.

---

## Uninstallation

### Automated Install Method
1. Delete the project folder
2. The virtual environment is contained within, so no system-wide changes remain

### Python Package Method
```bash
pip uninstall pioreactor-analysis-panel
```

### Standalone EXE
Just delete the executable file.

---

## Configuration

### Changing the Port
Set the `PORT` environment variable:

**Windows:**
```batch
set PORT=8080
python app.py
```

**PowerShell:**
```powershell
$env:PORT = "8080"
python app.py
```

**Linux/macOS:**
```bash
PORT=8080 python app.py
```

### Remote Access
By default, the app only accepts connections from your computer. To allow access from other devices:

Edit `app.py` and change:
```python
address = "0.0.0.0"  # Already set for remote access
```

Then connect from other devices using: `http://YOUR_IP_ADDRESS:7860`

---

## For Developers

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
pip install black isort
black .
isort .
```

### Building Documentation
```bash
pip install sphinx
cd docs
make html
```

---

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/DocRuzzy/pioreactor_analysis_panel/issues)
- **Documentation:** See `readme.md` for usage instructions
- **Deployment Guide:** See `deployment_guide.md` for server deployment

---

## System Requirements

### Minimum
- **OS:** Windows 7+, macOS 10.13+, Linux (any modern distro)
- **RAM:** 2 GB
- **Python:** 3.8+
- **Disk Space:** 500 MB (including dependencies)

### Recommended
- **OS:** Windows 10+, macOS 11+, Ubuntu 20.04+
- **RAM:** 4 GB or more
- **Python:** 3.10+
- **Disk Space:** 1 GB
- **Browser:** Chrome, Firefox, or Edge (latest versions)

---

## License

MIT License - see LICENSE file for details
