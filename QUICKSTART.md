# Quick Start Guide 🚀

Welcome to Pioreactor Analysis Panel! This guide will get you up and running in minutes.

## 🎯 Choose Your Installation Method

### 1️⃣ **Easiest: Double-Click Installation** (Windows)
Perfect for users who want to get started immediately without technical setup.

**Steps:**
1. Double-click `install.bat`
2. Wait for installation to complete (2-3 minutes)
3. Double-click `launch.bat`
4. Your browser opens automatically to the application! 🎉

**That's it!** The app is ready to use.

---

### 2️⃣ **Simplest: Standalone Executable** (Coming Soon)
No Python installation required!

**Steps:**
1. Download `PioreactorAnalysisPanel.exe` from Releases
2. Double-click to run
3. Browser opens automatically

To build the EXE yourself:
- Run `install.bat` (first time only)
- Run `build_exe.bat`
- Find the EXE in `dist\` folder

---

### 3️⃣ **Developer: Manual Setup** (All Platforms)

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` in your browser.

---

## 📚 What Can You Do?

This application provides two powerful analysis tools:

### **Batch Growth Rate Analysis**
- Automatically detect maximum specific growth rate (μmax)
- Calculate doubling times
- Determine apparent yield (Yx/s)
- Export data to CSV
- Export publication-ready plots
- Quality validation checks

### **Dilution Rate Analysis**
- Visualize dilution rates over time
- Track optical density (OD) vs targets
- Analyze time between dosing events
- Statistical breakdown by OD regions
- Export statistics and plots
- Synchronized zoom across all graphs

---

## 🎓 Usage Tips

1. **Upload Data:** Click the file upload button and select your CSV file
2. **Adjust Parameters:** Use the sidebar to tune analysis parameters (hover over parameters for help!)
3. **Auto-Detect:** Let the algorithm find the exponential phase for you
4. **Export Results:** Save your analysis as CSV or export plots as PNG images
5. **Compare Runs:** Add multiple analyses to the cumulative results table

---

## 🔧 Troubleshooting

### Can't run .bat files?
Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port already in use?
Change the port in `app.py`:
```python
port = int(os.environ.get("PORT", 8080))  # Change 7860 to 8080
```

### Python not found?
Download from [python.org](https://www.python.org/) and check "Add Python to PATH" during installation.

---

## 📖 More Information

- **Full Installation Guide:** See `INSTALL.md`
- **Usage Instructions:** See `readme.md`
- **Deployment:** See `deployment_guide.md`
- **Issues/Support:** [GitHub Issues](https://github.com/DocRuzzy/pioreactor_analysis_panel/issues)

---

## 🎉 You're Ready!

The application should now be running at: **http://localhost:7860**

**Need Help?** Check the full documentation in `INSTALL.md` or open an issue on GitHub.

---

## 📦 What's Included

```
pioreactor_analysis_panel/
├── install.bat              # 🚀 One-click installer (Windows)
├── launch.bat               # ▶️  Quick launcher (Windows)
├── build_exe.bat            # 📦 Build standalone EXE
├── app.py                   # 🎯 Main application
├── batch_growth_rate_analysis.py   # Batch analyzer
├── pioreactor_dilution_rate_panel.py   # Dilution analyzer
├── requirements.txt         # 📋 Dependencies
├── setup.py                 # 🔧 Package installer
├── pyinstaller.spec         # 📦 EXE build config
├── INSTALL.md               # 📖 Detailed install guide
├── readme.md                # 📚 Usage documentation
└── QUICKSTART.md            # ⚡ This file!
```

Happy analyzing! 🧫🔬
