---
title: Pioreactor Analysis Panel
emoji: 🧫
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Pioreactor Dilution Rate Analysis Tool

An interactive web application for analyzing bioreactor dilution rates from Pioreactor data. This tool is designed for academic research and can be used to analyze the performance of continuous culture bioreactors.

## Features

- Interactive visualization of dilution rates over time
- OD (Optical Density) tracking with target vs actual comparison
- Analysis of inter-dosing periods
- Statistical breakdown by target OD regions
- Bookmark system for marking important points
- CSV data upload capability

## Deployment

This application is deployed on [Hugging Face Spaces](https://huggingface.co/spaces) and can be accessed at: [huggingface.co/spaces/DocRuzzy/pioreactor_analysis_panel](https://huggingface.co/spaces/DocRuzzy/pioreactor_analysis_panel)

## Installation & Setup

### 🚀 Quick Start (Windows)

The easiest way to get started:

1. **Double-click** `install.bat` (installs everything automatically)
2. **Double-click** `launch.bat` (starts the application)
3. Browser opens to `http://localhost:7860`

**See [QUICKSTART.md](QUICKSTART.md) for the fastest way to get started!**

### 📦 Multiple Installation Options

We provide several installation methods to suit your needs:

- **Automated Installer** (Windows) - Double-click `install.bat`
- **Standalone EXE** - No Python required (run `build_exe.bat`)
- **Python Package** - `pip install -e .` for system-wide installation
- **Manual Setup** - Traditional virtual environment approach

**See [INSTALL.md](INSTALL.md) for detailed installation instructions for all platforms.**

## Local Development

### Quick Method (Windows)
```batch
install.bat
launch.bat
```

### Manual Method (All Platforms)

To run this application locally:

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser to `http://localhost:7860`

## Usage Instructions

1. Upload your Pioreactor events CSV file using the file upload widget
2. Adjust reactor volume and moving average window settings as needed
3. Click "Update" to refresh the analysis
4. Click on points in the graphs to add bookmarks
5. View statistics organized by OD target regions

## Citation

If you use this tool in your research, please cite:

```
Russell Kirk Pirlo & Claude. (2025). Pioreactor Dilution Rate Analysis Tool. 
Hugging Face Spaces. https://huggingface.co/spaces/DocRuzzy/pioreactor_analysis_panel
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
