# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for creating Pioreactor Analysis Panel executable
Build with: pyinstaller pyinstaller.spec
"""

block_cipher = None

# Collect all necessary data files and hidden imports
added_files = []
hidden_imports = [
    'panel',
    'holoviews',
    'hvplot',
    'hvplot.pandas',
    'bokeh',
    'bokeh.models',
    'bokeh.plotting',
    'bokeh.models.formatters',
    'pandas',
    'numpy',
    'scipy',
    'scipy.stats',
    'param',
    'matplotlib',
    'plotly',
    'PIL',
    'tornado',
    'jinja2',
    'yaml',
    'markdown',
    'requests',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PioreactorAnalysisPanel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add .ico file path here if you have an icon
)
