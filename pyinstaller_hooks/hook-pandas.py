"""
PyInstaller hook for pandas to include version metadata
"""
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata('pandas')
datas += copy_metadata('numpy')
datas += copy_metadata('pytz')
datas += copy_metadata('python-dateutil')
