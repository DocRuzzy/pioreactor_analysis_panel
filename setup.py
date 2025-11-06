"""
Setup script for Pioreactor Analysis Panel
Install with: pip install -e .
"""
from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pioreactor-analysis-panel",
    version="2.0.0",
    author="Russell Kirk Pirlo",
    description="Interactive analysis tools for Pioreactor bioreactor data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DocRuzzy/pioreactor_analysis_panel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pioreactor-analysis=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
