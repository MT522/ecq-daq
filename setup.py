#!/usr/bin/env python3
"""Setup script for ECG DAQ package."""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from package
def get_version():
    """Get version from package __init__.py."""
    init_file = Path(__file__).parent / "ecg_daq" / "__init__.py"
    if init_file.exists():
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Get long description from README.md."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    """Get requirements from requirements.txt."""
    req_file = Path(__file__).parent / "requirements.txt"
    requirements = []
    if req_file.exists():
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="ecg-daq",
    version=get_version(),
    author="Mehrshad",
    author_email="mehrshadtaji61@gmail.com",
    description="High-performance ECG Data Acquisition System",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mehrshad/ecg-daq",
    project_urls={
        "Bug Tracker": "https://github.com/MT522/ecg-daq/issues",
        "Documentation": "https://github.com/MT522/ecg-daq#readme",
        "Source Code": "https://github.com/MT522/ecg-daq",
    },
    packages=find_packages(exclude=["tests", "tests.*", "test_data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "Topic :: Communications :: Serial",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "visualization": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecg-monitor=ecg_daq.examples.real_time_monitor:main_sync",
            "ecg-plot=ecg_daq.examples.plot_ecg_data:main",
            "ecg-mock-hardware=ecg_daq.examples.test_uart_with_mock:main",
        ],
    },
    package_data={
        "ecg_daq": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "ecg", "electrocardiogram", "data-acquisition", "medical-devices",
        "serial-communication", "uart", "real-time", "signal-processing",
        "healthcare", "biomedical", "cardiology"
    ],
)
