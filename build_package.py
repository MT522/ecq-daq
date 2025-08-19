#!/usr/bin/env python3
"""
Build script for ECG DAQ package.

This script helps build and package the ECG DAQ system for distribution.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def clean_build():
    """Clean build directories."""
    print("\n🧹 Cleaning build directories...")
    
    dirs_to_clean = ['build', 'dist', 'ecg_daq.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")

def install_build_deps():
    """Install build dependencies."""
    deps = ['build', 'wheel', 'twine']
    return run_command([sys.executable, '-m', 'pip', 'install', '--upgrade'] + deps,
                      "Installing build dependencies")

def run_tests():
    """Run test suite."""
    if os.path.exists('tests'):
        return run_command([sys.executable, '-m', 'pytest', 'tests/'],
                          "Running test suite")
    else:
        print("⚠️  No tests directory found, skipping tests")
        return True

def build_package():
    """Build the package."""
    return run_command([sys.executable, '-m', 'build'],
                      "Building package")

def check_package():
    """Check package with twine."""
    return run_command([sys.executable, '-m', 'twine', 'check', 'dist/*'],
                      "Checking package")

def install_local():
    """Install package locally in development mode."""
    return run_command([sys.executable, '-m', 'pip', 'install', '-e', '.'],
                      "Installing package locally")

def main():
    """Main build process."""
    print("🚀 ECG DAQ Package Build Script")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    steps = [
        ("clean", clean_build),
        ("deps", install_build_deps),
        ("test", run_tests),
        ("build", build_package),
        ("check", check_package),
    ]
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        requested_steps = sys.argv[1:]
        steps = [(name, func) for name, func in steps if name in requested_steps]
    
    # Add install step if requested
    if "install" in sys.argv:
        steps.append(("install", install_local))
    
    # Execute steps
    for step_name, step_func in steps:
        success = step_func()
        if not success:
            print(f"\n❌ Build failed at step: {step_name}")
            sys.exit(1)
    
    print("\n✅ Build completed successfully!")
    
    # Show built files
    if os.path.exists('dist'):
        print("\n📦 Built packages:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")

if __name__ == "__main__":
    main()
