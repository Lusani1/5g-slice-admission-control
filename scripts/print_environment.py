#!/usr/bin/env python3
"""Print the Python and package versions relevant to the experiments."""
import platform
import importlib.metadata as md

packages = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "matplotlib"]
print(f"Python: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
for package in packages:
    try:
        print(f"{package}: {md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}: not installed")
