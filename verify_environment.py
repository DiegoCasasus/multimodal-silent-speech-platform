from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "psychopy": "psychopy",
    "pylsl": "pylsl",
    "pyxdf": "pyxdf",
    "PyQt6": "PyQt6",
    "wx": "wxPython",
    "yaml": "PyYAML",
    "psutil": "psutil",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
}

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print()

failed = []
for module_name, package_name in REQUIRED.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")
        print(f"[OK] {package_name}: {version}")
    except Exception as exc:
        failed.append((package_name, exc))
        print(f"[FAIL] {package_name}: {exc}")

print()
if failed:
    print("Environment verification FAILED.")
    for package_name, exc in failed:
        print(f" - {package_name}: {exc}")
    raise SystemExit(1)

print("Environment verification PASSED.")
