#!/usr/bin/env bash
# Double-click launcher for macOS Finder.
# Place this file in the project root and run: chmod +x launch_gui.command

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [[ -n "${TFM_PYTHON:-}" ]]; then
  PYTHON_EXE="$TFM_PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
  PYTHON_EXE="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v python)"
else
  echo "ERROR: Could not find Python. Set TFM_PYTHON to your environment's python executable."
  read -r -p "Press Enter to close..."
  exit 1
fi

echo "Project root: $SCRIPT_DIR"
echo "Python: $PYTHON_EXE"
"$PYTHON_EXE" -c "import sys; print('Python executable:', sys.executable)"
"$PYTHON_EXE" code/tfm_GUI/GUI.py
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
  echo ""
  echo "GUI exited with error code $EXIT_CODE."
  read -r -p "Press Enter to close..."
fi

exit $EXIT_CODE
