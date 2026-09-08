#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate tfm_unified || true
fi

python code/tfm_GUI/GUI.py
