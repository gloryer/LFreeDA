#!/bin/bash
# Removes the `lfreeda` conda environment, the Miniforge installation, and the
# shell init block created by setup_env.sh. Safe to run even if setup_env.sh
# only partially completed.
set -e

MINIFORGE_PREFIX="$HOME/miniforge3"

if [ -d "$MINIFORGE_PREFIX" ]; then
  source "$MINIFORGE_PREFIX/etc/profile.d/conda.sh"
  conda deactivate 2>/dev/null || true
  conda env remove -n lfreeda -y 2>/dev/null || true

  echo "Removing conda's shell init block from ~/.bashrc..."
  conda init --reverse bash >/dev/null 2>&1 || true

  echo "Removing Miniforge installation at $MINIFORGE_PREFIX..."
  rm -rf "$MINIFORGE_PREFIX"
  echo "Done."
else
  echo "No Miniforge installation found at $MINIFORGE_PREFIX; nothing to remove."
fi
