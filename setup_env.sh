#!/bin/bash
# Installs Miniforge (if not already present) into a dedicated, self-contained
# location, then creates the `lfreeda` conda environment from environment.yml
# and points it at its own bundled CUDA/cuDNN libraries so TensorFlow can find
# them (conda's cudatoolkit/cudnn packages don't add themselves to
# LD_LIBRARY_PATH on their own).
#
# To remove everything this script installs, run uninstall_env.sh.
set -e

MINIFORGE_PREFIX="$HOME/miniforge3"

if [ ! -d "$MINIFORGE_PREFIX" ]; then
  echo "Installing Miniforge to $MINIFORGE_PREFIX..."
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$MINIFORGE_PREFIX"
  rm -f /tmp/miniforge.sh
else
  echo "Found existing Miniforge install at $MINIFORGE_PREFIX, reusing it."
fi

source "$MINIFORGE_PREFIX/etc/profile.d/conda.sh"
conda init bash >/dev/null

conda env create -f environment.yml

conda activate lfreeda
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
conda deactivate
conda activate lfreeda

echo "Verifying GPU is detected by TensorFlow..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU devices:', gpus); assert gpus, 'No GPU detected by TensorFlow'"

echo
echo "Environment 'lfreeda' is ready."
echo "Open a new terminal (or run 'source ~/.bashrc') and activate it with:"
echo "    conda activate lfreeda"
