#!/usr/bin/env bash
set -euo pipefail

runtime_root=/root/panxy/particalMOE/.runtime
conda_root="$runtime_root/miniconda3"
env_root="$runtime_root/pt_env"
status_file="$runtime_root/SETUP_STATUS.txt"
mkdir -p "$runtime_root"
printf 'INSTALLING_MINICONDA\n' > "$status_file"

if [[ ! -x "$conda_root/bin/conda" ]]; then
  installer="$runtime_root/miniconda-installer.sh"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
  bash "$installer" -b -p "$conda_root"
fi

printf 'CREATING_PYTHON_ENV\n' > "$status_file"
if [[ ! -x "$env_root/bin/python" ]]; then
  "$conda_root/bin/conda" create -y -p "$env_root" python=3.11.15 pip
fi

printf 'INSTALLING_PACKAGES\n' > "$status_file"
"$env_root/bin/python" -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.11.0+cu126'
"$env_root/bin/python" -m pip install --no-cache-dir \
  'numpy==2.4.4' 'pandas==3.0.2' 'scipy==1.17.1'

"$env_root/bin/python" - <<'PY'
import json
import numpy
import pandas
import scipy
import torch

assert torch.__version__ == "2.11.0+cu126"
assert numpy.__version__ == "2.4.4"
assert torch.cuda.is_available()
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scipy": scipy.__version__,
    "gpu": torch.cuda.get_device_name(0),
}))
PY

printf 'PASS\n' > "$status_file"
