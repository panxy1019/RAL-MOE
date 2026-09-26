#!/usr/bin/env python3
"""Print the schema of representative sealed evaluation coefficient files."""
from pathlib import Path
import sys

import numpy as np

for pattern in sys.argv[1:]:
    paths = sorted(Path().glob(pattern)) if not pattern.startswith("/") else sorted(Path("/").glob(pattern[1:]))
    if not paths:
        print(f"MISSING {pattern}")
        continue
    path = paths[0]
    with np.load(path, allow_pickle=False) as data:
        print(path)
        for key in data.files:
            value = np.asarray(data[key])
            scalar = repr(value.item()) if value.ndim == 0 else ""
            print(f"  {key}: shape={value.shape} dtype={value.dtype} {scalar}")
