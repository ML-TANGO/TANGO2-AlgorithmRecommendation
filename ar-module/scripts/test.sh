#!/usr/bin/env bash
# Unit tests. Pure-Python tests always run; torch/fastapi tests skip when the packages are missing.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python}" -m unittest discover -s tests -t . -v "$@"
