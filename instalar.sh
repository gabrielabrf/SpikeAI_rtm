#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
# As dependências são instaladas acima para manter uma única versão do OpenCV.
.venv/bin/python -m pip install rtmlib==0.0.16 --no-deps
echo 'Pronto. Ative com: source .venv/bin/activate'
