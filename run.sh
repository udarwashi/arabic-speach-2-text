#!/usr/bin/env bash
# Start the app. First run downloads the selected Whisper model.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv found. Create one first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# CTranslate2 finds cuDNN/cuBLAS through the pip-installed nvidia-* packages.
SITE_PACKAGES="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for lib in cudnn cublas; do
  candidate="${SITE_PACKAGES}/nvidia/${lib}/lib"
  [[ -d "${candidate}" ]] && export LD_LIBRARY_PATH="${candidate}:${LD_LIBRARY_PATH:-}"
done

HOST="${S2T_HOST:-127.0.0.1}"
PORT="${S2T_PORT:-8749}"

echo "→ http://${HOST}:${PORT}"
exec .venv/bin/python -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" "$@"
