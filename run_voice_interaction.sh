#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
python "$ROOT_DIR/src/dialog_flask_service.py"
