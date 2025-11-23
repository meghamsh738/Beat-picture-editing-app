#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="$REPO_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
REQ_FILE="$REPO_DIR/requirements.txt"
REQ_STAMP="$VENV_DIR/.requirements.sha"
REQ_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating virtual environment in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ ! -f "$REQ_STAMP" ]] || [[ "$(cat "$REQ_STAMP")" != "$REQ_HASH" ]]; then
    echo "Installing Python requirements"
    "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
    printf '%s' "$REQ_HASH" > "$REQ_STAMP"
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec "$VENV_PY" timeline_builder.py "$@"
