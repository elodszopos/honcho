#!/usr/bin/env bash
set -euo pipefail

# Runs the deriver eval on the host against the running services, with the deriver's own settings.

REPO="${HONCHO_REPO:-$HOME/Projects/honcho}"
cd "$REPO"

set -a
. ./.env
set +a

export PYTHON_DOTENV_DISABLED=1
export PYTHONPATH="$REPO"
export DB_CONNECTION_URI="${DB_CONNECTION_URI/@database:5432/@127.0.0.1:18732}"
export CURATED_MEMORY_PATHS="{\"hermes\": [\"$HOME/.hermes/soul/delivery-contract.md\", \"$HOME/.hermes/memories/MEMORY.md\", \"$HOME/.hermes/memories/USER.md\"]}"
while IFS= read -r line; do
    name="${line%%=*}"
    value="${line#*=}"
    export "$name=${value//host.docker.internal/127.0.0.1}"
done < <(env | grep 'host.docker.internal' || true)

exec uv run python scripts/deriver_eval/deriver_eval.py "$@"
