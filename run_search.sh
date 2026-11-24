#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOTIF_PROJECT_PATH="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/scenario:$REPO_ROOT/scenario/search_lib:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python "$REPO_ROOT/scenario/sotif_search_main.py" \
  --config "$REPO_ROOT/scenario/config/search_config.yaml"
