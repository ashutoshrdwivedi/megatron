#!/usr/bin/env bash
# Run the full test suite on CPU (safe on shared GPU machines).
# conftest.py sets JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES="" automatically,
# but we also export them here so any ad-hoc python invocations in the session
# behave the same way.
#
# Usage:
#   ./test.sh              # run all tests
#   ./test.sh -k overfit   # run a subset by keyword
#   ./test.sh -v           # verbose output
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTEST="$REPO_ROOT/.venv/bin/pytest"

exec "$PYTEST" "$REPO_ROOT/tests/" "$@"
