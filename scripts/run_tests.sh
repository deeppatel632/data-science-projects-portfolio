#!/usr/bin/env bash
# run_tests.sh - Comprehensive Test and Lint Execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "============================================================"
echo "      Data Science Portfolio · Test & Quality Suite"
echo "============================================================"

echo ""
echo "1. Running Ruff Code Linter..."
if command -v ruff &>/dev/null; then
    ruff check .
    echo "✓ Ruff linting passed!"
else
    echo "Ruff not found in environment, skipping linting."
fi

echo ""
echo "2. Running Pytest Suite across all 3 projects..."
pytest -v

echo ""
echo "============================================================"
echo "✓ All tests and checks completed successfully!"
echo "============================================================"
