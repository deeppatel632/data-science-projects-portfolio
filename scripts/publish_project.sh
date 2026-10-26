#!/usr/bin/env bash
# publish_project.sh - Independent Project Export and Handover Utility
#
# Usage:
#   ./scripts/publish_project.sh <project-name> [--output-dir <path>] [--init-git] [--create-remote <remote-url>]
#
# Examples:
#   ./scripts/publish_project.sh customer-segmentation
#   ./scripts/publish_project.sh student-performance --output-dir /tmp/student-repo
#   ./scripts/publish_project.sh restaurant-analytics --init-git

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PROJECT_INPUT="$1"
OUTPUT_DIR=""
INIT_GIT=false
REMOTE_URL=""

shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --init-git)
            INIT_GIT=true
            shift
            ;;
        --create-remote)
            REMOTE_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/publish_project.sh <customer-segmentation | student-performance | restaurant-analytics> [--output-dir <dir>] [--init-git] [--create-remote <url>]"
            exit 1
            ;;
    esac
done

if [ -z "$PROJECT_INPUT" ]; then
    echo "Error: Missing project name argument."
    echo "Supported projects: customer-segmentation, student-performance, restaurant-analytics"
    exit 1
fi

# Standardize project name
case "$PROJECT_INPUT" in
    customer-segmentation|customer_segmentation|customer)
        PROJECT_DIR="projects/customer_segmentation"
        EXPORT_NAME="customer-segmentation"
        ;;
    student-performance|student_performance|student)
        PROJECT_DIR="projects/student_performance"
        EXPORT_NAME="student-performance"
        ;;
    restaurant-analytics|restaurant_analytics|restaurant)
        PROJECT_DIR="projects/restaurant_analytics"
        EXPORT_NAME="restaurant-analytics"
        ;;
    *)
        echo "Error: Unknown project '$PROJECT_INPUT'."
        echo "Available options: customer-segmentation, student-performance, restaurant-analytics"
        exit 1
        ;;
esac

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$REPO_ROOT/dist/$EXPORT_NAME"
fi

echo "============================================================"
echo "Exporting Standalone Project: $EXPORT_NAME"
echo "Source: $PROJECT_DIR"
echo "Destination: $OUTPUT_DIR"
echo "============================================================"

# Ensure destination is clean
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning existing export directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# 1. Copy project source files
echo "1. Copying project files..."
cp -R "$PROJECT_DIR"/* "$OUTPUT_DIR"/

# 2. Copy shared infrastructure modules
echo "2. Bundling shared module dependencies..."
mkdir -p "$OUTPUT_DIR/shared"
cp -R "$REPO_ROOT/shared"/* "$OUTPUT_DIR/shared/"

# 3. Copy configuration and runtime files
echo "3. Copying configuration files..."
cp "$REPO_ROOT/requirements.txt" "$OUTPUT_DIR/"
cp "$REPO_ROOT/pyproject.toml" "$OUTPUT_DIR/"
cp "$REPO_ROOT/.gitignore" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$REPO_ROOT/LICENSE" "$OUTPUT_DIR/" 2>/dev/null || true

# Copy Streamlit theme config
mkdir -p "$OUTPUT_DIR/.streamlit"
cp "$REPO_ROOT/.streamlit/config.toml" "$OUTPUT_DIR/.streamlit/"

# 4. Create standalone runner app.py at root of exported project
echo "4. Generating standalone root app.py..."
cat <<EOF > "$OUTPUT_DIR/standalone_app.py"
"""Standalone launcher for $EXPORT_NAME."""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import render_${EXPORT_NAME//-/_}_page

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_${EXPORT_NAME//-/_}_page()
EOF

# 5. Clean up any unintended artifacts, __pycache__, .DS_Store
find "$OUTPUT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$OUTPUT_DIR" -name ".DS_Store" -delete 2>/dev/null || true

# 6. Verify standalone test execution
echo "5. Verifying standalone test execution..."
if [ -d "$REPO_ROOT/.venv" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    PYTEST_BIN="$REPO_ROOT/.venv/bin/pytest"
    if [ -x "$PYTEST_BIN" ]; then
        cd "$OUTPUT_DIR"
        PYTHONPATH=. "$PYTEST_BIN" tests/ -v
        cd "$REPO_ROOT"
        echo "✓ Standalone test verification passed!"
    fi
fi

# 7. Optional Git initialization
if [ "$INIT_GIT" = true ]; then
    echo "6. Initializing Git repository in export directory..."
    cd "$OUTPUT_DIR"
    git init -b main
    git add .
    git commit -m "feat: initial standalone release for $EXPORT_NAME"
    if [ -n "$REMOTE_URL" ]; then
        echo "Adding remote origin: $REMOTE_URL"
        git remote add origin "$REMOTE_URL"
        echo "Push with: cd '$OUTPUT_DIR' && git push -u origin main"
    fi
    cd "$REPO_ROOT"
fi

echo ""
echo "============================================================"
echo "✓ Standalone export successfully prepared at:"
echo "  $OUTPUT_DIR"
echo ""
echo "To test run the standalone exported project:"
echo "  cd '$OUTPUT_DIR'"
echo "  streamlit run app.py"
echo "============================================================"
