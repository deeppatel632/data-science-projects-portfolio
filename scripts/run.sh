#!/usr/bin/env bash
# run.sh - Local Launch Script for Data Science Projects Portfolio

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================================"
echo "      Data Science Portfolio · Interactive Runner"
echo "============================================================"

# Check for virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using currently active virtual environment: $VIRTUAL_ENV"
else
    echo "No .venv found. Attempting to create one with Python 3.11..."
    if command -v python3.11 &>/dev/null; then
        python3.11 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    elif command -v python3 &>/dev/null; then
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "Error: Python 3 is required but was not found on PATH."
        exit 1
    fi
fi

# Route based on command-line argument
TARGET="${1:-portfolio}"

case "$TARGET" in
    portfolio|all|main)
        echo "Starting Central Portfolio Application (app.py)..."
        echo "Open your browser at the local Streamlit URL displayed below."
        streamlit run app.py
        ;;
    customer|customer_segmentation|p1)
        echo "Starting Project 1: Customer Behavioral Segmentation..."
        streamlit run projects/customer_segmentation/app.py
        ;;
    student|student_performance|p2)
        echo "Starting Project 2: Student Academic Performance..."
        streamlit run projects/student_performance/app.py
        ;;
    restaurant|restaurant_analytics|p3)
        echo "Starting Project 3: Restaurant Sales & Performance..."
        streamlit run projects/restaurant_analytics/app.py
        ;;
    test|tests)
        echo "Running automated test suite..."
        pytest -v
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: ./scripts/run.sh [portfolio | customer | student | restaurant | tests]"
        exit 1
        ;;
esac
