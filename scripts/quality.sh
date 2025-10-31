#!/usr/bin/env bash
# Quick quality check script for Linux/Mac

set -e

echo "Running code quality checks..."
echo

uv run python scripts/lint.py

echo
echo "All quality checks passed!"
