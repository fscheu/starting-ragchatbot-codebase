#!/usr/bin/env bash
# Format code script for Linux/Mac

set -e

echo "Formatting code..."
echo

uv run python scripts/format.py

echo
echo "Code formatted successfully!"
