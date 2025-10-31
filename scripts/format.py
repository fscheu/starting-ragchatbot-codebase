#!/usr/bin/env python3
"""Format code using black and ruff."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run formatting tools on the codebase."""
    root_dir = Path(__file__).parent.parent
    backend_dir = root_dir / "backend"

    print("[*] Running black formatter...")
    result = subprocess.run(
        ["uv", "run", "black", str(backend_dir), "main.py"],
        cwd=root_dir,
        capture_output=False
    )

    if result.returncode != 0:
        print("[!] Black formatting failed!")
        sys.exit(1)

    print("[+] Black formatting completed!")

    print("\n[*] Running ruff import sorting and fixes...")
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "--fix", "--select", "I", str(backend_dir), "main.py"],
        cwd=root_dir,
        capture_output=False
    )

    if result.returncode != 0:
        print("[!] Ruff found issues but attempted fixes")
    else:
        print("[+] Ruff fixes completed!")

    print("\n[+] Code formatting complete!")


if __name__ == "__main__":
    main()
