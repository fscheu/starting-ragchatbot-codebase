#!/usr/bin/env python3
"""Run linting checks on the codebase."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run linting tools on the codebase."""
    root_dir = Path(__file__).parent.parent
    backend_dir = root_dir / "backend"

    print("[*] Running ruff linter...")
    result = subprocess.run(
        ["uv", "run", "ruff", "check", str(backend_dir), "main.py"],
        cwd=root_dir
    )

    ruff_passed = result.returncode == 0

    print("\n[*] Running black format check...")
    result = subprocess.run(
        ["uv", "run", "black", "--check", str(backend_dir), "main.py"],
        cwd=root_dir
    )

    black_passed = result.returncode == 0

    print("\n[*] Running mypy type checker...")
    result = subprocess.run(
        ["uv", "run", "mypy", str(backend_dir), "main.py"],
        cwd=root_dir
    )

    mypy_passed = result.returncode == 0

    # Summary
    print("\n" + "="*50)
    print("Linting Summary:")
    print(f"  Ruff:  {'[+] Passed' if ruff_passed else '[!] Failed'}")
    print(f"  Black: {'[+] Passed' if black_passed else '[!] Failed'}")
    print(f"  Mypy:  {'[+] Passed' if mypy_passed else '[!] Failed'}")
    print("="*50)

    if not all([ruff_passed, black_passed, mypy_passed]):
        sys.exit(1)

    print("\n[+] All checks passed!")


if __name__ == "__main__":
    main()
