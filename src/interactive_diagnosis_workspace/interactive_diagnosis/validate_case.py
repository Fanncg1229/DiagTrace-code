from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import validate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an interactive diagnosis case JSON file.")
    parser.add_argument("case_file")
    args = parser.parse_args()

    path = Path(args.case_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    result = validate_case(data)

    print(f"case_file: {path}")
    print(f"valid: {result.ok}")
    for error in result.errors:
        print(f"ERROR: {error}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")

    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()

