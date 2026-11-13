from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import validate_case


def case_files(input_dir: Path, case_ids: list[str] | None, limit: int | None) -> list[Path]:
    selected = set(case_ids or [])
    paths = sorted(
        path
        for path in input_dir.glob("*.json")
        if not path.name.startswith("_") and not path.name.endswith(".failed.json")
    )
    if selected:
        paths = [path for path in paths if path.stem in selected]
    if limit is not None:
        paths = paths[:limit]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extracted interactive diagnosis case JSON files.")
    parser.add_argument("--input_dir", default="data/pmc_interactive_extraction/internalmed")
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show_clean", action="store_true")
    args = parser.parse_args()

    files = case_files(Path(args.input_dir), args.case_ids, args.limit)
    total_errors = 0
    total_warnings = 0
    dirty_cases = 0

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = validate_case(data)
        warnings = result.warnings
        errors = result.errors
        if not warnings and not errors and not args.show_clean:
            continue
        dirty_cases += int(bool(warnings or errors))
        total_errors += len(errors)
        total_warnings += len(warnings)
        print(f"\n{path.stem}: valid={result.ok} warnings={len(warnings)} errors={len(errors)}")
        for error in errors:
            print(f"  ERROR: {error}")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    print(
        f"\nAudited {len(files)} case(s): "
        f"{dirty_cases} with warnings/errors, {total_warnings} warning(s), {total_errors} error(s)."
    )
    raise SystemExit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
