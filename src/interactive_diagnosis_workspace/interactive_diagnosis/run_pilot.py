from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Print commands for the Setting 1 pilot.")
    parser.parse_args()
    print("1. Extract three raw PMC cases with GPT-4.1:")
    print("python -m interactive_diagnosis.extraction_converter --limit 3")
    print()
    print("2. Audit and validate extracted JSON files before running episodes:")
    print("python -m interactive_diagnosis.audit_extractions --limit 3 --show_clean")
    print()
    print("3. Run o4-mini clinician with the deterministic source-grounded replay controller:")
    print("python -m interactive_diagnosis.episode_runner --limit 3")
    print()
    print("4. Optional budget curve for one case:")
    print("python -m interactive_diagnosis.episode_runner --case_ids PMC10362754 --max_actions 15")
    print("python -m interactive_diagnosis.episode_runner --case_ids PMC10362754 --max_actions 25")
    print("python -m interactive_diagnosis.episode_runner --case_ids PMC10362754 --max_actions 35")


if __name__ == "__main__":
    main()
