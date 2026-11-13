from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _line(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def episode_to_markdown(path: Path) -> str:
    episode = _read_json(path)
    lines = [
        f"## {episode.get('case_id', path.stem)}",
        "",
        f"Known final diagnosis: {_line(episode.get('known_final_diagnosis'))}",
        f"Turns used: {episode.get('turns_used')} / {episode.get('max_turns')}",
        f"Termination: {episode.get('termination')}",
        "",
    ]
    for turn in episode.get("turns", []) or []:
        clinician = turn.get("clinician") if isinstance(turn, dict) else {}
        report = turn.get("report") if isinstance(turn, dict) else None
        lines.append(f"### Turn {turn.get('turn_number')}")
        lines.append("")
        if isinstance(clinician, dict):
            action = clinician.get("action", "")
            request = clinician.get("request", "")
            lines.append(f"A ({action}): {_line(request)}")
            if clinician.get("current_trajectory_hypothesis"):
                lines.append(f"- hypothesis: {_line(clinician.get('current_trajectory_hypothesis'))}")
            if clinician.get("missing_key_evidence"):
                lines.append(f"- missing: {_line(clinician.get('missing_key_evidence'))}")
            if action == "FINAL_TRAJECTORY":
                lines.append(f"- trajectory_summary: {_line(clinician.get('trajectory_summary'))}")
                lines.append(f"- key_supporting_evidence: {_line(clinician.get('key_supporting_evidence'))}")
                lines.append(f"- remaining_uncertainties: {_line(clinician.get('remaining_uncertainties'))}")
        if isinstance(report, dict):
            lines.append("")
            lines.append(f"B ({report.get('status', '')}, {report.get('evidence_id', '')}):")
            if report.get("source_text"):
                lines.append(f"> {report.get('source_text')}")
            elif report.get("explanation"):
                lines.append(f"> {report.get('explanation')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown transcripts for trajectory workflow episodes.")
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parts = [episode_to_markdown(path) for path in sorted(args.episode_dir.glob("*.json"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
