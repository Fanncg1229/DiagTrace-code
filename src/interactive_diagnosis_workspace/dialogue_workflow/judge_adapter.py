from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _reference_reasoning(case_data: dict[str, Any]) -> str:
    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    spans = hidden.get("diagnostic_reasoning_verbatim_spans") if isinstance(hidden, dict) else []
    if isinstance(spans, list):
        return "\n".join(str(span).strip() for span in spans if str(span).strip())
    return ""


def convert_case(case_path: Path, episode_path: Path) -> dict[str, Any]:
    case_data = _read_json(case_path)
    episode = _read_json(episode_path)
    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    final = episode.get("final") if isinstance(episode.get("final"), dict) else {}

    return {
        "case_name": str(case_data.get("case_id") or episode.get("case_id") or episode_path.stem),
        "status": "completed" if final else "failed",
        "created_at": _now(),
        "source_case_file": str(case_path),
        "source_episode_file": str(episode_path),
        "loaded_case": {
            "ground_truth_final_diagnosis": str(hidden.get("final_diagnosis_verbatim") or ""),
            "final_diagnosis": str(hidden.get("final_diagnosis_verbatim") or ""),
            "accepted_aliases": hidden.get("accepted_aliases", []) if isinstance(hidden.get("accepted_aliases"), list) else [],
            "ground_truth_diagnostic_reasoning": _reference_reasoning(case_data),
            "diagnostic_reasoning": _reference_reasoning(case_data),
            "image_metadata": [],
        },
        "model_result": {
            "final_analysis": {
                "final_diagnosis": str(final.get("diagnosis") or ""),
                "diagnostic_reasoning": str(final.get("reasoning") or ""),
            }
        },
        "dialogue_metadata": {
            "turns_used": episode.get("turns_used"),
            "max_turns": episode.get("max_turns"),
            "termination": episode.get("termination"),
            "disclosed_evidence_ids": episode.get("disclosed_evidence_ids", []),
            "supporting_evidence_ids": final.get("supporting_evidence_ids", []),
            "confidence": final.get("confidence"),
            "protocol_version": episode.get("protocol_version"),
        },
    }


def convert_run(run_dir: Path, output_dir: Path, case_ids: list[str] | None = None) -> dict[str, Any]:
    case_dir = run_dir / "cases"
    episode_dir = run_dir / "episodes"
    selected = case_ids or sorted(path.stem for path in episode_dir.glob("*.json") if not path.name.startswith("_"))
    manifest: dict[str, Any] = {
        "created_at": _now(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "cases": [],
    }
    for case_id in selected:
        case_path = case_dir / f"{case_id}.json"
        episode_path = episode_dir / f"{case_id}.json"
        try:
            adapted = convert_case(case_path, episode_path)
            output_path = output_dir / f"{case_id}.json"
            _write_json(output_path, adapted)
            manifest["cases"].append(
                {
                    "case_id": case_id,
                    "status": "converted",
                    "output_path": str(output_path),
                    "case_path": str(case_path),
                    "episode_path": str(episode_path),
                }
            )
        except Exception as exc:
            manifest["cases"].append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "case_path": str(case_path),
                    "episode_path": str(episode_path),
                }
            )
    _write_json(output_dir / "_adapter_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dialogue workflow episodes into llm_as_judge_eval inputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=None, help="Repeat to convert selected cases only.")
    args = parser.parse_args()
    manifest = convert_run(args.run_dir, args.output_dir, args.case_id)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
