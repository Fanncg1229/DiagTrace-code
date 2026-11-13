from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator import evaluate_episode, summarize_evaluations
from .workflow import TrajectoryWorkflow


PROTOCOL_VERSION = "2026-07-trajectory-workflow-setting2-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def run_case(
    *,
    case_id: str,
    case_dir: Path,
    run_dir: Path,
    clinician_model: str | None = None,
    report_model: str | None = None,
    max_turns: int = 15,
    resume: bool = True,
) -> dict[str, Any]:
    source_case_path = case_dir / f"{case_id}.json"
    case_path = run_dir / "cases" / f"{case_id}.json"
    episode_path = run_dir / "episodes" / f"{case_id}.json"
    evaluation_path = run_dir / "evaluations" / f"{case_id}.json"
    error_path = run_dir / "errors" / f"{case_id}.json"

    if resume and episode_path.exists() and evaluation_path.exists():
        return {
            "case_id": case_id,
            "status": "existing_completed",
            "case_path": str(case_path),
            "episode_path": str(episode_path),
            "evaluation_path": str(evaluation_path),
        }

    case_data = _read_json(source_case_path)
    _write_json(case_path, case_data)
    workflow = TrajectoryWorkflow(
        case_data,
        clinician_model=clinician_model,
        report_model=report_model,
        max_turns=max_turns,
    )
    episode = workflow.run_episode()
    episode.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "created_at": _now(),
            "source_case_file": str(case_path),
            "source_cache_file": str(source_case_path),
            "clinician_model": workflow.clinician_model,
            "report_model": workflow.report_model,
            "extraction_model": case_data.get("source_metadata", {}).get("extraction_model", ""),
            "role_contexts_separate": True,
        }
    )
    _write_json(episode_path, episode)
    evaluation = evaluate_episode(case_data, episode)
    evaluation["protocol_version"] = PROTOCOL_VERSION
    evaluation["created_at"] = _now()
    _write_json(evaluation_path, evaluation)
    error_path.unlink(missing_ok=True)
    return {
        "case_id": case_id,
        "status": "completed",
        "case_path": str(case_path),
        "episode_path": str(episode_path),
        "evaluation_path": str(evaluation_path),
    }


def run_batch(
    *,
    case_ids: list[str],
    case_dir: Path,
    run_dir: Path,
    clinician_model: str | None = None,
    report_model: str | None = None,
    max_turns: int = 15,
    resume: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "case_dir": str(case_dir),
        "run_dir": str(run_dir),
        "max_turns": max_turns,
        "requested_clinician_model": clinician_model or "env/default",
        "requested_report_model": report_model or "env/default",
        "resume": resume,
        "workers": workers,
        "cases": [],
    }

    def execute(case_id: str) -> dict[str, Any]:
        try:
            return run_case(
                case_id=case_id,
                case_dir=case_dir,
                run_dir=run_dir,
                clinician_model=clinician_model,
                report_model=report_model,
                max_turns=max_turns,
                resume=resume,
            )
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_at": _now(),
            }
            _write_json(run_dir / "errors" / f"{case_id}.json", failure)
            return failure

    worker_count = max(1, min(int(workers), len(case_ids)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(execute, case_id) for case_id in case_ids]
        for future in as_completed(futures):
            manifest["cases"].append(future.result())
            manifest["updated_at"] = _now()
            _write_json(run_dir / "_manifest.json", manifest)

    order = {case_id: index for index, case_id in enumerate(case_ids)}
    manifest["cases"].sort(key=lambda item: order.get(str(item.get("case_id")), len(order)))
    evaluations: list[dict[str, Any]] = []
    for result in manifest["cases"]:
        path_value = result.get("evaluation_path")
        if path_value and Path(path_value).exists():
            evaluations.append(_read_json(Path(path_value)))
    summary = summarize_evaluations(evaluations)
    summary.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "created_at": _now(),
            "num_requested": len(case_ids),
            "num_failed": sum(item["status"] == "failed" for item in manifest["cases"]),
        }
    )
    _write_json(run_dir / "_summary.json", summary)
    manifest["summary_path"] = str(run_dir / "_summary.json")
    manifest["updated_at"] = _now()
    _write_json(run_dir / "_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Setting 2 diagnosis-aware trajectory reconstruction workflow")
    parser.add_argument("--case-id", action="append", required=True, help="Repeat for multiple PMC cases")
    parser.add_argument("--case-dir", type=Path, required=True, help="Directory containing extracted case JSON files")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clinician-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = run_batch(
        case_ids=args.case_id,
        case_dir=args.case_dir,
        run_dir=args.run_dir,
        clinician_model=args.clinician_model,
        report_model=args.report_model,
        max_turns=args.max_turns,
        resume=not args.force,
        workers=args.workers,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
