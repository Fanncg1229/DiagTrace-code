from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator import evaluate_episode, summarize_evaluations
from .extraction import extract_from_nxml
from .workflow import DialogueWorkflow


PROTOCOL_VERSION = "2026-07-dialogue-workflow-setting1-v6-safe-report"


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


def find_nxml(input_root: Path, case_id: str) -> Path:
    case_directory = input_root / case_id
    matches = sorted(case_directory.glob("*.nxml"))
    if not matches:
        raise FileNotFoundError(f"No NXML found for {case_id} under {case_directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple NXML files found for {case_id}: {matches}")
    return matches[0]


def run_case(
    *,
    case_id: str,
    nxml_path: Path,
    run_dir: Path,
    extraction_model: str | None,
    clinician_model: str | None,
    report_model: str | None,
    max_turns: int = 25,
    resume: bool = True,
    rerun_episode: bool = False,
    case_cache_dir: Path | None = None,
) -> dict[str, Any]:
    case_path = run_dir / "cases" / f"{case_id}.json"
    episode_path = run_dir / "episodes" / f"{case_id}.json"
    evaluation_path = run_dir / "evaluations" / f"{case_id}.json"
    partial_path = run_dir / "partials" / f"{case_id}.json"
    error_path = run_dir / "errors" / f"{case_id}.json"

    if resume and not rerun_episode and episode_path.exists() and evaluation_path.exists():
        existing_episode = _read_json(episode_path)
        mismatches = []
        if clinician_model and existing_episode.get("clinician_model") != clinician_model:
            mismatches.append(
                f"clinician_model={existing_episode.get('clinician_model')} (requested {clinician_model})"
            )
        if report_model and existing_episode.get("report_model") != report_model:
            mismatches.append(
                f"report_model={existing_episode.get('report_model')} (requested {report_model})"
            )
        if int(existing_episode.get("max_turns") or 0) != max_turns:
            mismatches.append(
                f"max_turns={existing_episode.get('max_turns')} (requested {max_turns})"
            )
        if mismatches:
            raise ValueError(
                "Existing episode configuration mismatch; use --rerun-episodes: "
                + "; ".join(mismatches)
            )
        error_path.unlink(missing_ok=True)
        return {
            "case_id": case_id,
            "status": "existing_completed",
            "case_path": str(case_path),
            "episode_path": str(episode_path),
            "evaluation_path": str(evaluation_path),
        }

    if case_path.exists() and (resume or rerun_episode):
        case_data = _read_json(case_path)
        extraction_status = "existing"
    elif case_cache_dir is not None and (case_cache_dir / f"{case_id}.json").exists():
        case_data = _read_json(case_cache_dir / f"{case_id}.json")
        _write_json(case_path, case_data)
        extraction_status = "cached"
    else:
        case_data = extract_from_nxml(
            nxml_path,
            case_id=case_id,
            model=extraction_model,
        )
        _write_json(case_path, case_data)
        extraction_status = "completed"

    partial_episode = (
        _read_json(partial_path)
        if resume and not rerun_episode and partial_path.exists()
        else None
    )
    workflow = DialogueWorkflow(
        case_data,
        clinician_model=clinician_model,
        report_model=report_model,
        max_turns=max_turns,
        resume_episode=partial_episode,
    )

    def checkpoint(payload: dict[str, Any]) -> None:
        payload.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "updated_at": _now(),
                "source_case_file": str(case_path),
                "source_nxml": str(nxml_path),
                "clinician_model": workflow.clinician_model,
                "report_model": workflow.report_model,
                "role_contexts_separate": True,
            }
        )
        _write_json(partial_path, payload)

    episode = workflow.run_episode(on_turn=checkpoint)
    episode.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "created_at": _now(),
            "source_case_file": str(case_path),
            "source_nxml": str(nxml_path),
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
    partial_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)
    return {
        "case_id": case_id,
        "status": "completed",
        "extraction_status": extraction_status,
        "case_path": str(case_path),
        "episode_path": str(episode_path),
        "evaluation_path": str(evaluation_path),
    }


def run_batch(
    *,
    case_ids: list[str],
    input_root: Path,
    run_dir: Path,
    extraction_model: str | None = None,
    clinician_model: str | None = None,
    report_model: str | None = None,
    max_turns: int = 25,
    resume: bool = True,
    workers: int = 1,
    rerun_episodes: bool = False,
    case_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if not case_ids:
        raise ValueError("At least one case_id is required")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "input_root": str(input_root),
        "run_dir": str(run_dir),
        "max_turns": max_turns,
        "requested_extraction_model": extraction_model or "env/default",
        "requested_clinician_model": clinician_model or "env/default",
        "requested_report_model": report_model or "env/default",
        "resume": resume,
        "workers": workers,
        "rerun_episodes": rerun_episodes,
        "case_cache_dir": str(case_cache_dir) if case_cache_dir else None,
        "cases": [],
    }

    def execute(case_id: str) -> dict[str, Any]:
        try:
            return run_case(
                case_id=case_id,
                nxml_path=find_nxml(input_root, case_id),
                run_dir=run_dir,
                extraction_model=extraction_model,
                clinician_model=clinician_model,
                report_model=report_model,
                max_turns=max_turns,
                resume=resume,
                rerun_episode=rerun_episodes,
                case_cache_dir=case_cache_dir,
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

    case_order = {case_id: index for index, case_id in enumerate(case_ids)}
    manifest["cases"].sort(key=lambda item: case_order.get(str(item.get("case_id")), len(case_order)))
    evaluations: list[dict[str, Any]] = []
    for result in manifest["cases"]:
        evaluation_path_value = result.get("evaluation_path")
        if evaluation_path_value:
            evaluation_path = Path(evaluation_path_value)
            if evaluation_path.exists():
                evaluations.append(_read_json(evaluation_path))

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
    parser = argparse.ArgumentParser(description="Run the Setting 1 two-LLM diagnosis workflow")
    parser.add_argument("--case-id", action="append", required=True, help="Repeat for multiple PMC cases")
    parser.add_argument("--input-root", type=Path, default=Path("internalmed"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("interactive_diagnosis_workspace/dialogue_workflow/results/pilot"),
    )
    parser.add_argument("--extraction-model", default=None)
    parser.add_argument("--clinician-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--case-cache-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of cases to run concurrently")
    parser.add_argument("--force", action="store_true", help="Rerun and overwrite existing case outputs")
    parser.add_argument(
        "--rerun-episodes",
        action="store_true",
        help="Reuse extracted case JSON but rerun and overwrite episodes/evaluations",
    )
    args = parser.parse_args()

    manifest = run_batch(
        case_ids=args.case_id,
        input_root=args.input_root,
        run_dir=args.run_dir,
        extraction_model=args.extraction_model,
        clinician_model=args.clinician_model,
        report_model=args.report_model,
        max_turns=args.max_turns,
        resume=not args.force,
        workers=args.workers,
        rerun_episodes=args.rerun_episodes,
        case_cache_dir=args.case_cache_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
