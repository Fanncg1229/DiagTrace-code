from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_judge_common import (
    TARGET_CONFIGS,
    call_foundry_model,
    get_foundry_client,
    load_json,
    save_json,
)


def _now() -> str:
    return datetime.now().isoformat()


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("’", "'").split())


def final_judge_case_files(judge_dirs: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for judge_dir in judge_dirs:
        cases_dir = judge_dir / "cases" if judge_dir.name != "cases" else judge_dir
        for path in sorted(cases_dir.glob("*.final_diagnosis_judge.json")):
            case_id = path.name.split(".")[0]
            out[case_id] = path
    return out


def non_final_ddx_turns(episode: dict[str, Any]) -> list[tuple[int, list[str]]]:
    out: list[tuple[int, list[str]]] = []
    for turn in episode.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        clinician = turn.get("clinician") if isinstance(turn.get("clinician"), dict) else {}
        if str(clinician.get("action") or "").upper() == "FINAL":
            continue
        ddx = clinician.get("current_ddx")
        if isinstance(ddx, list) and ddx:
            out.append((int(turn.get("turn_number") or len(out) + 1), [str(item) for item in ddx if str(item).strip()]))
    return out


def collect_last_window_ddx(episode: dict[str, Any], window: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    turns = non_final_ddx_turns(episode)[-window:]
    appearances: list[dict[str, Any]] = []
    unique: dict[str, dict[str, str]] = {}
    for turn_number, ddx in turns:
        for rank, text in enumerate(ddx, start=1):
            key = normalize_text(text)
            if not key:
                continue
            if key not in unique:
                unique[key] = {"id": f"M{len(unique) + 1}", "text": text}
            appearances.append({"turn_number": turn_number, "rank": rank, "text": text, "model_id": unique[key]["id"]})
    return appearances, list(unique.values())


def ddx_alignment_prompt(reference_items: list[dict[str, str]], model_items: list[dict[str, str]]) -> str:
    payload = {"reference_diagnoses": reference_items, "ddx_diagnoses": model_items}
    return f"""You are matching differential diagnosis items against extracted final-diagnosis entities.
You receive extracted items only, not raw source paragraphs.

Rules:
- Do not decompose items.
- Do not calculate TP, FP, FN, precision, recall, or F1.
- Do not match by list order.
- Classify each ddx_diagnosis independently.
- Do not use one-to-one matching: multiple ddx_diagnoses may align with the same reference_diagnosis.
- Full match requires preservation of the clinically essential diagnostic meaning. Medical synonyms and accepted abbreviations may fully match.
- Partial match requires a shared diagnostic evidence core between the reference and DDX item.
- Vague overlap, same organ only, same broad disease category only, same clinical context only, generic diagnostic language, or overlap through external inference should be classified as no match rather than partial match.
- No match means the DDX item does not share the same core diagnosis/evidence/finding as the reference item, describes a different entity/finding, is too generic to verify the reference item, or overlaps only through external inference.
- If the DDX item misses a critical etiology, organism, subtype, anatomical site, or named complication, classify as partial, not full.
- Different disease entities should not be treated as full matches.
- Extra unsupported diagnoses are not automatically contradictions.
- Contradiction requires direct incompatibility or opposition.

Reference: "fatal mucormycotic osteomyelitis caused by Rhizopus microsporus"
DDX item: "pelvic osteomyelitis complicated by gluteal and thigh abscesses"
Expected: partial / broader (misses fungal etiology and organism).

Reference: "benign schwannoma"
DDX item: "esophageal leiomyoma"
Expected: none (different diagnostic entities).

Input items:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "item_alignments": [
    {{
      "model_id": "M1",
      "alignment": "full",
      "matched_reference_ids": ["R1"],
      "match_type": "equivalent",
      "explanation": "..."
    }},
    {{
      "model_id": "M2",
      "alignment": "partial",
      "matched_reference_ids": ["R1"],
      "match_type": "broader",
      "explanation": "..."
    }},
    {{
      "model_id": "M3",
      "alignment": "none",
      "matched_reference_ids": [],
      "match_type": "none",
      "explanation": "..."
    }}
  ]
}}"""


def normalize_alignment_response(raw: Any, model_items: list[dict[str, str]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    model_ids = {item["id"] for item in model_items}
    alignments: dict[str, dict[str, Any]] = {}
    items = raw.get("item_alignments") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        warnings.append("item_alignments missing or not a list; treating all items as none")
        items = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(f"item_alignments[{index}] is not an object; dropped")
            continue
        model_id = str(item.get("model_id") or "").strip()
        if model_id not in model_ids:
            warnings.append(f"item_alignments[{index}] references unknown model_id={model_id}; dropped")
            continue
        alignment = str(item.get("alignment") or "none").strip().lower()
        if alignment not in {"full", "partial", "none", "contradiction"}:
            warnings.append(f"item_alignments[{index}] has invalid alignment={alignment}; set to none")
            alignment = "none"
        matched_reference_ids = item.get("matched_reference_ids")
        if not isinstance(matched_reference_ids, list):
            matched_reference_ids = []
        alignments[model_id] = {
            "model_id": model_id,
            "alignment": alignment,
            "matched_reference_ids": [str(value) for value in matched_reference_ids],
            "match_type": str(item.get("match_type") or alignment),
            "explanation": str(item.get("explanation") or ""),
        }
    for model_item in model_items:
        model_id = model_item["id"]
        alignments.setdefault(
            model_id,
            {
                "model_id": model_id,
                "alignment": "none",
                "matched_reference_ids": [],
                "match_type": "none",
                "explanation": "No alignment returned by judge.",
            },
        )
    return alignments, warnings


def load_reference_items(final_judge_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    data = load_json(final_judge_path)
    evaluation = data.get("final_diagnosis_evaluation") if isinstance(data.get("final_diagnosis_evaluation"), dict) else {}
    extraction = evaluation.get("reference_extraction") if isinstance(evaluation.get("reference_extraction"), dict) else {}
    items = extraction.get(TARGET_CONFIGS["final_diagnosis"].reference_item_key) or extraction.get("items") or []
    reference_items = [
        {"id": str(item.get("id") or f"R{idx + 1}"), "text": str(item.get("text") or "")}
        for idx, item in enumerate(items)
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    return reference_items, evaluation


def evaluate_case(
    *,
    case_id: str,
    episode_path: Path,
    final_judge_path: Path,
    output_dir: Path,
    window: int,
    overwrite: bool,
    matching_max_tokens: int,
    max_retries: int,
    retry_base_seconds: float,
    request_timeout: float,
) -> dict[str, Any]:
    output_path = output_dir / "cases" / f"{case_id}.ddx_alignment_judge.json"
    if output_path.exists() and not overwrite:
        existing = load_json(output_path)
        existing["status"] = "loaded_existing"
        return existing

    episode = load_json(episode_path)
    reference_items, final_eval = load_reference_items(final_judge_path)
    appearances, model_items = collect_last_window_ddx(episode, window)
    _client, foundry = get_foundry_client()

    payload: dict[str, Any] = {
        "case_id": case_id,
        "status": "completed",
        "created_at": _now(),
        "window": window,
        "episode_path": str(episode_path),
        "final_judge_path": str(final_judge_path),
        "reference_text": final_eval.get("reference_text", ""),
        "final_model_text": final_eval.get("model_text", ""),
        "reference_items": reference_items,
        "ddx_items": model_items,
        "ddx_appearances": appearances,
        "matcher_deployment": foundry.matcher_deployment,
        "foundry_endpoint_host": foundry.endpoint_host,
        "matching_max_tokens": matching_max_tokens,
    }
    if not reference_items or not model_items:
        payload.update(
            {
                "judge_status": "missing_items",
                "item_alignments": {},
                "validation_warnings": ["missing reference_items or ddx_items"],
                "error": "missing reference_items or ddx_items",
            }
        )
        save_json(output_path, payload)
        return payload

    try:
        prompt = ddx_alignment_prompt(reference_items, model_items)
        call = call_foundry_model(
            foundry.matcher_deployment,
            [
                {
                    "role": "system",
                    "content": "You are the semantic matcher. Return only valid JSON. Do not include markdown or prose outside JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=matching_max_tokens,
            temperature=0,
            role_name="ddx_alignment_matcher",
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            request_timeout=request_timeout,
        )
        alignments, warnings = normalize_alignment_response(call.parsed_json, model_items)
        payload.update(
            {
                "judge_status": "completed",
                "item_alignments": alignments,
                "validation_warnings": warnings,
                "raw_matcher_response": call.raw_response,
                "matcher_model": call.model,
                "finish_reason": call.finish_reason,
                "usage": call.usage,
                "response_format_used": call.response_format_used,
                "error": "",
            }
        )
    except Exception as exc:
        payload.update(
            {
                "judge_status": "matcher_failed",
                "item_alignments": {},
                "validation_warnings": [],
                "error": str(exc),
            }
        )
    save_json(output_path, payload)
    return payload


def iter_episode_paths(run_dirs: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for run_dir in run_dirs:
        for path in sorted((run_dir / "episodes").glob("*.json")):
            out.setdefault(path.stem, path)
    return out


def alignment_rank_state(case_result: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    allowed = {"full"} if policy == "strict" else {"full", "partial"}
    alignments = case_result.get("item_alignments") if isinstance(case_result.get("item_alignments"), dict) else {}
    rows: list[dict[str, Any]] = []
    for appearance in case_result.get("ddx_appearances", []) or []:
        if not isinstance(appearance, dict):
            continue
        model_id = str(appearance.get("model_id") or "")
        alignment = alignments.get(model_id, {}) if isinstance(alignments.get(model_id), dict) else {}
        if str(alignment.get("alignment")) in allowed:
            rows.append(
                {
                    "case_id": case_result.get("case_id", ""),
                    "turn_number": appearance.get("turn_number", ""),
                    "rank": appearance.get("rank", ""),
                    "text": appearance.get("text", ""),
                    "alignment": alignment.get("alignment", ""),
                    "matched_reference_ids": "|".join(alignment.get("matched_reference_ids", []) or []),
                    "match_type": alignment.get("match_type", ""),
                    "explanation": alignment.get("explanation", ""),
                }
            )
    return rows


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": _now(),
        "num_cases": len(results),
        "num_completed": sum(item.get("judge_status") == "completed" for item in results),
        "num_failed": sum(item.get("judge_status") not in {"completed", "loaded_existing"} for item in results),
        "cases": [
            {
                "case_id": item.get("case_id"),
                "judge_status": item.get("judge_status"),
                "num_ddx_items": len(item.get("ddx_items", []) or []),
                "error": item.get("error", ""),
            }
            for item in results
        ],
    }
    save_json(output_dir / "_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for item in results:
        alignments = item.get("item_alignments") if isinstance(item.get("item_alignments"), dict) else {}
        by_id = {str(ddx.get("id")): ddx for ddx in item.get("ddx_items", []) or [] if isinstance(ddx, dict)}
        for model_id, alignment in alignments.items():
            ddx = by_id.get(str(model_id), {})
            rows.append(
                {
                    "case_id": item.get("case_id", ""),
                    "model_id": model_id,
                    "ddx_text": ddx.get("text", ""),
                    "alignment": alignment.get("alignment", ""),
                    "matched_reference_ids": "|".join(alignment.get("matched_reference_ids", []) or []),
                    "match_type": alignment.get("match_type", ""),
                    "explanation": alignment.get("explanation", ""),
                }
            )
    with (output_dir / "ddx_item_alignment_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["case_id", "model_id", "ddx_text", "alignment", "matched_reference_ids", "match_type", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge last-window DDX items against article final diagnosis.")
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--final-judge-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--matching-max-tokens", type=int, default=4000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()

    episode_paths = iter_episode_paths(args.run_dir)
    judge_paths = final_judge_case_files(args.final_judge_dir)
    case_ids = args.case_id or sorted(set(episode_paths) & set(judge_paths))
    results: list[dict[str, Any]] = []
    for case_id in case_ids:
        if case_id not in episode_paths or case_id not in judge_paths:
            results.append({"case_id": case_id, "judge_status": "missing_episode_or_final_judge", "error": "missing episode or final judge"})
            continue
        results.append(
            evaluate_case(
                case_id=case_id,
                episode_path=episode_paths[case_id],
                final_judge_path=judge_paths[case_id],
                output_dir=args.output_dir,
                window=args.window,
                overwrite=args.overwrite,
                matching_max_tokens=args.matching_max_tokens,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                request_timeout=args.request_timeout,
            )
        )
    write_summary(args.output_dir, results)
    print(json.dumps({"output_dir": str(args.output_dir), "num_cases": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
