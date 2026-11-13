from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import load_json


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def resolve_source_case(episode_path: Path, episode: dict[str, Any], workspace: Path) -> Path | None:
    raw = str(episode.get("source_case_file") or "").strip()
    candidates = [Path(raw), workspace / raw, episode_path.parent / raw] if raw else []
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_episode(episode_path: Path, workspace: Path) -> str:
    episode = load_json(episode_path)
    source_path = resolve_source_case(episode_path, episode, workspace)
    source_case = load_json(source_path) if source_path else {}
    initial = source_case.get("initial_observation") or {}
    visible_initial = {
        "demographics": initial.get("demographics", ""),
        "encounter_context": initial.get("encounter_context", ""),
        "chief_complaint": initial.get("chief_complaint", ""),
        "first_contact_summary": initial.get("first_contact_summary") or initial.get("pre_referral_summary", ""),
        "known_prior_history": initial.get("known_prior_history") or [],
        "presenting_symptoms": initial.get("presenting_symptoms") or [],
        "initial_available_data": initial.get("initial_available_data") or [],
        "initial_context": initial.get("initial_context", ""),
    }

    lines = [
        f"# Interactive diagnosis transcript — {episode.get('case_id', episode_path.stem)}",
        "",
        f"- Clinician model: `{episode.get('clinician_model', '')}`",
        f"- Information model: `{episode.get('information_model', '')}`",
        f"- Maximum actions: `{episode.get('max_actions', '')}`",
        f"- Termination: `{episode.get('termination', '')}`",
        f"- Outcome category: `{(episode.get('final_outcome') or {}).get('outcome_category', '')}`",
        f"- In-budget actions used: `{episode.get('num_in_budget_actions', '')}`",
        "",
        "## Initial presentation shown to Clinician",
        "",
        json_block(visible_initial),
    ]

    for turn in episode.get("turns") or []:
        action_number = turn.get("action_number", "")
        outside = " — forced outside-budget final" if turn.get("outside_budget_forced_final") else ""
        lines.extend(
            [
                "",
                f"## Action {action_number}{outside}",
                "",
                "### Clinician",
                "",
                json_block(turn.get("clinician_action") or {}),
            ]
        )
        if turn.get("information_observation") is not None:
            lines.extend(
                [
                    "",
                    "### Information environment",
                    "",
                    json_block(turn.get("information_observation") or {}),
                ]
            )
            images = turn.get("attached_image_paths") or []
            if images:
                lines.extend(["", "Attached raw images:"])
                lines.extend(f"- `{path}`" for path in images)

    lines.extend(
        [
            "",
            "## Hidden reference (not shown to Clinician)",
            "",
            json_block(
                {
                    "case_level_diagnosis": (episode.get("hidden_reference") or {}).get(
                        "case_level_diagnosis", ""
                    ),
                    "final_diagnosis_verbatim": (episode.get("hidden_reference") or {}).get(
                        "final_diagnosis_verbatim", ""
                    ),
                    "diagnostic_reasoning_verbatim_spans": (episode.get("hidden_reference") or {}).get(
                        "diagnostic_reasoning_verbatim_spans", []
                    ),
                }
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render episode JSON as readable Markdown transcripts.")
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    output_dir = Path(args.output_dir)
    workspace = Path(args.workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(episode_dir.glob("*.json")):
        if path.name.startswith("_") or path.name.endswith(".failed.json"):
            continue
        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(render_episode(path, workspace), encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
