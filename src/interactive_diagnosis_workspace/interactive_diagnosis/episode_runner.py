from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    add_token_usage,
    azure_client,
    default_model,
    empty_token_usage,
    encode_image,
    extract_json_object,
    load_json,
    save_json,
    token_usage,
)
from .action_schema import CLINICIAN_ACTIONS, normalize_action
from .outcome_evaluator import evaluate_final_outcome
from .prompts import CLINICIAN_ACTION_INSTRUCTIONS, CLINICIAN_SYSTEM_PROMPT
from .schemas import validate_case
from .state_controller import DeterministicStateController


def initial_prompt(case_data: dict[str, Any], max_actions: int) -> str:
    initial = case_data.get("initial_observation") or {}
    visible = {
        "demographics": initial.get("demographics", ""),
        "encounter_context": initial.get("encounter_context", ""),
        "chief_complaint": initial.get("chief_complaint", ""),
        "first_contact_summary": initial.get("first_contact_summary") or initial.get("pre_referral_summary", ""),
        "known_prior_history": initial.get("known_prior_history") or [],
        "presenting_symptoms": initial.get("presenting_symptoms") or [],
        "initial_available_data": initial.get("initial_available_data") or [],
        "initial_context": initial.get("initial_context", ""),
    }
    return (
        "Initial patient presentation:\n"
        f"{json.dumps(visible, ensure_ascii=False, indent=2)}\n\n"
        f"You have {max_actions} total actions, including FINAL. Choose the highest-value next action.\n\n"
        f"{CLINICIAN_ACTION_INSTRUCTIONS}"
    )


def observation_text(observation: dict[str, Any], action_number: int, max_actions: int) -> str:
    safe = {key: value for key, value in observation.items() if key != "image_paths"}
    return (
        "Information environment response:\n"
        f"{json.dumps(safe, ensure_ascii=False, indent=2)}\n\n"
        f"Actions used: {action_number}/{max_actions}. Actions remaining: {max_actions - action_number}.\n"
        "Update the differential using all evidence disclosed so far, then submit FINAL or ask one focused next question.\n\n"
        f"{CLINICIAN_ACTION_INSTRUCTIONS}"
    )


def multimodal_observation_content(observation: dict[str, Any], text_prompt: str) -> tuple[str | list[dict[str, Any]], list[str]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    attached: list[str] = []
    for raw_path in observation.get("image_paths") or []:
        path = Path(raw_path)
        encoded = encode_image(path)
        if not encoded:
            continue
        content.append({"type": "image_url", "image_url": {"url": encoded, "detail": "high"}})
        attached.append(str(path))
    return (content if attached else text_prompt), attached


def final_quality_flags(action: dict[str, Any], disclosed_ids: set[str]) -> list[str]:
    flags: list[str] = []
    diagnosis = str(action.get("final_diagnosis") or "").strip()
    certainty = str(action.get("diagnostic_certainty") or "").lower()
    if not disclosed_ids:
        flags.append("no new source-grounded evidence has been acquired beyond the initial presentation")
    if re.search(r"\b(low|moderate|uncertain|possible|approximately\s*[0-6][0-9]%)\b", certainty):
        flags.append("diagnostic certainty is stated as low or moderate")
    if re.search(r"\b(or|versus|vs\.?|and/or)\b", diagnosis, re.IGNORECASE):
        flags.append("final_diagnosis contains more than one unresolved alternative")
    return flags


def normalize_diagnosis(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def final_matches_hidden_reference(action: dict[str, Any], case_data: dict[str, Any]) -> bool:
    predicted = normalize_diagnosis(action.get("final_diagnosis"))
    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    candidates = [hidden.get("case_level_diagnosis"), hidden.get("final_diagnosis_verbatim")]
    candidates.extend(hidden.get("diagnosis_synonyms_or_variants") or [])
    variants = [normalize_diagnosis(value) for value in candidates if normalize_diagnosis(value)]
    if not predicted or not variants:
        return False
    if "hepatocellular carcinoma" in predicted and any(
        value in {"hepatocellular carcinoma", "hcc"} or "hepatocellular carcinoma" in value
        for value in variants
    ):
        return True
    for variant in variants:
        if predicted == variant:
            return True
        if len(variant) >= 8 and variant in predicted:
            return True
        if len(predicted) >= 8 and predicted in variant:
            return True
    return False


class ClinicianAgent:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 4000,
        timeout: int = 600,
    ) -> None:
        self.client = azure_client(timeout)
        self.model = model or default_model()
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    async def act(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, int]]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        return raw, normalize_action(extract_json_object(raw)), token_usage(response)


class InteractiveEpisodeRunner:
    def __init__(
        self,
        clinician_model: str | None = None,
        information_model: str | None = None,
        clinician_reasoning_effort: str = "medium",
        information_reasoning_effort: str = "low",
        max_actions_override: int | None = None,
        force_final_on_exhaustion: bool = True,
        timeout: int = 600,
    ) -> None:
        self.clinician = ClinicianAgent(
            model=clinician_model,
            reasoning_effort=clinician_reasoning_effort,
            timeout=timeout,
        )
        self.information_model = information_model or "deterministic-state-controller"
        self.max_actions_override = max_actions_override
        self.force_final_on_exhaustion = force_final_on_exhaustion

    async def run_case(self, case_path: Path) -> dict[str, Any]:
        case_data = load_json(case_path)
        validation = validate_case(case_data)
        if not validation.ok:
            raise ValueError(f"invalid extracted case: {validation.errors}")

        configured = int((case_data.get("budget") or {}).get("max_actions", 15) or 15)
        max_actions = self.max_actions_override if self.max_actions_override is not None else max(configured, 15)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CLINICIAN_SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt(case_data, max_actions)},
        ]
        cleaned_conversation: list[dict[str, Any]] = [dict(message) for message in messages]
        controller = DeterministicStateController(case_data)
        turns: list[dict[str, Any]] = []
        clinician_usage = empty_token_usage()
        information_usage = empty_token_usage()
        termination = "budget_exhausted"
        final_analysis: dict[str, Any] = {}
        final_outcome: dict[str, Any] = {}

        for action_number in range(1, max_actions + 1):
            print(f"[episode] {case_data.get('case_id')}: action {action_number}/{max_actions}", flush=True)
            raw, action, usage = await self.clinician.act(messages)
            add_token_usage(clinician_usage, usage)
            messages.append({"role": "assistant", "content": raw})
            cleaned_conversation.append({"role": "assistant", "content": raw})

            turn: dict[str, Any] = {
                "action_number": action_number,
                "clinician_action": action,
                "clinician_raw": raw,
                "clinician_token_usage": usage,
            }
            if action["action"] == "FINAL":
                final_correct = final_matches_hidden_reference(action, case_data)
                outcome = evaluate_final_outcome(
                    final_action=action,
                    final_correct=final_correct,
                    case_data=case_data,
                    turns=turns,
                )
                termination = "final_submitted"
                final_analysis = action
                final_outcome = outcome
                turn["final_correct"] = final_correct
                turn["final_outcome"] = outcome
                if not final_correct:
                    termination = "final_submitted_incorrect"
                quality_flags = final_quality_flags(action, controller.state.disclosed_ids)
                if quality_flags:
                    turn["final_quality_flags"] = quality_flags
                turns.append(turn)
                break

            controller_result = controller.handle(action)
            observation = controller_result["public"]
            turn["controller_observation_internal"] = controller_result["internal"]
            turn["information_observation"] = observation
            turn["information_token_usage"] = empty_token_usage()

            prompt = observation_text(observation, action_number, max_actions)
            content, attached = multimodal_observation_content(
                {"image_paths": controller_result.get("image_paths") or []},
                prompt,
            )
            turn["attached_image_paths"] = attached
            messages.append({"role": "user", "content": content})
            cleaned_conversation.append(
                {
                    "role": "user",
                    "content": prompt,
                    "attached_image_paths": attached,
                }
            )
            turns.append(turn)

        if termination == "budget_exhausted" and self.force_final_on_exhaustion:
            forced_prompt = (
                "The interaction budget is exhausted. You may not request more information. Submit your best "
                "FINAL diagnosis now using only the disclosed evidence.\n\n"
                f"{CLINICIAN_ACTION_INSTRUCTIONS}"
            )
            messages.append({"role": "user", "content": forced_prompt})
            cleaned_conversation.append({"role": "user", "content": forced_prompt})
            try:
                raw, forced, usage = await self.clinician.act(messages)
                add_token_usage(clinician_usage, usage)
                if forced.get("action") == "FINAL":
                    final_analysis = forced
                    final_correct = final_matches_hidden_reference(forced, case_data)
                    final_outcome = evaluate_final_outcome(
                        final_action=forced,
                        final_correct=final_correct,
                        case_data=case_data,
                        turns=turns,
                    )
                turns.append(
                    {
                        "action_number": max_actions + 1,
                        "outside_budget_forced_final": True,
                        "clinician_action": forced,
                        "clinician_raw": raw,
                        "clinician_token_usage": usage,
                        "final_outcome": final_outcome if forced.get("action") == "FINAL" else {},
                    }
                )
            except Exception as exc:
                turns.append({"outside_budget_forced_final": True, "error": str(exc)})

        return {
            "protocol_version": "2026-07-12-source-grounded-clinical-replay-v9-component-order-lifecycle",
            "case_id": case_data.get("case_id", case_path.stem),
            "source_case_file": str(case_path),
            "created_at": datetime.now().isoformat(),
            "clinician_model": self.clinician.model,
            "information_model": self.information_model,
            "role_contexts_separate": True,
            "max_actions": max_actions,
            "action_space": sorted(CLINICIAN_ACTIONS),
            "termination": termination,
            "num_in_budget_actions": min(len([t for t in turns if not t.get("outside_budget_forced_final")]), max_actions),
            "disclosed_evidence_ids": sorted(controller.state.disclosed_ids),
            "final_phase_index": controller.state.current_phase_index,
            "orders": controller.state.orders,
            "turns": turns,
            "final_analysis": final_analysis,
            "final_outcome": final_outcome,
            "hidden_reference": case_data.get("hidden_reference", {}),
            "clinician_token_usage": clinician_usage,
            "information_token_usage": information_usage,
            "cleaned_clinician_conversation": cleaned_conversation,
            "case_validation_warnings": validation.warnings,
        }


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run Setting 1 interactive diagnosis episodes.")
    parser.add_argument("--input_dir", default="data/pmc_interactive_extraction/internalmed")
    parser.add_argument("--output_dir", default="results/episodes/setting1-clinical-replay-v9")
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--clinician_model", default=None)
    parser.add_argument("--information_model", default=None)
    parser.add_argument("--clinician_reasoning_effort", default="medium")
    parser.add_argument("--information_reasoning_effort", default="low")
    parser.add_argument("--max_actions", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_force_final_on_exhaustion", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    cases = sorted(
        path
        for path in input_dir.glob("*.json")
        if not path.name.startswith("_") and not path.name.endswith(".failed.json")
    )
    if args.case_ids:
        selected = {value.strip() for item in args.case_ids for value in item.split(",") if value.strip()}
        cases = [path for path in cases if path.stem in selected]
    if args.limit is not None:
        cases = cases[: args.limit]

    runner = InteractiveEpisodeRunner(
        clinician_model=args.clinician_model,
        information_model=args.information_model,
        clinician_reasoning_effort=args.clinician_reasoning_effort,
        information_reasoning_effort=args.information_reasoning_effort,
        max_actions_override=args.max_actions,
        force_final_on_exhaustion=not args.no_force_final_on_exhaustion,
        timeout=args.timeout,
    )
    model_label = runner.clinician.model.replace("/", "_")
    output_dir = Path(args.output_dir) / model_label
    manifest_cases = []
    for case_path in cases:
        output_path = output_dir / case_path.name
        if output_path.exists() and not args.overwrite:
            print(f"[episode] {case_path.stem}: skipped existing", flush=True)
            manifest_cases.append({"case_id": case_path.stem, "status": "skipped_existing"})
            continue
        try:
            result = await runner.run_case(case_path)
            save_json(output_path, result)
            status = "completed"
        except Exception as exc:
            status = "failed"
            save_json(output_path.with_suffix(".failed.json"), {"case_id": case_path.stem, "error": str(exc)})
        manifest_cases.append({"case_id": case_path.stem, "status": status, "output_path": str(output_path)})

    case_index = {str(item.get("case_id")): item for item in manifest_cases}
    for existing in sorted(output_dir.glob("*.json")):
        if existing.name.startswith("_") or existing.name.endswith(".failed.json"):
            continue
        case_index.setdefault(
            existing.stem,
            {
                "case_id": existing.stem,
                "status": "existing_completed",
                "output_path": str(existing),
            },
        )
    all_manifest_cases = [case_index[key] for key in sorted(case_index)]

    save_json(
        output_dir / "_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "protocol_version": "2026-07-12-source-grounded-clinical-replay-v9-component-order-lifecycle",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "clinician_model": runner.clinician.model,
            "information_model": runner.information_model,
            "cases": all_manifest_cases,
        },
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
