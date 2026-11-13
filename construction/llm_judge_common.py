#!/usr/bin/env python3
"""Shared multi-model LLM-as-judge utilities.

The evaluation architecture is intentionally split into isolated extraction
calls and matcher-only calls:

reference raw text -> extractor -> reference items
model raw text     -> extractor -> model items
items only         -> matcher   -> semantic relationships
metrics            -> Python
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError


DIAGNOSIS_EXTRACTION_PROMPT_VERSION = "2026-06-15-diagnosis-extract-v2-clinical-modifiers"
DIAGNOSIS_MATCHING_PROMPT_VERSION = "2026-06-15-diagnosis-match-v3-core-evidence"
REASONING_EXTRACTION_PROMPT_VERSION = "2026-06-15-reasoning-extract-v3-gt-aware-evidence-filter"
REASONING_MATCHING_PROMPT_VERSION = "2026-06-15-reasoning-match-v3-core-evidence"
IMAGE_EXTRACTION_PROMPT_VERSION = "2026-06-15-image-extract-v2-visible-filter"
IMAGE_MATCHING_PROMPT_VERSION = "2026-06-15-image-match-v2-visual-attributes"
CONSISTENCY_SCHEMA_VERSION = "2026-06-08-reasoning-consistency-v1"

DEFAULT_EXTRACTOR_DEPLOYMENT = "gpt-4.1"
DEFAULT_MATCHER_DEPLOYMENT = "gpt-5.1"
VALID_CASE_STATUSES = {"completed", "completed_with_blocked_images"}

_CLIENT: Optional[OpenAI] = None
_CONFIG: Optional["FoundryConfig"] = None
_JSON_MODE_SUPPORTED_BY_DEPLOYMENT: Dict[str, bool] = {}
_TOKEN_PARAM_BY_DEPLOYMENT: Dict[str, str] = {}
_TEMPERATURE_SUPPORTED_BY_DEPLOYMENT: Dict[str, bool] = {}


@dataclass(frozen=True)
class FoundryConfig:
    api_key: str
    base_url: str
    extractor_deployment: str
    matcher_deployment: str

    @property
    def endpoint_host(self) -> str:
        return urlparse(self.base_url).netloc


@dataclass
class ModelCallResult:
    parsed_json: Dict[str, Any]
    raw_response: str
    deployment: str
    role_name: str
    model: Optional[str]
    finish_reason: Optional[str]
    usage: Dict[str, Any]
    created_at: str
    response_format_used: bool


@dataclass(frozen=True)
class EvaluationTargetConfig:
    target_name: str
    reference_item_key: str
    model_item_key: str
    reference_id_prefix: str
    model_id_prefix: str
    extraction_prompt_version: str
    matching_prompt_version: str
    extraction_prompt_builder: Callable[[str, str, Dict[str, Any]], str]
    matching_prompt_builder: Callable[[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]], str]
    extra_rate_key: str
    item_label: str


class JudgeParseError(ValueError):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_foundry_base_url(base_url: Optional[str] = None, endpoint: Optional[str] = None) -> str:
    value = (base_url or "").strip()
    if not value:
        value = (endpoint or "").strip()
    if not value:
        raise RuntimeError(
            "Missing AZURE_FOUNDRY_BASE_URL. Set it directly or set AZURE_OPENAI_ENDPOINT so /openai/v1/ can be derived."
        )

    stripped = value.rstrip("/")
    marker = "/openai/v1"
    if stripped.endswith(marker):
        return stripped + "/"
    if stripped.endswith("/openai"):
        stripped = stripped[: -len("/openai")]
    return stripped + "/openai/v1/"


def load_foundry_config() -> FoundryConfig:
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = normalize_foundry_base_url(
        os.getenv("AZURE_FOUNDRY_BASE_URL"),
        os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    extractor = os.getenv("AZURE_FOUNDRY_EXTRACTOR_DEPLOYMENT") or DEFAULT_EXTRACTOR_DEPLOYMENT
    matcher = (
        os.getenv("AZURE_FOUNDRY_MATCHER_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4")
        or DEFAULT_MATCHER_DEPLOYMENT
    )
    missing = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not extractor:
        missing.append("AZURE_FOUNDRY_EXTRACTOR_DEPLOYMENT")
    if not matcher:
        missing.append("AZURE_FOUNDRY_MATCHER_DEPLOYMENT/AZURE_OPENAI_JUDGE_DEPLOYMENT/AZURE_OPENAI_DEPLOYMENT_GPT4")
    if missing:
        raise RuntimeError("Missing required Foundry environment variables: " + ", ".join(missing))
    return FoundryConfig(
        api_key=str(api_key),
        base_url=base_url,
        extractor_deployment=str(extractor),
        matcher_deployment=str(matcher),
    )


def get_foundry_client() -> Tuple[OpenAI, FoundryConfig]:
    global _CLIENT, _CONFIG
    if _CLIENT and _CONFIG:
        return _CLIENT, _CONFIG
    _CONFIG = load_foundry_config()
    _CLIENT = OpenAI(api_key=_CONFIG.api_key, base_url=_CONFIG.base_url)
    return _CLIENT, _CONFIG


def extract_json_from_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        raise ValueError("response does not contain a JSON object")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("could not extract a complete JSON object from response")


def _usage_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return usage if isinstance(usage, dict) else {}


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (APIError, APITimeoutError, APIConnectionError, RateLimitError)):
        status_code = getattr(exc, "status_code", None)
        return status_code in {None, 408, 409, 429, 500, 502, 503, 504}
    text = str(exc).lower()
    return any(marker in text for marker in ["timeout", "timed out", "connection", "rate limit", "502", "503", "504"])


def _response_format_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text


def _max_tokens_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return "max_tokens" in text and "max_completion_tokens" in text


def _temperature_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return "temperature" in text and ("unsupported" in text or "not supported" in text)


def _create_chat_completion_with_fallbacks(
    client: OpenAI,
    deployment: str,
    kwargs: Dict[str, Any],
    use_json_mode: bool,
) -> Tuple[Any, bool]:
    try:
        if use_json_mode:
            return client.chat.completions.create(**kwargs, response_format={"type": "json_object"}), True
        return client.chat.completions.create(**kwargs), False
    except Exception as exc:
        if _max_tokens_rejected(exc) and "max_tokens" in kwargs:
            kwargs = dict(kwargs)
            max_tokens = kwargs.pop("max_tokens")
            kwargs["max_completion_tokens"] = max_tokens
            _TOKEN_PARAM_BY_DEPLOYMENT[deployment] = "max_completion_tokens"
            if use_json_mode:
                return client.chat.completions.create(**kwargs, response_format={"type": "json_object"}), True
            return client.chat.completions.create(**kwargs), False
        if _temperature_rejected(exc) and "temperature" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("temperature", None)
            _TEMPERATURE_SUPPORTED_BY_DEPLOYMENT[deployment] = False
            if use_json_mode:
                return client.chat.completions.create(**kwargs, response_format={"type": "json_object"}), True
            return client.chat.completions.create(**kwargs), False
        raise


def call_foundry_model(
    deployment: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    role_name: str,
    max_retries: int = 3,
    retry_base_seconds: float = 5,
    request_timeout: float = 120,
) -> ModelCallResult:
    client, _config = get_foundry_client()
    total_attempts = max(1, max_retries + 1)
    json_mode_supported = _JSON_MODE_SUPPORTED_BY_DEPLOYMENT.get(deployment, True)
    token_param = _TOKEN_PARAM_BY_DEPLOYMENT.get(deployment, "max_tokens")
    temperature_supported = _TEMPERATURE_SUPPORTED_BY_DEPLOYMENT.get(deployment, True)
    last_exc: Optional[Exception] = None

    for attempt in range(total_attempts):
        try:
            kwargs = {
                "model": deployment,
                "messages": messages,
                "timeout": request_timeout,
            }
            if temperature_supported:
                kwargs["temperature"] = temperature
            kwargs[token_param] = max_tokens
            try:
                response, response_format_used = _create_chat_completion_with_fallbacks(
                    client,
                    deployment,
                    kwargs,
                    json_mode_supported,
                )
            except Exception as exc:
                if json_mode_supported and _response_format_rejected(exc):
                    _JSON_MODE_SUPPORTED_BY_DEPLOYMENT[deployment] = False
                    json_mode_supported = False
                    response, response_format_used = _create_chat_completion_with_fallbacks(
                        client,
                        deployment,
                        kwargs,
                        False,
                    )
                else:
                    raise

            choice = response.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            if not raw.strip() or finish_reason == "length":
                if attempt < total_attempts - 1:
                    time.sleep(retry_base_seconds * (2**attempt))
                    continue
            try:
                parsed = extract_json_from_response(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt < total_attempts - 1 and (not raw.strip() or finish_reason == "length"):
                    time.sleep(retry_base_seconds * (2**attempt))
                    continue
                raise JudgeParseError(str(exc), raw) from exc

            return ModelCallResult(
                parsed_json=parsed,
                raw_response=raw,
                deployment=deployment,
                role_name=role_name,
                model=getattr(response, "model", None),
                finish_reason=finish_reason,
                usage=_usage_dict(response),
                created_at=datetime.now().isoformat(),
                response_format_used=response_format_used,
            )
        except JudgeParseError:
            raise
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_exception(exc) or attempt >= total_attempts - 1:
                raise
            time.sleep(retry_base_seconds * (2**attempt))

    if last_exc:
        raise last_exc
    raise RuntimeError("Foundry call failed without an exception")


def _system_json_message(role_name: str) -> Dict[str, str]:
    return {
        "role": "system",
        "content": f"You are the {role_name}. Return only valid JSON. Do not include markdown or prose outside JSON.",
    }


def normalize_extracted_items(raw: Any, id_prefix: str) -> Tuple[List[Dict[str, str]], List[str]]:
    warnings: List[str] = []
    if not isinstance(raw, list):
        warnings.append("items is not a list; treating as empty")
        return [], warnings
    normalized: List[Dict[str, str]] = []
    seen_text = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"items[{idx}] is not an object; dropped")
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if not text:
            warnings.append(f"items[{idx}] has empty text; dropped")
            continue
        duplicate_key = text.casefold()
        if duplicate_key in seen_text:
            warnings.append(f"items[{idx}] duplicates exact item text; dropped")
            continue
        seen_text.add(duplicate_key)
        normalized_item: Dict[str, str] = {"id": f"{id_prefix}{len(normalized) + 1}", "text": text}
        # preserve optional metadata fields when present
        for key in ["category", "importance", "evidence_role"]:
            if item.get(key):
                normalized_item[key] = str(item.get(key)).strip()
        normalized.append(normalized_item)
    return normalized, warnings


def normalize_excluded_items(raw: Any) -> Tuple[List[Dict[str, str]], List[str]]:
    warnings: List[str] = []
    if raw is None:
        return [], warnings
    if not isinstance(raw, list):
        warnings.append("excluded_items is not a list; treating as empty")
        return [], warnings
    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"excluded_items[{idx}] is not an object; dropped")
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        reason = str(item.get("reason") or "").strip()
        if not text:
            warnings.append(f"excluded_items[{idx}] has empty text; dropped")
            continue
        if not reason:
            warnings.append(f"excluded_items[{idx}] missing reason; set to unknown")
            reason = "unknown"
        normalized.append({"text": text, "reason": reason})
    return normalized, warnings


def normalize_extraction_result(
    call_result: ModelCallResult,
    id_prefix: str,
    item_key: str,
) -> Dict[str, Any]:
    items, warnings = normalize_extracted_items(call_result.parsed_json.get("items"), id_prefix)
    excluded_items, excluded_warnings = normalize_excluded_items(call_result.parsed_json.get("excluded_items"))
    warnings.extend(excluded_warnings)
    return {
        "extractor_status": "completed",
        item_key: items,
        "items": items,
        "excluded_items": excluded_items,
        "validation_warnings": warnings,
        "raw_extractor_response": call_result.raw_response,
        "deployment": call_result.deployment,
        "model": call_result.model,
        "finish_reason": call_result.finish_reason,
        "usage": call_result.usage,
        "created_at": call_result.created_at,
        "response_format_used": call_result.response_format_used,
    }


def failed_extraction_result(error_status: str, error: str, item_key: str, deployment: str = "") -> Dict[str, Any]:
    return {
        "extractor_status": error_status,
        item_key: [],
        "items": [],
        "excluded_items": [],
        "validation_warnings": [],
        "raw_extractor_response": "",
        "deployment": deployment,
        "error": error,
    }


def _validate_match_list(
    matches: Any,
    match_key: str,
    reference_ids: set,
    model_ids: set,
    warnings: List[str],
    blocked_reference_ids: Optional[set] = None,
    blocked_model_ids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], set, set]:
    if not isinstance(matches, list):
        warnings.append(f"{match_key} is not a list; treating as empty")
        return [], set(), set()
    blocked_reference_ids = blocked_reference_ids or set()
    blocked_model_ids = blocked_model_ids or set()
    seen_pairs = set()
    used_reference_ids = set()
    used_model_ids = set()
    valid: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        if not isinstance(match, dict):
            warnings.append(f"{match_key}[{idx}] is not an object; dropped")
            continue
        reference_id = str(match.get("reference_id") or "").strip()
        model_id = str(match.get("model_id") or "").strip()
        pair = (reference_id, model_id)
        if reference_id not in reference_ids or model_id not in model_ids:
            warnings.append(f"{match_key}[{idx}] references non-existing id(s) {reference_id}/{model_id}; dropped")
            continue
        if pair in seen_pairs:
            warnings.append(f"{match_key}[{idx}] duplicates a prior pair; dropped")
            continue
        if reference_id in blocked_reference_ids or model_id in blocked_model_ids:
            warnings.append(f"{match_key}[{idx}] reuses id already used in full match {reference_id}/{model_id}; dropped")
            continue
        if reference_id in used_reference_ids or model_id in used_model_ids:
            warnings.append(f"{match_key}[{idx}] violates one-to-one matching for {reference_id}/{model_id}; dropped")
            continue
        seen_pairs.add(pair)
        used_reference_ids.add(reference_id)
        used_model_ids.add(model_id)
        valid.append(match)
    return valid, used_reference_ids, used_model_ids


def _validate_contradictions(contradictions: Any, reference_ids: set, model_ids: set, warnings: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(contradictions, list):
        warnings.append("contradictions is not a list; treating as empty")
        return []
    valid: List[Dict[str, Any]] = []
    seen_pairs = set()
    for idx, contradiction in enumerate(contradictions):
        if not isinstance(contradiction, dict):
            warnings.append(f"contradictions[{idx}] is not an object; dropped")
            continue
        reference_id = str(contradiction.get("reference_id") or "").strip()
        model_id = str(contradiction.get("model_id") or "").strip()
        if reference_id not in reference_ids or model_id not in model_ids:
            warnings.append(f"contradictions[{idx}] references non-existing id(s) {reference_id}/{model_id}; dropped")
            continue
        pair = (reference_id, model_id)
        if pair in seen_pairs:
            warnings.append(f"contradictions[{idx}] duplicates a prior contradiction pair; dropped")
            continue
        seen_pairs.add(pair)
        valid.append(contradiction)
    return valid


def validate_matching_and_compute_metrics(
    matcher_output: Dict[str, Any],
    reference_items: List[Dict[str, str]],
    model_items: List[Dict[str, str]],
    extra_rate_key: str,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], Dict[str, Any]]:
    warnings: List[str] = []
    reference_ids = {item["id"] for item in reference_items}
    model_ids = {item["id"] for item in model_items}
    full_matches, full_reference_ids, full_model_ids = _validate_match_list(
        matcher_output.get("full_matches", []),
        "full_matches",
        reference_ids,
        model_ids,
        warnings,
    )
    partial_matches, partial_reference_ids, partial_model_ids = _validate_match_list(
        matcher_output.get("partial_matches", []),
        "partial_matches",
        reference_ids,
        model_ids,
        warnings,
        blocked_reference_ids=full_reference_ids,
        blocked_model_ids=full_model_ids,
    )
    contradictions = _validate_contradictions(matcher_output.get("contradictions", []), reference_ids, model_ids, warnings)

    tp = len(full_matches)
    fp = len(model_items) - len(full_model_ids)
    fn = len(reference_items) - len(full_reference_ids)
    strict_metrics = {
        "reference_item_count": len(reference_items),
        "model_item_count": len(model_items),
        "full_match_count": len(full_matches),
        "partial_match_count": len(partial_matches),
        "contradiction_count": len(contradictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **compute_prf(tp, fp, fn),
    }

    lenient_reference_ids = full_reference_ids | partial_reference_ids
    lenient_model_ids = full_model_ids | partial_model_ids
    lenient_tp = len(full_matches) + len(partial_matches)
    lenient_fp = len(model_items) - len(lenient_model_ids)
    lenient_fn = len(reference_items) - len(lenient_reference_ids)
    lenient_metrics = {
        "reference_item_count": len(reference_items),
        "model_item_count": len(model_items),
        "full_match_count": len(full_matches),
        "partial_match_count": len(partial_matches),
        "contradiction_count": len(contradictions),
        "tp": lenient_tp,
        "fp": lenient_fp,
        "fn": lenient_fn,
        **compute_prf(lenient_tp, lenient_fp, lenient_fn),
    }
    strict_metrics[extra_rate_key] = safe_div(fp, len(model_items))
    lenient_metrics[extra_rate_key] = safe_div(lenient_fp, len(model_items))
    strict_metrics["hallucination_rate"] = strict_metrics[extra_rate_key]
    lenient_metrics["hallucination_rate"] = lenient_metrics[extra_rate_key]

    validated = {
        "full_matches": full_matches,
        "partial_matches": partial_matches,
        "contradictions": contradictions,
    }
    return validated, warnings, strict_metrics, lenient_metrics


def evaluate_extracted_target(
    reference_text: str,
    model_text: str,
    config: EvaluationTargetConfig,
    run_id: str,
    source_result_file: str,
    case_name: str,
    extraction_max_tokens: int = 2000,
    matching_max_tokens: int = 4000,
    temperature: float = 0,
    max_retries: int = 3,
    retry_base_seconds: float = 5,
    request_timeout: float = 120,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = context or {}
    client, foundry = get_foundry_client()
    del client
    reference_text = str(reference_text or "")
    model_text = str(model_text or "")
    provenance = {
        "case_name": case_name,
        "target": config.target_name,
        "run_id": run_id,
        "source_result_file": source_result_file,
        "extractor_deployment": foundry.extractor_deployment,
        "matcher_deployment": foundry.matcher_deployment,
        "foundry_endpoint_host": foundry.endpoint_host,
        "extractor_prompt_version": config.extraction_prompt_version,
        "matcher_prompt_version": config.matching_prompt_version,
        "temperature": temperature,
        "extraction_max_tokens": extraction_max_tokens,
        "matching_max_tokens": matching_max_tokens,
        "max_retries": max_retries,
        "retry_base_seconds": retry_base_seconds,
        "request_timeout": request_timeout,
        "created_at": datetime.now().isoformat(),
    }

    if not reference_text.strip() or not model_text.strip():
        return {
            **provenance,
            "reference_text": reference_text,
            "model_text": model_text,
            "judge_status": "missing_text",
            "reference_extraction": failed_extraction_result("not_run", "missing_text", config.reference_item_key, foundry.extractor_deployment),
            "model_extraction": failed_extraction_result("not_run", "missing_text", config.model_item_key, foundry.extractor_deployment),
            "matching_result": {"matcher_status": "not_run"},
            "validation_warnings": [],
            "strict_metrics": {},
            "lenient_metrics": {},
            "error": "missing_text",
        }

    try:
        reference_prompt = config.extraction_prompt_builder(reference_text, "reference", context)
        reference_call = call_foundry_model(
            foundry.extractor_deployment,
            [_system_json_message("atomic/entity extractor"), {"role": "user", "content": reference_prompt}],
            max_tokens=extraction_max_tokens,
            temperature=temperature,
            role_name="extractor:reference",
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            request_timeout=request_timeout,
        )
        reference_extraction = normalize_extraction_result(reference_call, config.reference_id_prefix, config.reference_item_key)

        model_prompt = config.extraction_prompt_builder(model_text, "model", context)
        model_call = call_foundry_model(
            foundry.extractor_deployment,
            [_system_json_message("atomic/entity extractor"), {"role": "user", "content": model_prompt}],
            max_tokens=extraction_max_tokens,
            temperature=temperature,
            role_name="extractor:model",
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            request_timeout=request_timeout,
        )
        model_extraction = normalize_extraction_result(model_call, config.model_id_prefix, config.model_item_key)
    except JudgeParseError as exc:
        return {
            **provenance,
            "reference_text": reference_text,
            "model_text": model_text,
            "judge_status": "extractor_parse_failed",
            "reference_extraction": failed_extraction_result("extractor_parse_failed", str(exc), config.reference_item_key, foundry.extractor_deployment),
            "model_extraction": failed_extraction_result("extractor_parse_failed", str(exc), config.model_item_key, foundry.extractor_deployment),
            "matching_result": {"matcher_status": "not_run"},
            "validation_warnings": [],
            "strict_metrics": {},
            "lenient_metrics": {},
            "raw_extractor_response": exc.raw_response,
            "error": f"extractor_parse_failed: {exc}",
        }
    except Exception as exc:
        return {
            **provenance,
            "reference_text": reference_text,
            "model_text": model_text,
            "judge_status": "extractor_api_failed",
            "reference_extraction": failed_extraction_result("extractor_api_failed", str(exc), config.reference_item_key, foundry.extractor_deployment),
            "model_extraction": failed_extraction_result("extractor_api_failed", str(exc), config.model_item_key, foundry.extractor_deployment),
            "matching_result": {"matcher_status": "not_run"},
            "validation_warnings": [],
            "strict_metrics": {},
            "lenient_metrics": {},
            "error": f"extractor_api_failed: {exc}",
        }

    reference_items = reference_extraction[config.reference_item_key]
    model_items = model_extraction[config.model_item_key]

    try:
        match_prompt = config.matching_prompt_builder(reference_items, model_items, context)
        matcher_call = call_foundry_model(
            foundry.matcher_deployment,
            [_system_json_message("semantic matcher"), {"role": "user", "content": match_prompt}],
            max_tokens=matching_max_tokens,
            temperature=temperature,
            role_name="matcher",
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            request_timeout=request_timeout,
        )
    except JudgeParseError as exc:
        return {
            **provenance,
            "reference_text": reference_text,
            "model_text": model_text,
            "reference_extraction": reference_extraction,
            "model_extraction": model_extraction,
            "matching_result": {"matcher_status": "matcher_parse_failed", "raw_matcher_response": exc.raw_response},
            "validation_warnings": reference_extraction["validation_warnings"] + model_extraction["validation_warnings"],
            "strict_metrics": {},
            "lenient_metrics": {},
            "judge_status": "matcher_parse_failed",
            "error": f"matcher_parse_failed: {exc}",
        }
    except Exception as exc:
        return {
            **provenance,
            "reference_text": reference_text,
            "model_text": model_text,
            "reference_extraction": reference_extraction,
            "model_extraction": model_extraction,
            "matching_result": {"matcher_status": "matcher_api_failed"},
            "validation_warnings": reference_extraction["validation_warnings"] + model_extraction["validation_warnings"],
            "strict_metrics": {},
            "lenient_metrics": {},
            "judge_status": "matcher_api_failed",
            "error": f"matcher_api_failed: {exc}",
        }

    validated_matches, match_warnings, strict_metrics, lenient_metrics = validate_matching_and_compute_metrics(
        matcher_call.parsed_json,
        reference_items,
        model_items,
        config.extra_rate_key,
    )
    if config.target_name == "final_diagnosis":
        strict_metrics["exact_match"] = (
            strict_metrics["fp"] == 0
            and strict_metrics["fn"] == 0
            and strict_metrics["tp"] == len(reference_items)
            and len(reference_items) == len(model_items)
        )
    else:
        strict_metrics["coverage"] = strict_metrics["recall"]
        lenient_metrics["coverage"] = lenient_metrics["recall"]

    matching_result = {
        "matcher_status": "completed",
        **validated_matches,
        "raw_matcher_response": matcher_call.raw_response,
        "deployment": matcher_call.deployment,
        "model": matcher_call.model,
        "finish_reason": matcher_call.finish_reason,
        "usage": matcher_call.usage,
        "created_at": matcher_call.created_at,
        "response_format_used": matcher_call.response_format_used,
    }
    return {
        **provenance,
        "reference_text": reference_text,
        "model_text": model_text,
        "reference_extraction": reference_extraction,
        "model_extraction": model_extraction,
        "matching_result": matching_result,
        "validation_warnings": reference_extraction["validation_warnings"] + model_extraction["validation_warnings"] + match_warnings,
        "strict_metrics": strict_metrics,
        "lenient_metrics": lenient_metrics,
        "judge_status": "completed",
        "error": "",
    }


def _json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def diagnosis_extraction_prompt(text: str, side: str, context: Dict[str, Any]) -> str:
    return f"""
Extract separately asserted diagnostic entities from this {side} final diagnosis only.

This is an extraction step. Do not perform matching. Do not calculate TP, FP, FN, precision, recall, or F1.
Extract diseases, syndromes, etiologies, causal organisms, disease subtypes, named complications, and causal diagnoses asserted as final diagnoses.

Preserve clinically essential modifiers when they are part of the diagnosis, including: etiology, causal organism, anatomical site, disease subtype, named complication, severity when clinically diagnostic, and causal relationships such as "caused by", "secondary to", or "associated with".
Do not reduce a specific diagnosis to only a broad disease category. Do not extract ordinary symptoms, signs, measurements, routine lab abnormalities, image appearances, severity adjectives alone, or anatomical locations alone.
Do not invent synonyms or inferred diagnoses.

Example: if the final diagnosis is "fatal mucormycotic osteomyelitis caused by Rhizopus microsporus", do not extract only "osteomyelitis". Keep the fungal etiology and organism when they are part of the diagnosis.

Return JSON only:
{{"items": [{{"text": "mucormycotic osteomyelitis caused by Rhizopus microsporus"}}]}}

Final diagnosis text:
{text}
"""


def reasoning_extraction_prompt(text: str, side: str, context: Dict[str, Any]) -> str:
        # Side-specific behavior: reference -> gold checklist builder; model -> model extractor with relevance filter
        if side == "reference":
                ground_truth_final_diagnosis = context.get("ground_truth_final_diagnosis", "")
                reference_image_findings = context.get("reference_image_findings", [])
                reference_image_findings_text = json.dumps(reference_image_findings, ensure_ascii=False, indent=2)
                return f"""
You are building a gold diagnostic evidence checklist for evaluating clinical diagnostic reasoning.

This is an extraction and filtering step, not a matching step.
Do not perform matching. Do not calculate TP, FP, FN, precision, recall, or F1.
Do not add external medical knowledge beyond the provided reference information.

Ground-truth final diagnosis:
{ground_truth_final_diagnosis or "N/A"}

Reference image findings, if available:
{reference_image_findings_text}

Task:
Extract clinically meaningful diagnostic evidence units from the reference diagnostic reasoning text and the provided reference evidence.

A diagnostic evidence unit is not simply any atomic medical fact. It should be a clinically meaningful step that supports, excludes, differentiates, localizes, characterizes, or explains the ground-truth diagnosis.

Extract:
- key clinical findings used diagnostically;
- key imaging findings used diagnostically;
- pathology, histology, cytology, immunohistochemistry, genetic, microbiology, molecular, serology, or laboratory findings used diagnostically;
- rule-in or rule-out evidence;
- causal, etiologic, organism-level, subtype-level, or anatomical-site evidence;
- key differential diagnosis evidence when stated;
- evidence needed to distinguish the ground-truth diagnosis from plausible alternatives;
- synthesis statements only when they connect concrete evidence to the diagnosis.

Exclude:
- background history not used diagnostically;
- treatment, management, follow-up, or outcome unless directly used as diagnostic evidence;
- routine descriptive details without diagnostic role;
- repeated diagnosis statements without evidence;
- generic medical knowledge;
- overly granular adjectives that do not independently contribute to diagnosis;
- redundant restatements;
- vague filler statements such as “taken together” without concrete evidence.

Granularity:
- Prefer one clinically meaningful evidence unit per diagnostic evidence source.
- Keep related findings from the same test together when they support the same diagnostic point.
- Prefer fewer high-signal evidence units over many low-level atomic facts.
- For most cases, aim for about 3-10 diagnostic evidence units unless the reference truly contains more distinct diagnostic evidence.
- Mark evidence as "essential" if missing it would substantially weaken the ability to reach the ground-truth diagnosis.
- Mark evidence as "supporting" if it helps but is not central.

Return JSON only:
{{
    "items": [
        {{
            "text": "Immunohistochemistry showed S100 positivity and absence of alpha-SMA, CD34, and CD117 staining, supporting benign schwannoma.",
            "category": "immunohistochemistry_evidence",
            "importance": "essential",
            "evidence_role": "supports_diagnosis"
        }}
    ],
    "excluded_items": [
        {{
            "text": "The patient received treatment.",
            "reason": "treatment_not_diagnostic"
        }}
    ]
}}

Reference diagnostic reasoning text:
{text}
"""
        else:
                # model side
                ground_truth_final_diagnosis = context.get("ground_truth_final_diagnosis", "")
                model_final_diagnosis = context.get("model_final_diagnosis", "")
                return f"""
You are extracting diagnostically relevant evidence units from a model-generated diagnostic reasoning trace.

This is an extraction and filtering step, not a matching step.
Do not perform matching. Do not calculate TP, FP, FN, precision, recall, or F1.
Do not add external medical knowledge.

Ground-truth final diagnosis, for relevance filtering only:
{ground_truth_final_diagnosis or "N/A"}

Model final diagnosis, if available:
{model_final_diagnosis or "N/A"}

Use the ground-truth final diagnosis only to understand the diagnostic question being evaluated and to filter out clearly unrelated background information.
Do NOT use the ground truth to delete model errors.

Very important:
For model-side extraction, keep incorrect but diagnostic claims if the model uses them to support, exclude, differentiate, localize, characterize, or justify a diagnosis, even if they appear medically wrong or do not match the ground truth.
Do not remove a model claim simply because it disagrees with the ground-truth diagnosis.

Extract:
- clinical findings used diagnostically;
- imaging findings used diagnostically;
- pathology, histology, cytology, immunohistochemistry, genetic, microbiology, molecular, serology, or laboratory findings used diagnostically;
- rule-in or rule-out reasoning;
- differential diagnosis reasoning;
- causal, etiologic, organism-level, subtype-level, or anatomical-site reasoning;
- synthesis claims that connect evidence to a diagnosis.

Exclude:
- background history not used diagnostically;
- treatment, management, follow-up, or outcome not used diagnostically;
- generic medical knowledge;
- vague filler statements;
- repeated final diagnosis statements without evidence;
- ordinary symptoms or labs not connected to diagnosis;
- overly granular adjectives that do not independently contribute to diagnosis;
- redundant restatements.

Granularity:
- Prefer clinically meaningful diagnostic evidence units over many small atomic facts.
- Keep related findings from the same test together when they support the same diagnostic point.
- If a sentence contains both diagnostic and non-diagnostic details, keep only the diagnostic evidence.
- For most cases, aim for about 3-10 diagnostic evidence units unless the model reasoning truly contains more distinct diagnostic evidence.

Return JSON only:
{{
    "items": [
        {{
            "text": "The model interpreted the spindle-cell lesion as compatible with smooth muscle differentiation, supporting leiomyoma.",
            "category": "pathology_evidence",
            "importance": "essential",
            "evidence_role": "supports_diagnosis"
        }}
    ],
    "excluded_items": [
        {{
            "text": "The patient was hospitalized.",
            "reason": "background_not_diagnostic"
        }}
    ]
}}

Model diagnostic reasoning text:
{text}
"""


def image_extraction_prompt(text: str, side: str, context: Dict[str, Any]) -> str:
    image_id = context.get("image_id", "")
    image_type = context.get("image_type", "")
    return f"""Extract visible-image claims from this {side} image finding text only.

You may use this context only to identify the image: image_id={image_id}; image_type={image_type}.
Do not use the actual image file. Do not use final diagnosis, diagnostic reasoning, clinical history, treatment, or laboratory values.
Do not perform matching. Do not calculate metrics.

Extract visible claims including anatomy, lesion morphology, size, shape, distribution, density, signal, enhancement, color, mass effect, edema, hemorrhage, opacity, ulceration, erosion, necrosis, calcification, lymphadenopathy, staining pattern, cellular morphology, nuclear pattern, and tissue architecture.

If the text includes diagnostic impressions or disease labels such as "compatible with leiomyoma", "suggestive of schwannoma", or "diagnostic of osteomyelitis", do not include the disease label as a visible-image claim. Do not remove the underlying visible finding: for example, from "spindle-cell lesion compatible with leiomyoma," extract "spindle-cell lesion" but do not extract "compatible with leiomyoma." Exclude generic statements such as "supports the diagnosis" and nonvisual diagnostic reasoning.

Return JSON only:
{{"items": [{{"text": "There is stenosis of the right renal artery."}}]}}

Image finding text:
{text}
"""


def diagnosis_matching_prompt(reference_items: List[Dict[str, str]], model_items: List[Dict[str, str]], context: Dict[str, Any]) -> str:
    rules = (
        "Full match requires preservation of the clinically essential diagnostic meaning. Medical synonyms and accepted abbreviations may fully match. "
        "Partial match requires a shared diagnostic evidence core between the reference and model items. "
        "Vague overlap, same organ only, same broad disease category only, same clinical context only, generic diagnostic language, or overlap through external inference should be classified as no match rather than partial match. "
        "No match means the model item does not share the same core diagnosis/evidence/finding as the reference item, describes a different entity/finding, is too generic to verify the reference item, or overlaps only through external inference. "
        "If the model misses a critical etiology, organism, subtype, anatomical site, or named complication, classify as partial, not full. "
        "Different disease entities should not be treated as full matches. Extra unsupported diagnoses are false positives, not automatically contradictions. "
        "Contradiction requires direct incompatibility or opposition."
    )
    rules = (
        rules
        + "\nReference: \"fatal mucormycotic osteomyelitis caused by Rhizopus microsporus\"\nModel: \"pelvic osteomyelitis complicated by gluteal and thigh abscesses\"\nExpected: partial / broader (misses fungal etiology and organism).\n"
        + "Reference: \"benign schwannoma\"\nModel: \"esophageal leiomyoma\"\nExpected: not full match (different diagnostic entities)."
    )
    return _matching_prompt(
        "diagnosis entities",
        "reference_diagnoses",
        "model_diagnoses",
        reference_items,
        model_items,
        rules,
    )


def reasoning_matching_prompt(reference_items: List[Dict[str, str]], model_items: List[Dict[str, str]], context: Dict[str, Any]) -> str:
    rules = (
        "Match clinically meaningful diagnostic evidence units. Do not split, merge, rewrite, add, or delete evidence units. Use one-to-one matching. "
        "Do not match by list order. Full match means the model evidence unit captures the same clinically essential diagnostic meaning as the reference evidence unit. "
        "Partial match requires a shared diagnostic evidence core. Vague overlap, same organ only, same broad disease category only, same clinical context only, generic diagnostic language, or overlap through external inference should be classified as no match, not partial. "
        "No match means the model item does not share the same core diagnosis/evidence/finding as the reference item, describes a different entity/finding, is too generic to verify the reference item, or overlaps only through external inference. "
        "A model claim that supports a wrong diagnosis should not be matched as full unless the evidence itself is equivalent to the reference evidence. "
        "Contradiction requires direct incompatibility with a reference evidence unit. Extra diagnostic claims that do not match the gold checklist are false positives, not automatically contradictions."
    )
    return _matching_prompt(
        "clinically meaningful diagnostic evidence units",
        "reference_claims",
        "model_claims",
        reference_items,
        model_items,
        rules,
    )


def image_matching_prompt(reference_items: List[Dict[str, str]], model_items: List[Dict[str, str]], context: Dict[str, Any]) -> str:
    rules = (
        "Full match requires the same visible abnormality and preservation of clinically important visual attributes when present, including anatomy, location, laterality, morphology, size, count, distribution, density/signal/intensity, enhancement, staining pattern, severity, extent, and mass effect. "
        "Partial match requires the same core visible abnormality but with missing, broader, narrower, or imprecise visual attributes. "
        "No match should be used when the model gives only a diagnosis label, generic image language, a different visual abnormality, or broad anatomic overlap without the same visible finding. "
        "Contradiction includes wrong presence/absence, wrong laterality/site, wrong morphology, wrong staining/marker pattern, wrong signal/density/enhancement pattern, or incompatible severity/extent. "
        "For pathology images, also consider tissue type, stain type, cellular morphology, architectural pattern, nuclear features, necrosis, inflammation, invasion pattern, organism appearance, and immunohistochemical or special-stain positivity/negativity when visually described. "
        "Do not split, merge, rewrite, add, or delete visual claims. Extra absent model visual claims are false positives, not automatically contradictions."
    )
    return _matching_prompt(
        "atomic visible-image claims",
        "reference_visual_claims",
        "model_visual_claims",
        reference_items,
        model_items,
        rules,
    )


def _matching_prompt(
    item_description: str,
    reference_key: str,
    model_key: str,
    reference_items: List[Dict[str, str]],
    model_items: List[Dict[str, str]],
    rules: str,
) -> str:
    payload = {reference_key: reference_items, model_key: model_items}
    return f"""You are matching extracted {item_description}. You receive extracted items only, not raw source paragraphs.

Rules:
- Do not decompose items.
- Do not calculate TP, FP, FN, precision, recall, or F1.
- Do not match by list order.
- Use one-to-one matching.
- Each reference_id can appear at most once across full_matches and partial_matches.
- Each model_id can appear at most once across full_matches and partial_matches.
- {rules}

Input items:
{_json_block(payload)}

Return JSON only:
{{
  "full_matches": [
    {{"reference_id": "R1", "model_id": "M1", "match_type": "equivalent", "explanation": "..."}}
  ],
  "partial_matches": [
    {{"reference_id": "R2", "model_id": "M2", "match_type": "partial | broader | narrower", "explanation": "..."}}
  ],
  "contradictions": [
    {{"reference_id": "R3", "model_id": "M3", "explanation": "..."}}
  ]
}}"""


TARGET_CONFIGS: Dict[str, EvaluationTargetConfig] = {
    "final_diagnosis": EvaluationTargetConfig(
        target_name="final_diagnosis",
        reference_item_key="reference_diagnoses",
        model_item_key="model_diagnoses",
        reference_id_prefix="R",
        model_id_prefix="M",
        extraction_prompt_version=DIAGNOSIS_EXTRACTION_PROMPT_VERSION,
        matching_prompt_version=DIAGNOSIS_MATCHING_PROMPT_VERSION,
        extraction_prompt_builder=diagnosis_extraction_prompt,
        matching_prompt_builder=diagnosis_matching_prompt,
        extra_rate_key="extra_entity_rate",
        item_label="diagnosis",
    ),
    "diagnostic_reasoning": EvaluationTargetConfig(
        target_name="diagnostic_reasoning",
        reference_item_key="reference_claims",
        model_item_key="model_claims",
        reference_id_prefix="R",
        model_id_prefix="M",
        extraction_prompt_version=REASONING_EXTRACTION_PROMPT_VERSION,
        matching_prompt_version=REASONING_MATCHING_PROMPT_VERSION,
        extraction_prompt_builder=reasoning_extraction_prompt,
        matching_prompt_builder=reasoning_matching_prompt,
        extra_rate_key="extra_claim_rate",
        item_label="reasoning",
    ),
    "image_finding": EvaluationTargetConfig(
        target_name="image_finding",
        reference_item_key="reference_visual_claims",
        model_item_key="model_visual_claims",
        reference_id_prefix="R",
        model_id_prefix="M",
        extraction_prompt_version=IMAGE_EXTRACTION_PROMPT_VERSION,
        matching_prompt_version=IMAGE_MATCHING_PROMPT_VERSION,
        extraction_prompt_builder=image_extraction_prompt,
        matching_prompt_builder=image_matching_prompt,
        extra_rate_key="extra_visual_claim_rate",
        item_label="image",
    ),
}


def aggregate_metric_group(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    if not rows:
        return {
            "num_items": 0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_tp": 0,
            "micro_fp": 0,
            "micro_fn": 0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }
    tp = sum(int(row.get(f"{prefix}_tp") or 0) for row in rows)
    fp = sum(int(row.get(f"{prefix}_fp") or 0) for row in rows)
    fn = sum(int(row.get(f"{prefix}_fn") or 0) for row in rows)
    micro = compute_prf(tp, fp, fn)
    return {
        "num_items": len(rows),
        "macro_precision": safe_div(sum(float(row.get(f"{prefix}_precision") or 0.0) for row in rows), len(rows)),
        "macro_recall": safe_div(sum(float(row.get(f"{prefix}_recall") or 0.0) for row in rows), len(rows)),
        "macro_f1": safe_div(sum(float(row.get(f"{prefix}_f1") or 0.0) for row in rows), len(rows)),
        "micro_tp": tp,
        "micro_fp": fp,
        "micro_fn": fn,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
    }


def diff_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def iter_case_files(results_dir: Path) -> List[Path]:
    return sorted(path for path in results_dir.glob("*.json") if not path.name.startswith("_"))
