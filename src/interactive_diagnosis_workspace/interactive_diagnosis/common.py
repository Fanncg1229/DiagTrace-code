from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import AsyncAzureOpenAI


load_dotenv(find_dotenv(usecwd=True))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if not value:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("model response JSON is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.I | re.S)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = value.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(value[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("extracted JSON is not an object")
                return parsed
    raise ValueError("incomplete JSON object in model response")


def azure_client(timeout: int = 600) -> AsyncAzureOpenAI:
    required = {
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return AsyncAzureOpenAI(
        api_key=required["AZURE_OPENAI_API_KEY"],
        azure_endpoint=required["AZURE_OPENAI_ENDPOINT"],
        api_version=required["AZURE_OPENAI_API_VERSION"],
        timeout=timeout,
    )


def default_model() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT") or "o4-mini"


def token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def add_token_usage(total: dict[str, int], addition: dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int(addition.get(key, 0) or 0)


def empty_token_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def encode_image(path: Path) -> str | None:
    try:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"
    except OSError:
        return None

