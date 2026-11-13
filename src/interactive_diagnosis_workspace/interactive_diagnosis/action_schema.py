from __future__ import annotations

from typing import Any
import re


NON_FINAL_ACTIONS = {
    "ASK_HISTORY",
    "REQUEST_EXAM",
    "ORDER_TEST",
    "REQUEST_RESULT",
    "ORDER_PROCEDURE",
    "REVIEW_IMAGE",
    "WAIT_REASSESS",
}
CLINICIAN_ACTIONS = NON_FINAL_ACTIONS | {"FINAL"}


RAW_IMAGE_REQUEST_RE = re.compile(
    r"\b("
    r"raw\s+image|raw\s+pixel|pixels?|"
    r"review\s+(?:the\s+)?(?:image|images|figure|figures|scan|slides?)|"
    r"look\s+at\s+(?:the\s+)?(?:image|images|figure|figures|scan|slides?)|"
    r"(?:ct|mri|pet|ultrasound|ceus|x[- ]?ray|radiograph|pathology|histology|microscopy|endoscopy)"
    r"\s+(?:image|images|figure|figures|pixel|pixels|slide|slides)|"
    r"image\s+pixels?|figure\s+\d*|figures?|pictures?|photographs?|photos?|"
    r"pathology\s+slide|histology\s+slide|microscopy\s+slide"
    r")\b",
    re.IGNORECASE,
)


def requests_raw_image(request_text: str) -> bool:
    text = request_text.lower()
    if not RAW_IMAGE_REQUEST_RE.search(text):
        return False
    # Phrases such as "imaging report" or "image findings in the report" should
    # remain report/result requests unless the clinician explicitly asks to view
    # pixels, figures, photos, slides, or images directly.
    if re.search(r"\b(imaging|image)\s+report\b", text) and not re.search(
        r"\b(raw|pixel|figure|picture|photo|slide|directly|view|review|look at)\b",
        text,
    ):
        return False
    return True


def normalize_action(parsed: dict[str, Any]) -> dict[str, Any]:
    action = str(parsed.get("action") or "").strip().upper()
    if action not in CLINICIAN_ACTIONS:
        raise ValueError(f"clinician action must be one of: {', '.join(sorted(CLINICIAN_ACTIONS))}")
    parsed["action"] = action
    if action == "FINAL":
        ddx = parsed.get("working_differential")
        parsed["working_differential"] = ddx if isinstance(ddx, list) else []
        return parsed

    request_text = str(parsed.get("request") or "").strip()
    if not request_text and action != "WAIT_REASSESS":
        raise ValueError(f"{action} action must contain a non-empty request")
    parsed["request"] = request_text or "Wait and reassess the next source-grounded clinical event."
    if action in {"ORDER_TEST", "ORDER_PROCEDURE", "REQUEST_RESULT"} and requests_raw_image(parsed["request"]):
        parsed["action"] = "REVIEW_IMAGE"
    ddx = parsed.get("working_differential")
    parsed["working_differential"] = ddx if isinstance(ddx, list) else []
    parsed.setdefault("rationale", "")
    return parsed
