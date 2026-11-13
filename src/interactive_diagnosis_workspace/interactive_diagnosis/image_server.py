from __future__ import annotations

from typing import Any

from .evidence_matching import best_matches, public_evidence_view


class ImageServer:
    """Serves raw patient image pixels only when explicitly requested."""

    def __init__(self, case_data: dict[str, Any]) -> None:
        self.case_data = case_data

    def review_image(
        self,
        *,
        request: str,
        current_phase_index: int,
        disclosed_ids: set[str],
    ) -> dict[str, Any]:
        matches = best_matches(
            self.case_data,
            "REVIEW_IMAGE",
            request,
            current_phase_index=current_phase_index,
            include_future=False,
            max_items=2,
        )
        new_items = [item for item in matches if str(item.get("evidence_id")) not in disclosed_ids]
        for item in new_items:
            disclosed_ids.add(str(item.get("evidence_id")))
        image_paths: list[str] = []
        for item in new_items:
            image_paths.extend(item.get("image_paths") or [])
        if matches and not new_items:
            status = "already_disclosed"
            message = "The matching image has already been disclosed."
        elif new_items:
            status = "answered"
            message = "Source-grounded raw image content matching the request is attached."
        else:
            status = "no_result_currently_available"
            message = "No source-grounded raw image matching the request is currently available in the replay state."
        return {
            "public": {
                "status": status,
                "message": message,
                "available_information": [public_evidence_view(item) for item in new_items],
            },
            "internal": {
                "matched_evidence_ids": [str(item.get("evidence_id")) for item in matches],
                "disclosed_evidence_ids": [str(item.get("evidence_id")) for item in new_items],
                "current_phase_index": current_phase_index,
            },
            "image_paths": list(dict.fromkeys(image_paths)),
        }
