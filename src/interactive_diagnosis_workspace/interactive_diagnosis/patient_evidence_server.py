from __future__ import annotations

from typing import Any

from .evidence_matching import best_matches, public_evidence_view


class PatientEvidenceServer:
    """Serves currently available patient history and bedside examination facts."""

    def __init__(self, case_data: dict[str, Any]) -> None:
        self.case_data = case_data

    def answer(
        self,
        action: str,
        request: str,
        current_phase_index: int,
        disclosed_ids: set[str],
    ) -> dict[str, Any]:
        matches = best_matches(
            self.case_data,
            action,
            request,
            current_phase_index=current_phase_index,
            include_future=False,
            max_items=3,
        )
        new_items = [item for item in matches if str(item.get("evidence_id")) not in disclosed_ids]
        for item in new_items:
            disclosed_ids.add(str(item.get("evidence_id")))
        if matches and not new_items:
            status = "already_disclosed"
            message = "The matching patient information has already been disclosed."
        elif new_items:
            status = "answered"
            message = "Source-grounded patient information matching the request is provided."
        else:
            status = "not_reported"
            message = "The requested patient information is not reported in the currently accessible case record. This is not a negative finding."
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
            "image_paths": [],
        }
