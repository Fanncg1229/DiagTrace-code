from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence_matching import action_allowed_for_item, is_postdiagnostic_item, public_evidence_view, source_items
from .image_server import ImageServer
from .measurement_evidence_server import MeasurementEvidenceServer
from .patient_evidence_server import PatientEvidenceServer


@dataclass
class ReplayState:
    current_phase_index: int = 0
    disclosed_ids: set[str] = field(default_factory=set)
    orders: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_phase_index": self.current_phase_index,
            "disclosed_evidence_ids": sorted(self.disclosed_ids),
            "orders": self.orders,
        }


class DeterministicStateController:
    """Owns replay state, event cursor, order lifecycle, and public sanitization."""

    def __init__(self, case_data: dict[str, Any]) -> None:
        self.case_data = case_data
        self.state = ReplayState()
        self.patient_server = PatientEvidenceServer(case_data)
        self.measurement_server = MeasurementEvidenceServer(case_data)
        self.image_server = ImageServer(case_data)
        self.max_diagnostic_phase = max(
            [0]
            + [
                int(item.get("phase_index", 0) or 0)
                for item in source_items(case_data)
                if not is_postdiagnostic_item(item)
            ]
        )

    def handle(self, action: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action") or "")
        request = str(action.get("request") or "")
        before = self.state.snapshot()
        if action_type == "ASK_HISTORY":
            bundle = self.patient_server.answer(
                action_type, request, self.state.current_phase_index, self.state.disclosed_ids
            )
        elif action_type == "REQUEST_EXAM":
            bundle = self.patient_server.answer(
                action_type, request, self.state.current_phase_index, self.state.disclosed_ids
            )
        elif action_type in {"ORDER_TEST", "ORDER_PROCEDURE"}:
            bundle = self.measurement_server.create_order(
                action=action_type,
                request=request,
                current_phase_index=self.state.current_phase_index,
                orders=self.state.orders,
            )
        elif action_type == "REQUEST_RESULT":
            bundle = self.measurement_server.request_result(
                request=request,
                current_phase_index=self.state.current_phase_index,
                orders=self.state.orders,
                disclosed_ids=self.state.disclosed_ids,
            )
        elif action_type == "REVIEW_IMAGE":
            bundle = self.image_server.review_image(
                request=request,
                current_phase_index=self.state.current_phase_index,
                disclosed_ids=self.state.disclosed_ids,
            )
        elif action_type == "WAIT_REASSESS":
            bundle = self.wait_reassess()
        else:
            bundle = {
                "public": {
                    "status": "invalid_action",
                    "message": "The submitted action is not supported by the replay controller.",
                    "available_information": [],
                    "pending_orders": self.measurement_server.public_pending_orders(self.state.orders),
                },
                "internal": {"unsupported_action": action_type},
                "image_paths": [],
            }
        after = self.state.snapshot()
        public = dict(bundle.get("public") or {})
        public.setdefault("available_information", [])
        public.setdefault("pending_orders", self.measurement_server.public_pending_orders(self.state.orders))
        return {
            "public": public,
            "internal": {
                "state_before": before,
                "state_after": after,
                **(bundle.get("internal") or {}),
            },
            "image_paths": bundle.get("image_paths") or [],
        }

    def wait_reassess(self) -> dict[str, Any]:
        if self.state.current_phase_index >= self.max_diagnostic_phase:
            return {
                "public": {
                    "status": "no_new_clinical_update",
                    "message": "No additional pre-final source-grounded diagnostic event is available in the replay.",
                    "available_information": [],
                    "pending_orders": self.measurement_server.public_pending_orders(self.state.orders),
                },
                "internal": {
                    "advanced": False,
                    "reason": "at_max_diagnostic_phase",
                    "current_phase_index": self.state.current_phase_index,
                },
                "image_paths": [],
            }

        self.state.current_phase_index += 1
        current = self.state.current_phase_index
        wait_items = [
            item
            for item in source_items(self.case_data)
            if int(item.get("phase_index", 0) or 0) == current
            and action_allowed_for_item(item, "WAIT_REASSESS")
            and not is_postdiagnostic_item(item)
            and str(item.get("evidence_id")) not in self.state.disclosed_ids
        ]
        order_items = self.measurement_server.release_due_order_results(
            current_phase_index=current,
            orders=self.state.orders,
            disclosed_ids=self.state.disclosed_ids,
        )
        new_items = list({str(item.get("evidence_id")): item for item in wait_items + order_items}.values())
        for item in new_items:
            self.state.disclosed_ids.add(str(item.get("evidence_id")))
        if new_items:
            has_partial_order = any(order.get("status") == "partial" for order in self.state.orders)
            status = "answered_partial" if has_partial_order else "answered"
            message = (
                "The replay advanced by one source-grounded diagnostic event; a partial ordered result is provided and some ordered components remain pending."
                if has_partial_order
                else "The replay advanced by one source-grounded diagnostic event; newly available information is provided."
            )
        else:
            status = "no_new_clinical_update"
            message = "The replay advanced by one source-grounded diagnostic event, but no new source-grounded information is currently available."
        return {
            "public": {
                "status": status,
                "message": message,
                "available_information": [public_evidence_view(item) for item in new_items],
                "pending_orders": self.measurement_server.public_pending_orders(self.state.orders),
            },
            "internal": {
                "advanced": True,
                "current_phase_index": current,
                "released_evidence_ids": [str(item.get("evidence_id")) for item in new_items],
            },
            "image_paths": [],
        }
