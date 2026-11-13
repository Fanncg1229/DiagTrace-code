from __future__ import annotations

from typing import Any

from .evidence_matching import (
    PROCEDURE_TREATMENT_RE,
    best_matches,
    filter_candidates,
    is_postdiagnostic_item,
    public_evidence_view,
    split_compound_request,
    tokens,
)


def _overlap_score(a: str, b: str) -> int:
    return len(tokens(_normalize_order_text(a)).intersection(tokens(_normalize_order_text(b))))


def _normalize_order_text(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "complete blood count": "cbc complete blood count",
        "alpha-fetoprotein": "afp alpha fetoprotein",
        "alpha fetoprotein": "afp alpha fetoprotein",
        "peripheral blood smear": "smear peripheral blood smear",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


ORDER_MATCH_KEY_TOKENS = {
    "afp",
    "alpha",
    "fetoprotein",
    "cbc",
    "mri",
    "ct",
    "pet",
    "ceus",
    "ultrasound",
    "biopsy",
    "marrow",
    "uba1",
    "mutation",
    "histology",
    "pathology",
    "culture",
    "serology",
    "antibody",
    "cryoglobulin",
}


def _same_order_target(a: str, b: str) -> bool:
    shared = tokens(_normalize_order_text(a)).intersection(tokens(_normalize_order_text(b)))
    return len(shared) >= 2 or bool(shared.intersection(ORDER_MATCH_KEY_TOKENS))


class MeasurementEvidenceServer:
    """Handles diagnostic orders and result retrieval without advancing time."""

    def __init__(self, case_data: dict[str, Any]) -> None:
        self.case_data = case_data

    def create_order(
        self,
        *,
        action: str,
        request: str,
        current_phase_index: int,
        orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if action == "ORDER_PROCEDURE" and PROCEDURE_TREATMENT_RE.search(request):
            return {
                "public": {
                    "status": "invalid_action",
                    "message": "The replay does not allow clinician-controlled treatment, surgery, or other therapeutic intervention. No procedure was performed.",
                    "available_information": [],
                    "pending_orders": self.public_pending_orders(orders),
                },
                "internal": {
                    "rejected_order": request,
                    "reason": "therapeutic_or_original_clinician_intervention",
                    "current_phase_index": current_phase_index,
                },
                "image_paths": [],
            }

        split_components = split_compound_request(request)
        components = list(dict.fromkeys(split_components if len(split_components) > 1 else [request]))
        order_id = f"O{len(orders) + 1:03d}"
        order_components: list[dict[str, Any]] = []
        all_matched_ids: list[str] = []
        for component in components:
            matches = best_matches(
                self.case_data,
                action,
                component,
                current_phase_index=current_phase_index,
                include_future=True,
                max_items=2,
            )
            matched_ids = [str(item.get("evidence_id")) for item in matches]
            all_matched_ids.extend(matched_ids)
            order_components.append(
                {
                    "component": component,
                    "matched_evidence_ids": matched_ids,
                    "available_phase_indices": sorted(
                        {int(item.get("phase_index", 0) or 0) for item in matches}
                    ),
                    "status": "pending",
                    "released_evidence_ids": [],
                }
            )
        all_matched_ids = list(dict.fromkeys(all_matched_ids))
        order = {
            "order_id": order_id,
            "action": action,
            "request": request,
            "components": order_components,
            "matched_evidence_ids": all_matched_ids,
            "placed_at_phase_index": current_phase_index,
            "status": "pending",
        }
        orders.append(order)
        return {
            "public": {
                "status": "ordered",
                "message": "Order recorded. No result is released by the order action alone.",
                "available_information": [],
                "pending_orders": self.public_pending_orders(orders),
            },
            "internal": {
                "order": order,
                "current_phase_index": current_phase_index,
            },
            "image_paths": [],
        }

    def request_result(
        self,
        *,
        request: str,
        current_phase_index: int,
        orders: list[dict[str, Any]],
        disclosed_ids: set[str],
    ) -> dict[str, Any]:
        matching_orders = self.matching_orders(request, orders)
        if matching_orders:
            items = self.available_order_items(
                matching_orders,
                current_phase_index,
                disclosed_ids,
                request=request,
            )
            return self._result_response(
                request=request,
                current_phase_index=current_phase_index,
                items=items,
                disclosed_ids=disclosed_ids,
                no_result_status="no_result_currently_available",
                no_result_message="A matching order is pending, but no source-grounded result is currently available in the replay state.",
                orders=orders,
                internal_extra={"matched_order_ids": [order["order_id"] for order in matching_orders]},
            )

        current_matches = best_matches(
            self.case_data,
            "REQUEST_RESULT",
            request,
            current_phase_index=current_phase_index,
            include_future=False,
            max_items=3,
        )
        if current_matches:
            return self._result_response(
                request=request,
                current_phase_index=current_phase_index,
                items=current_matches,
                disclosed_ids=disclosed_ids,
                no_result_status="already_disclosed",
                no_result_message="The matching result has already been disclosed.",
                orders=orders,
            )

        future_matches = filter_candidates(
            self.case_data,
            "REQUEST_RESULT",
            request,
            current_phase_index=current_phase_index,
            include_future=True,
        )
        status = "no_result_currently_available" if future_matches else "not_reported"
        message = (
            "No source-grounded result is currently available in the replay state."
            if future_matches
            else "The requested result is not reported in the currently accessible case record. This is not a negative result."
        )
        return {
            "public": {
                "status": status,
                "message": message,
                "available_information": [],
                "pending_orders": self.public_pending_orders(orders),
            },
            "internal": {
                "future_match_count": len(future_matches),
                "current_phase_index": current_phase_index,
            },
            "image_paths": [],
        }

    def release_due_order_results(
        self,
        *,
        current_phase_index: int,
        orders: list[dict[str, Any]],
        disclosed_ids: set[str],
    ) -> list[dict[str, Any]]:
        due: list[dict[str, Any]] = []
        by_id = self.items_by_id()
        for order in orders:
            if order.get("status") == "completed":
                continue
            component = self.next_releasable_component(order, by_id, disclosed_ids)
            available = self.releasable_items_for_component(component, by_id, disclosed_ids) if component else []
            if available:
                due.extend(available)
                self.mark_component_released(order, component, available)
        return list({str(item.get("evidence_id")): item for item in due}.values())

    def available_order_items(
        self,
        orders: list[dict[str, Any]],
        current_phase_index: int,
        disclosed_ids: set[str],
        request: str = "",
    ) -> list[dict[str, Any]]:
        by_id = self.items_by_id()
        items: list[dict[str, Any]] = []
        for order in orders:
            component = self.best_component_for_request(order, request, by_id, disclosed_ids)
            component_items = self.releasable_items_for_component(component, by_id, disclosed_ids) if component else []
            if component_items:
                items.extend(component_items)
                self.mark_component_released(order, component, component_items)
                break
        return list({str(item.get("evidence_id")): item for item in items}.values())

    def matching_orders(self, request: str, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matches = [
            order
            for order in orders
            if order.get("status") != "completed"
            and (
                _same_order_target(request, str(order.get("request") or ""))
                or any(
                    _same_order_target(request, str(component.get("component") or ""))
                    for component in order.get("components") or []
                )
            )
        ]
        return matches

    def order_result_releasable(self, item: dict[str, Any]) -> bool:
        return not is_postdiagnostic_item(item)

    def items_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("evidence_id")): item
            for item in self.case_data.get("evidence_items") or []
            if isinstance(item, dict)
        }

    def pending_components(self, order: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            component
            for component in order.get("components") or []
            if component.get("status") != "completed"
        ]

    def releasable_items_for_component(
        self,
        component: dict[str, Any] | None,
        by_id: dict[str, dict[str, Any]],
        disclosed_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not component:
            return []
        items: list[dict[str, Any]] = []
        for evidence_id in component.get("matched_evidence_ids") or []:
            item = by_id.get(str(evidence_id))
            if not item or str(evidence_id) in disclosed_ids:
                continue
            if self.order_result_releasable(item):
                items.append(item)
        return items

    def next_releasable_component(
        self,
        order: dict[str, Any],
        by_id: dict[str, dict[str, Any]],
        disclosed_ids: set[str],
    ) -> dict[str, Any] | None:
        for component in self.pending_components(order):
            if self.releasable_items_for_component(component, by_id, disclosed_ids):
                return component
        return None

    def best_component_for_request(
        self,
        order: dict[str, Any],
        request: str,
        by_id: dict[str, dict[str, Any]],
        disclosed_ids: set[str],
    ) -> dict[str, Any] | None:
        candidates = [
            component
            for component in self.pending_components(order)
            if self.releasable_items_for_component(component, by_id, disclosed_ids)
        ]
        if not candidates:
            return None
        matching = [
            component
            for component in candidates
            if _same_order_target(request, str(component.get("component") or ""))
        ]
        return (matching or candidates)[0]

    def mark_component_released(
        self,
        order: dict[str, Any],
        component: dict[str, Any] | None,
        items: list[dict[str, Any]],
    ) -> None:
        if not component:
            return
        component["status"] = "completed"
        component["released_evidence_ids"] = [str(item.get("evidence_id")) for item in items]
        if any(component.get("status") != "completed" for component in order.get("components") or []):
            order["status"] = "partial"
        else:
            order["status"] = "completed"

    def _result_response(
        self,
        *,
        request: str,
        current_phase_index: int,
        items: list[dict[str, Any]],
        disclosed_ids: set[str],
        no_result_status: str,
        no_result_message: str,
        orders: list[dict[str, Any]],
        internal_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        new_items = [item for item in items if str(item.get("evidence_id")) not in disclosed_ids]
        for item in new_items:
            disclosed_ids.add(str(item.get("evidence_id")))
        if new_items:
            pending = self.public_pending_orders(orders)
            status = "answered_partial" if any(order.get("status") == "partial" for order in orders) else "answered"
            public = {
                "status": status,
                "message": "Source-grounded result matching the request is provided."
                if status == "answered"
                else "Source-grounded partial result is provided; some ordered components remain pending.",
                "available_information": [public_evidence_view(item) for item in new_items],
                "pending_orders": pending,
            }
        else:
            public = {
                "status": no_result_status,
                "message": no_result_message,
                "available_information": [],
                "pending_orders": self.public_pending_orders(orders),
            }
        internal = {
            "request": request,
            "current_phase_index": current_phase_index,
            "matched_evidence_ids": [str(item.get("evidence_id")) for item in items],
            "disclosed_evidence_ids": [str(item.get("evidence_id")) for item in new_items],
        }
        if internal_extra:
            internal.update(internal_extra)
        return {"public": public, "internal": internal, "image_paths": []}

    def public_pending_orders(self, orders: list[dict[str, Any]]) -> list[dict[str, str]]:
        public: list[dict[str, Any]] = []
        for order in orders:
            if order.get("status") == "completed":
                continue
            entry: dict[str, Any] = {
                "description": str(order.get("request") or ""),
                "status": str(order.get("status") or "pending"),
            }
            pending = [
                str(component.get("component") or "")
                for component in self.pending_components(order)
                if str(component.get("component") or "")
            ]
            if pending:
                entry["pending_components"] = pending
            public.append(entry)
        return public
