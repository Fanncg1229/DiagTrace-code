from __future__ import annotations

import pytest

from interactive_diagnosis_workspace.setting_2_evaluation.metrics import categorize_request_text


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("Please provide the pathology findings.", "pathology"),
        ("Please provide the histopathology findings.", "pathology"),
        ("Please provide the cytological findings.", "pathology"),
        ("Please provide the immunohistochemistry results.", "pathology"),
        ("Please describe the cellular morphology.", "pathology"),
        ("Please provide the sonographic findings.", "imaging"),
        ("Please provide the elastography findings.", "imaging"),
        ("Please provide the echocardiography findings.", "imaging"),
        ("Please provide the angiographic findings.", "imaging"),
        ("Please provide the endoscopic findings.", "imaging"),
        ("Please provide the laboratory results.", "lab_tests"),
        ("Please provide the serological results.", "lab_tests"),
        ("Please provide the immunosuppressive treatment response.", "medication_or_treatment"),
    ],
)
def test_request_classifier_matches_complete_word_forms(request_text: str, expected: str) -> None:
    assert categorize_request_text(request_text, "REQUEST_RESULT") == expected
