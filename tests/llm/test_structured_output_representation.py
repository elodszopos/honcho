import pytest
from pydantic import BaseModel

from src.llm.structured_output import repair_response_model_json
from src.utils.representation import AdmissionRepresentation, ExtractedRepresentation


@pytest.mark.parametrize(
    ("response_model", "reasons_field"),
    [(ExtractedRepresentation, "skipped"), (AdmissionRepresentation, "rejected")],
)
def test_broken_representation_json_falls_back_to_empty(
    response_model: type[BaseModel], reasons_field: str
) -> None:
    repaired = repair_response_model_json("not-json", response_model, "test-model")

    assert isinstance(repaired, response_model)
    assert repaired.model_dump() == {"explicit": [], reasons_field: []}


@pytest.mark.parametrize(
    ("response_model", "reasons_field"),
    [(ExtractedRepresentation, "skipped"), (AdmissionRepresentation, "rejected")],
)
def test_null_lists_normalize_to_empty(
    response_model: type[BaseModel], reasons_field: str
) -> None:
    parsed = response_model.model_validate({"explicit": None, reasons_field: None})

    assert parsed.model_dump() == {"explicit": [], reasons_field: []}


def test_left_out_items_carry_their_reason() -> None:
    extracted = ExtractedRepresentation.model_validate(
        {
            "explicit": [],
            "skipped": [
                {
                    "statement": "The user paid the plumber today.",
                    "reason": "one occasion",
                }
            ],
        }
    )
    admitted = AdmissionRepresentation.model_validate(
        {"rejected": [{"admission_case_id": 0, "reason": "no user message states it"}]}
    )

    assert extracted.skipped[0].reason == "one occasion"
    assert admitted.rejected[0].admission_case_id == 0
