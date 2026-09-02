import pytest
from pydantic import BaseModel

from src.llm.structured_output import repair_response_model_json
from src.utils.representation import AdmissionRepresentation, ExtractedRepresentation


@pytest.mark.parametrize(
    "response_model",
    [ExtractedRepresentation, AdmissionRepresentation],
)
def test_broken_representation_json_falls_back_to_empty(
    response_model: type[BaseModel],
) -> None:
    repaired = repair_response_model_json("not-json", response_model, "test-model")

    assert isinstance(repaired, response_model)
    assert repaired.model_dump() == {"explicit": []}


@pytest.mark.parametrize(
    "response_model",
    [ExtractedRepresentation, AdmissionRepresentation],
)
def test_null_explicit_list_normalizes_to_empty(
    response_model: type[BaseModel],
) -> None:
    parsed = response_model.model_validate({"explicit": None})

    assert parsed.model_dump() == {"explicit": []}
