from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from src.utils.json_parser import validate_and_repair_json
from src.utils.representation import AdmissionRepresentation, ExtractedRepresentation

_REPRESENTATION_RESPONSE_MODELS = (ExtractedRepresentation, AdmissionRepresentation)


class StructuredOutputError(ValueError):
    """Raised when structured output cannot be validated or repaired."""


def repair_response_model_json(
    raw_content: str,
    response_model: type[BaseModel],
    _model: str,
) -> BaseModel:
    """Repair truncated or malformed JSON and validate against the response model."""

    try:
        final = validate_and_repair_json(raw_content)
        repaired_data = json.loads(final)

        final = json.dumps(repaired_data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        final = ""

    try:
        return response_model.model_validate_json(final)
    except ValidationError:
        if response_model in _REPRESENTATION_RESPONSE_MODELS:
            return response_model.model_validate({"explicit": []})
        raise


def validate_structured_output(
    content: object,
    response_model: type[BaseModel],
) -> BaseModel:
    if isinstance(content, response_model):
        return content
    if isinstance(content, str):
        return response_model.model_validate_json(content)
    if isinstance(content, dict):
        return response_model.model_validate(content)
    raise StructuredOutputError(
        f"Unsupported structured output payload: {type(content).__name__}"
    )


def empty_structured_output(response_model: type[BaseModel]) -> BaseModel:
    return response_model.model_validate({})
