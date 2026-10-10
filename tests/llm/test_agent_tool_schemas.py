from typing import Any, cast

from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    TOOLS,
)


def _observation_items_schema(tool_key: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        TOOLS[tool_key]["input_schema"]["properties"]["observations"]["items"],
    )


def test_deductive_specialist_tool_requires_evidence_fields() -> None:
    items = _observation_items_schema("create_observations_deductive")

    assert TOOLS["create_observations_deductive"]["name"] == (
        "create_observations_deductive"
    )
    assert set(items["required"]) == {
        "content",
        "source_ids",
        "premises",
        "action",
        "reason_for_entry",
        "search_query",
        "searched_conclusion_ids",
    }
    assert items["properties"]["source_ids"]["minItems"] == 1
    assert items["properties"]["premises"]["minItems"] == 1


def test_inductive_specialist_tool_requires_pattern_fields() -> None:
    items = _observation_items_schema("create_observations_inductive")

    assert TOOLS["create_observations_inductive"]["name"] == (
        "create_observations_inductive"
    )
    assert set(items["required"]) == {
        "content",
        "source_ids",
        "sources",
        "pattern_type",
        "confidence",
        "action",
        "reason_for_entry",
        "search_query",
        "searched_conclusion_ids",
    }
    assert items["properties"]["source_ids"]["minItems"] == 2
    assert items["properties"]["sources"]["minItems"] == 2


def test_dreamer_specialists_use_level_specific_creation_tools() -> None:
    deduction_tool_names = {tool["name"] for tool in DEDUCTION_SPECIALIST_TOOLS}
    induction_tool_names = {tool["name"] for tool in INDUCTION_SPECIALIST_TOOLS}

    assert "create_observations_deductive" in deduction_tool_names
    assert "create_observations_inductive" in induction_tool_names
    assert "create_observations" not in deduction_tool_names
    assert "create_observations" not in induction_tool_names
