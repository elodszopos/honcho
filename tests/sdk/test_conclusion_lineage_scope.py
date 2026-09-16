"""A pair-scoped view answers for its own pair only, lineage included."""

from __future__ import annotations

import pytest
from nanoid import generate as generate_nanoid

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.conclusions import ConclusionCreateParams
from sdks.python.src.honcho.http import NotFoundError


def _operator(content: str) -> ConclusionCreateParams:
    return ConclusionCreateParams(
        content=content,
        action="create",
        reason_for_entry="Pair scoping probe",
        search_query="scoping",
        searched_conclusion_ids=[],
        source_tool_call_id="lineage-scope-probe",
        entry_origin="operator_sdk",
        agent_trace_id="lineage-scope-trace",
        agent_model="integration-test-model",
    )


@pytest.mark.asyncio
async def test_lineage_refuses_another_pairs_conclusion(
    client_fixture: tuple[Honcho, str],
) -> None:
    client, client_type = client_fixture
    suffix = generate_nanoid(size=10)

    if client_type == "async":
        observer = await client.aio.peer(id=f"lineage-observer-{suffix}")
        mine = await client.aio.peer(id=f"lineage-mine-{suffix}")
        theirs = await client.aio.peer(id=f"lineage-theirs-{suffix}")
        created = await observer.conclusions_of(mine).aio.create(
            [_operator("The user keeps a paper notebook.")]
        )
    else:
        observer = client.peer(id=f"lineage-observer-{suffix}")
        mine = client.peer(id=f"lineage-mine-{suffix}")
        theirs = client.peer(id=f"lineage-theirs-{suffix}")
        created = observer.conclusions_of(mine).create(
            [_operator("The user keeps a paper notebook.")]
        )

    conclusion_id = created[0].id
    own_view = observer.conclusions_of(mine)
    other_view = observer.conclusions_of(theirs)

    if client_type == "async":
        assert (await own_view.aio.lineage(conclusion_id)).id == conclusion_id
        with pytest.raises(NotFoundError):
            await other_view.aio.lineage(conclusion_id)
        assert (await client.aio.conclusions.lineage(conclusion_id)).id == conclusion_id
    else:
        assert own_view.lineage(conclusion_id).id == conclusion_id
        with pytest.raises(NotFoundError):
            other_view.lineage(conclusion_id)
        assert client.conclusions.lineage(conclusion_id).id == conclusion_id
