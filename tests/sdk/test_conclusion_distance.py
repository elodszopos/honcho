"""Conclusion query results carry their cosine distance, closest first; listings carry none."""

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.conclusions import Conclusion
from tests.sdk.test_conclusions import _operator_conclusion

CONTENTS = [
    "User loves Italian cuisine, especially pasta",
    "User runs along the river every morning",
    "User keeps paper receipts in a shoebox",
]


def _assert_ranked_by_distance(results: list[Conclusion]) -> None:
    distances = [conclusion.distance for conclusion in results]
    assert distances, "the query returned nothing"
    for distance in distances:
        assert distance is not None
        assert 0.0 <= distance <= 2.0
    assert distances == sorted(distances)


@pytest.mark.asyncio
async def test_query_results_carry_distance_closest_first(
    client_fixture: tuple[Honcho, str],
):
    honcho, client_type = client_fixture

    if client_type == "async":
        observer = await honcho.aio.peer(id="distance-observer")
        target = await honcho.aio.peer(id="distance-target")
        session = await honcho.aio.session(id="distance-session")
        await session.aio.add_messages(
            [observer.message("hello"), target.message("hi there")]
        )
        scope = observer.conclusions_of(target)
        await scope.aio.create(
            [_operator_conclusion(content, session.id) for content in CONTENTS]
        )
        results = await scope.aio.query("food preferences", top_k=3)
    else:
        observer = honcho.peer(id="distance-observer")
        target = honcho.peer(id="distance-target")
        session = honcho.session(id="distance-session")
        session.add_messages([observer.message("hello"), target.message("hi there")])
        scope = observer.conclusions_of(target)
        scope.create([_operator_conclusion(content, session.id) for content in CONTENTS])
        results = scope.query("food preferences", top_k=3)

    _assert_ranked_by_distance(results)


def test_listed_conclusions_carry_no_distance(honcho_sync_test_client: Honcho):
    observer = honcho_sync_test_client.peer(id="distance-list-observer")
    target = honcho_sync_test_client.peer(id="distance-list-target")
    session = honcho_sync_test_client.session(id="distance-list-session")
    session.add_messages([observer.message("hello"), target.message("hi there")])
    scope = observer.conclusions_of(target)
    scope.create([_operator_conclusion(CONTENTS[0], session.id)])

    listed = list(scope.list().items)

    assert listed
    assert all(conclusion.distance is None for conclusion in listed)
