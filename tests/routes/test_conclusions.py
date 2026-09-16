import datetime

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.models import Peer, Workspace


def _removal_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "category": "misderived",
        "reason": "The test retired this conclusion",
        "entry_origin": "operator_sdk",
        "agent_trace_id": "test-trace",
        "agent_model": "test-model",
    }
    body.update(overrides)
    return body


class TestConclusionRoutes:
    """Test suite for conclusion API endpoints."""

    @staticmethod
    def _admission_fields(
        *,
        content: str,
        searched_ids: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "action": "create",
            "target_id": None,
            "reason_for_entry": "Test fixture represents a durable fact",
            "search_query": content,
            "searched_conclusion_ids": searched_ids or [],
            "source_message_ids": [],
            "source_tool_call_id": "test-tool-call",
            "entry_origin": "explicit_agent",
            "agent_trace_id": "test-agent-trace",
            "agent_model": "test-agent",
        }

    def _admit(
        self,
        client: TestClient,
        *,
        workspace: str,
        content: str,
        observer: str,
        observed: str,
        session_id: str | None = None,
    ):
        query_response = client.post(
            f"/v3/workspaces/{workspace}/conclusions/query",
            json={
                "query": content,
                "top_k": 10,
                "filters": {"observer": observer, "observed": observed},
            },
        )
        assert query_response.status_code == 200
        searched_ids = [row["id"] for row in query_response.json()]
        conclusion = {
            "content": content,
            "observer_id": observer,
            "observed_id": observed,
            **self._admission_fields(content=content, searched_ids=searched_ids),
        }
        if session_id is not None:
            conclusion["session_id"] = session_id
        return client.post(
            f"/v3/workspaces/{workspace}/conclusions",
            json={"conclusions": [conclusion]},
        )

    async def _create_collection(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> models.Collection:
        """Helper to create collection for tests"""
        collection = models.Collection(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        db_session.add(collection)
        await db_session.flush()
        return collection

    @pytest.mark.asyncio
    async def test_list_conclusions_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions for a session"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create test conclusions (documents)
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User prefers dark mode",
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User works late at night",
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # List conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2

        # Check conclusion structure
        conclusion = data["items"][0]
        assert "id" in conclusion
        assert "content" in conclusion
        assert "observer_id" in conclusion
        assert "observed_id" in conclusion
        assert "session_id" in conclusion
        assert "created_at" in conclusion

        # Verify content
        contents = [item["content"] for item in data["items"]]
        assert "User prefers dark mode" in contents
        assert "User works late at night" in contents

    @pytest.mark.asyncio
    async def test_list_conclusions_empty_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions for a session with no conclusions"""
        test_workspace, _test_peer = sample_data

        # Create a session without any conclusions
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # List conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_list_conclusions_with_filters(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions with observer/observed filters"""
        test_workspace, test_peer = sample_data

        # Create two more peers
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_peer3 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, test_peer3])
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collections for both observer/observed pairs
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )
        await self._create_collection(
            db_session, test_workspace.name, test_peer2.name, test_peer3.name
        )

        # Create conclusions with different observer/observed pairs
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Peer1 observes Peer2",
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer2.name,
            observed=test_peer3.name,
            content="Peer2 observes Peer3",
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # List conclusions filtered by observer
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {"observer": test_peer.name, "session_id": test_session.name}
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["content"] == "Peer1 observes Peer2"
        assert data["items"][0]["observer_id"] == test_peer.name

    @pytest.mark.asyncio
    async def test_list_conclusions_reverse_order(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions in reverse chronological order"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create conclusions with distinct timestamps — docs committed in one
        # transaction share created_at, which would make ordering arbitrary
        base = datetime.datetime.now(datetime.UTC)
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="First conclusion",
            embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
            session_name=test_session.name,
            created_at=base,
        )
        db_session.add(doc1)
        await db_session.flush()

        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Second conclusion",
            embedding=[0.2] * settings.EMBEDDING.VECTOR_DIMENSIONS,
            session_name=test_session.name,
            created_at=base + datetime.timedelta(seconds=1),
        )
        db_session.add(doc2)
        await db_session.commit()

        # List conclusions in reverse (oldest first)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list?reverse=true",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["content"] == "First conclusion"
        assert data["items"][1]["content"] == "Second conclusion"

    @pytest.mark.asyncio
    async def test_list_conclusions_pagination(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test pagination of conclusions list"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create multiple conclusions
        for i in range(15):
            doc = models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                content=f"Conclusion {i}",
                embedding=[0.1 * i] * settings.EMBEDDING.VECTOR_DIMENSIONS,
                session_name=test_session.name,
            )
            db_session.add(doc)
        await db_session.commit()

        # Get first page (default size)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list?page=1&size=10",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10
        assert data["total"] == 15

        # Get second page
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list?page=2&size=10",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5
        assert data["total"] == 15

    @pytest.mark.asyncio
    async def test_query_conclusions_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying conclusions with semantic search"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Admit test conclusions sequentially so each one receives a fresh search.
        for content in ["User loves pizza and pasta", "User dislikes vegetables"]:
            create_response = self._admit(
                client,
                workspace=test_workspace.name,
                content=content,
                observer=test_peer.name,
                observed=test_peer2.name,
                session_id=test_session.name,
            )
            assert create_response.status_code == 201

        # Query conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "food preferences",
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # pyright: ignore

        # Check conclusion structure
        conclusion = data[0]  # pyright: ignore
        assert "id" in conclusion
        assert "content" in conclusion
        assert "observer_id" in conclusion
        assert "observed_id" in conclusion

    @pytest.mark.asyncio
    async def test_query_conclusions_with_top_k(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying conclusions with top_k limit"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Admit multiple conclusions sequentially.
        for i in range(5):
            create_response = self._admit(
                client,
                workspace=test_workspace.name,
                content=f"Conclusion about topic {i}",
                observer=test_peer.name,
                observed=test_peer2.name,
                session_id=test_session.name,
            )
            assert create_response.status_code == 201

        # Query with top_k=2
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "relevant topic",
                "top_k": 2,
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 2  # pyright: ignore

    @pytest.mark.asyncio
    async def test_query_conclusions_with_distance_threshold(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying conclusions with distance threshold"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Admit test conclusion through the mandatory search gate.
        create_response = self._admit(
            client,
            workspace=test_workspace.name,
            content="Test conclusion",
            observer=test_peer.name,
            observed=test_peer2.name,
            session_id=test_session.name,
        )
        assert create_response.status_code == 201

        # Query with distance threshold
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "test",
                "distance": 0.8,
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_query_conclusions_requires_observer_observed(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test query conclusions requires observer and observed in filters"""
        test_workspace, _test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Query without observer/observed filters should fail
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={"query": "test"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_conclusions_invalid_top_k(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test query conclusions validates top_k range"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Query with invalid top_k (too high)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "test",
                "top_k": 101,  # Max is 100
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_conclusion_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test deleting a conclusion"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create a test conclusion
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test conclusion to delete",
            embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        conclusion_id = doc.id

        # Delete conclusion
        response = client.request(
            "DELETE",
            f"/v3/workspaces/{test_workspace.name}/conclusions/{conclusion_id}",
            json=_removal_body(),
        )

        assert response.status_code == 204

        # Verify conclusion is deleted
        from sqlalchemy import select

        stmt = select(models.Document).where(models.Document.id == conclusion_id)
        result = await db_session.execute(stmt)
        doc = result.scalar_one_or_none()
        assert doc is not None
        assert doc.deleted_at is not None

    @pytest.mark.asyncio
    async def test_delete_conclusion_not_found(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test deleting a non-existent conclusion"""
        test_workspace, _test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to delete non-existent conclusion
        response = client.request(
            "DELETE",
            f"/v3/workspaces/{test_workspace.name}/conclusions/nonexistent_id",
            json=_removal_body(),
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_conclusion_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test getting a single conclusion by ID with attribution fields"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create premise conclusions and a derived conclusion referencing them
        premise1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User works late at night",
            embedding=[0.1] * 1536,
        )
        premise2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User prefers dark mode",
            embedding=[0.1] * 1536,
        )
        db_session.add_all([premise1, premise2])
        await db_session.flush()

        derived = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User is likely a night owl",
            embedding=[0.1] * 1536,
            level="deductive",
            source_ids=[premise1.id, premise2.id],
            times_derived=3,
        )
        db_session.add(derived)
        await db_session.commit()

        # Get conclusion by ID
        response = client.get(
            f"/v3/workspaces/{test_workspace.name}/conclusions/{derived.id}"
        )

        assert response.status_code == 200
        conclusion = response.json()
        assert conclusion["id"] == derived.id
        assert conclusion["content"] == "User is likely a night owl"
        assert conclusion["observer_id"] == test_peer.name
        assert conclusion["observed_id"] == test_peer2.name
        assert conclusion["level"] == "deductive"
        assert conclusion["source_ids"] == [premise1.id, premise2.id]
        assert conclusion["times_derived"] == 3

        # Verify internal fields are NOT exposed
        assert "embedding" not in conclusion
        assert "internal_metadata" not in conclusion

    @pytest.mark.asyncio
    async def test_get_conclusion_legacy_source_ids(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """The property still reads well-formed metadata source_ids until drain.
        Malformed ids fail SOURCE_ID_RE and surface as None."""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        source_ids = [str(generate_nanoid()), str(generate_nanoid())]
        # Legacy documents stored source_ids in internal_metadata, not the column
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Legacy derived conclusion",
            embedding=[0.1] * 1536,
            level="deductive",
            internal_metadata={"source_ids": source_ids},
        )
        db_session.add(doc)
        await db_session.commit()

        response = client.get(
            f"/v3/workspaces/{test_workspace.name}/conclusions/{doc.id}"
        )

        assert response.status_code == 200
        conclusion = response.json()
        assert conclusion["source_ids"] == source_ids

    @pytest.mark.asyncio
    async def test_get_conclusion_not_found(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test getting a non-existent conclusion"""
        test_workspace, _test_peer = sample_data

        response = client.get(
            f"/v3/workspaces/{test_workspace.name}/conclusions/nonexistent_id"
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_conclusion_soft_deleted(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that a soft-deleted conclusion returns 404"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Conclusion to delete",
            embedding=[0.1] * 1536,
        )
        db_session.add(doc)
        await db_session.commit()

        delete_response = client.request(
            "DELETE",
            f"/v3/workspaces/{test_workspace.name}/conclusions/{doc.id}",
            json={
                "category": "low_value",
                "reason": "Retired so the plain get can be checked.",
                "entry_origin": "operator_sdk",
                "agent_trace_id": "route-test-trace",
                "agent_model": "route-test-model",
            },
        )
        assert delete_response.status_code == 204

        # Retired rows are kept and stay reachable through /lineage, but the
        # plain get is a live-pool read and must not return them.
        response = client.get(
            f"/v3/workspaces/{test_workspace.name}/conclusions/{doc.id}"
        )
        assert response.status_code == 404

        lineage_response = client.get(
            f"/v3/workspaces/{test_workspace.name}/conclusions/{doc.id}/lineage"
        )
        assert lineage_response.status_code == 200
        assert lineage_response.json()["removal"]["category"] == "low_value"

    @pytest.mark.asyncio
    async def test_list_conclusions_includes_attribution_fields(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that list surfaces source_ids and times_derived"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # document_sources check-constrains source_id to nanoid shape
        source_ids = [str(generate_nanoid()), str(generate_nanoid())]
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Derived conclusion",
            embedding=[0.1] * 1536,
            level="inductive",
            source_ids=source_ids,
            times_derived=2,
        )
        db_session.add(doc)
        await db_session.commit()

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {
                    "observer_id": test_peer.name,
                    "observed_id": test_peer2.name,
                }
            },
        )

        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["source_ids"] == source_ids
        assert items[0]["times_derived"] == 2

    @pytest.mark.asyncio
    async def test_list_conclusions_nonexistent_session(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions for non-existent session"""
        test_workspace, _test_peer = sample_data

        # Try to list conclusions for non-existent session
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": "nonexistent_session"}},
        )

        # Should return empty result, not error (session might exist but no conclusions)
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_conclusions_field_mapping(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that conclusion fields are properly mapped from document model"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create test conclusion
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test conclusion content",
            embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        # List conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        conclusion = data["items"][0]

        # Verify field mappings
        assert conclusion["id"] == doc.id
        assert conclusion["content"] == doc.content
        assert conclusion["observer_id"] == doc.observer
        assert conclusion["observed_id"] == doc.observed
        assert conclusion["session_id"] == doc.session_name
        assert conclusion["level"] == "explicit"
        assert conclusion["source_ids"] is None
        assert conclusion["times_derived"] == 1
        assert "created_at" in conclusion

        # Verify internal fields are NOT exposed
        assert "embedding" not in conclusion
        assert "internal_metadata" not in conclusion
        assert "collection" not in conclusion

    @pytest.mark.asyncio
    async def test_list_conclusions_filter_by_level(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Filtering by `level` returns only conclusions at that reasoning level.

        `level="explicit"` is the "not dreamed on" view — it excludes the
        deductive/inductive conclusions produced during dreaming.
        """
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Two explicit, one deductive, one inductive
        levels = ["explicit", "explicit", "deductive", "inductive"]
        for i, level in enumerate(levels):
            db_session.add(
                models.Document(
                    workspace_name=test_workspace.name,
                    observer=test_peer.name,
                    observed=test_peer2.name,
                    content=f"{level} conclusion {i}",
                    embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
                    session_name=test_session.name,
                    level=level,
                )
            )
        await db_session.commit()

        # No level filter -> all four
        all_resp = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )
        assert all_resp.status_code == 200
        assert all_resp.json()["total"] == 4

        # level="explicit" -> only the two non-dreamed conclusions
        explicit_resp = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name, "level": "explicit"}},
        )
        assert explicit_resp.status_code == 200
        explicit_data = explicit_resp.json()
        assert explicit_data["total"] == 2
        assert all(item["level"] == "explicit" for item in explicit_data["items"])

        # level="deductive" -> only the one deductive conclusion
        deductive_resp = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name, "level": "deductive"}},
        )
        assert deductive_resp.status_code == 200
        deductive_data = deductive_resp.json()
        assert deductive_data["total"] == 1
        assert deductive_data["items"][0]["level"] == "deductive"

    @pytest.mark.asyncio
    async def test_create_conclusion_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating a single conclusion"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create conclusion through mandatory semantic search admission.
        response = self._admit(
            client,
            workspace=test_workspace.name,
            content="User prefers dark mode",
            observer=test_peer.name,
            observed=test_peer2.name,
            session_id=test_session.name,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1

        conclusion = data[0]
        assert conclusion["content"] == "User prefers dark mode"
        assert conclusion["observer_id"] == test_peer.name
        assert conclusion["observed_id"] == test_peer2.name
        assert conclusion["session_id"] == test_session.name
        assert "id" in conclusion
        assert "created_at" in conclusion

    @pytest.mark.asyncio
    async def test_create_conclusions_batch(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Each conclusion in a batch carries its own agent search decision."""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create multiple conclusions via API
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "User prefers dark mode",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="User prefers dark mode"),
                    },
                    {
                        "content": "User works late at night",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="User works late at night"),
                    },
                    {
                        "content": "User enjoys programming",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="User enjoys programming"),
                    },
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 3
        assert {row["content"] for row in data} == {
            "User prefers dark mode",
            "User works late at night",
            "User enjoys programming",
        }

    @pytest.mark.asyncio
    async def test_create_conclusion_nonexistent_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusion with non-existent session fails"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.commit()

        # Try to create conclusion with non-existent session
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Test conclusion",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": "nonexistent_session",
                        **self._admission_fields(content="Test conclusion"),
                    }
                ]
            },
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_conclusion_nonexistent_peer(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusion with non-existent peer fails"""
        test_workspace, test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to create conclusion with non-existent observer
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Test conclusion",
                        "observer_id": "nonexistent_peer",
                        "observed_id": test_peer.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="Test conclusion"),
                    }
                ]
            },
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_conclusion_empty_content(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusion with empty content fails validation"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to create conclusion with empty content
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_conclusion_empty_list(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusions with empty list fails validation"""
        test_workspace, _test_peer = sample_data

        # Try to create with empty conclusions list
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={"conclusions": []},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_conclusion_creates_collection(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that creating conclusion auto-creates collection if needed"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Admission should also create the scoped collection.
        response = self._admit(
            client,
            workspace=test_workspace.name,
            content="Test conclusion",
            observer=test_peer.name,
            observed=test_peer2.name,
            session_id=test_session.name,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1

        # The conclusion was created successfully, which means the collection
        # was created (since documents require a collection)
        conclusion = data[0]
        assert conclusion["observer_id"] == test_peer.name
        assert conclusion["observed_id"] == test_peer2.name

    @pytest.mark.asyncio
    async def test_create_conclusion_different_observer_observed_pairs(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusions with different observer/observed pairs in single batch"""
        test_workspace, test_peer = sample_data

        # Create two more peers
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_peer3 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, test_peer3])
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create conclusions with different observer/observed pairs
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Peer1 observes Peer2",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="Peer1 observes Peer2"),
                    },
                    {
                        "content": "Peer2 observes Peer3",
                        "observer_id": test_peer2.name,
                        "observed_id": test_peer3.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="Peer2 observes Peer3"),
                    },
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 2
        assert {(row["observer_id"], row["observed_id"]) for row in data} == {
            (test_peer.name, test_peer2.name),
            (test_peer2.name, test_peer3.name),
        }

    @pytest.mark.asyncio
    async def test_created_conclusions_are_searchable(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that created conclusions can be found via list endpoint"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Admit conclusion through mandatory semantic search.
        create_response = self._admit(
            client,
            workspace=test_workspace.name,
            content="Unique test content for searchability",
            observer=test_peer.name,
            observed=test_peer2.name,
            session_id=test_session.name,
        )

        assert create_response.status_code == 201
        created_id = create_response.json()[0]["id"]

        # List conclusions and verify the created one is there
        list_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                }
            },
        )

        assert list_response.status_code == 200
        data = list_response.json()
        ids = [obs["id"] for obs in data["items"]]
        assert created_id in ids

    @pytest.mark.asyncio
    async def test_create_conclusion_without_session_id(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating a conclusion without session_id (sessionless/global conclusion)"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.commit()

        # Admit conclusion without session_id.
        response = self._admit(
            client,
            workspace=test_workspace.name,
            content="User prefers dark mode (global)",
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1

        conclusion = data[0]
        assert conclusion["content"] == "User prefers dark mode (global)"
        assert conclusion["observer_id"] == test_peer.name
        assert conclusion["observed_id"] == test_peer2.name
        assert conclusion["session_id"] is None  # Should be null
        assert "id" in conclusion
        assert "created_at" in conclusion

    @pytest.mark.asyncio
    async def test_create_conclusions_mixed_session_and_sessionless(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating a batch with both session-scoped and sessionless conclusions"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create mixed batch: one with session, one without
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Session-scoped conclusion",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                        **self._admission_fields(content="Session-scoped conclusion"),
                    },
                    {
                        "content": "Global conclusion without session",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        **self._admission_fields(
                            content="Global conclusion without session"
                        ),
                        # No session_id
                    },
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 2
        assert {row["session_id"] for row in data} == {test_session.name, None}

    @pytest.mark.asyncio
    async def test_list_sessionless_conclusions(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that sessionless conclusions can be listed without session filter"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.commit()

        # Admit sessionless conclusion.
        create_response = self._admit(
            client,
            workspace=test_workspace.name,
            content="Sessionless conclusion for list test",
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        assert create_response.status_code == 201
        created_id = create_response.json()[0]["id"]

        # List all conclusions (no session filter)
        list_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {
                    "observer_id": test_peer.name,
                    "observed_id": test_peer2.name,
                }
            },
        )

        assert list_response.status_code == 200
        data = list_response.json()
        ids = [obs["id"] for obs in data["items"]]
        assert created_id in ids

        # Verify the conclusion has null session_id
        conclusion = next(c for c in data["items"] if c["id"] == created_id)
        assert conclusion["session_id"] is None

    async def _create_reasoning_tree(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> tuple[models.Document, models.Document, models.Document]:
        """Helper to create a premise with two derived conclusions referencing it.

        Returns (premise, derived1, derived2), where derived2 is newer.
        """
        # Explicit created_at values: rows created in one transaction all get
        # the same now(), which would make recency ordering unstable.
        base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        premise = models.Document(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User works late at night",
            embedding=[0.1] * 1536,
            created_at=base,
        )
        db_session.add(premise)
        await db_session.flush()

        derived1 = models.Document(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User is likely a night owl",
            embedding=[0.1] * 1536,
            level="deductive",
            source_ids=[premise.id],
            created_at=base + datetime.timedelta(minutes=1),
        )
        db_session.add(derived1)
        await db_session.flush()

        derived2 = models.Document(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User tends to respond slowly in the mornings",
            embedding=[0.1] * 1536,
            level="inductive",
            source_ids=[premise.id, derived1.id],
            times_derived=2,
            created_at=base + datetime.timedelta(minutes=2),
        )
        db_session.add(derived2)
        await db_session.commit()
        return premise, derived1, derived2

    @pytest.mark.asyncio
    async def test_get_derived_conclusions_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test traversing the reasoning DAG upward via source_ids membership"""
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        premise, derived1, derived2 = await self._create_reasoning_tree(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"source_ids": {"contains": premise.id}}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        ids = [item["id"] for item in data["items"]]
        # Newest first by default
        assert ids == [derived2.id, derived1.id]

        # Attribution fields are present on the children
        child = next(item for item in data["items"] if item["id"] == derived2.id)
        assert child["level"] == "inductive"
        assert child["source_ids"] == [premise.id, derived1.id]
        assert child["times_derived"] == 2

        # derived1 is itself a source of derived2
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"source_ids": {"contains": derived1.id}}},
        )
        assert response.status_code == 200
        data = response.json()
        assert [item["id"] for item in data["items"]] == [derived2.id]

    @pytest.mark.asyncio
    async def test_get_derived_conclusions_reverse(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that reverse=true returns oldest first"""
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        premise, derived1, derived2 = await self._create_reasoning_tree(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            params={"reverse": "true"},
            json={"filters": {"source_ids": {"contains": premise.id}}},
        )

        assert response.status_code == 200
        ids = [item["id"] for item in response.json()["items"]]
        assert ids == [derived1.id, derived2.id]

    @pytest.mark.asyncio
    async def test_get_derived_conclusions_leaf_is_empty(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that a conclusion nothing was derived from returns an empty page"""
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        _premise, _derived1, derived2 = await self._create_reasoning_tree(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"source_ids": {"contains": derived2.id}}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_get_derived_conclusions_not_found(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """A nonexistent source id yields an empty page, not an error."""
        test_workspace, _test_peer = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"source_ids": {"contains": str(generate_nanoid())}}},
        )

        assert response.status_code == 200
        assert response.json()["items"] == []

    @pytest.mark.asyncio
    async def test_list_conclusions_filter_by_source_ids_contains(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test filtering the list endpoint by source_ids membership"""
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        premise, derived1, derived2 = await self._create_reasoning_tree(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"source_ids": {"contains": premise.id}}},
        )

        assert response.status_code == 200
        ids = {item["id"] for item in response.json()["items"]}
        assert ids == {derived1.id, derived2.id}

    @pytest.mark.asyncio
    async def test_list_conclusions_filter_by_ids(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test batch-fetching conclusions by ID through the list endpoint"""
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        premise, derived1, derived2 = await self._create_reasoning_tree(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Fetch derived2's premises the way an SDK tree walk would
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"id": {"in": [premise.id, derived1.id]}}},
        )

        assert response.status_code == 200
        ids = {item["id"] for item in response.json()["items"]}
        assert ids == {premise.id, derived1.id}
        assert derived2.id not in ids

    @pytest.mark.asyncio
    async def test_negation_includes_conclusions_with_no_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Negation must not silently drop workspace-level conclusions.

        A conclusion with no session is not "some other session", so excluding
        that session has to leave it in the result. Under SQL's three-valued
        logic a comparison against NULL is NULL, which would drop the row.

        This is only visible by counting returned rows — the filter builds and
        executes cleanly either way.
        """
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        other_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_session, other_session])
        await db_session.commit()

        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        scoped = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Scoped to a session",
            session_name=test_session.name,
        )
        workspace_level = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Not scoped to any session",
            session_name=None,
        )
        db_session.add_all([scoped, workspace_level])
        await db_session.commit()

        def contents(filters: dict[str, object]) -> set[str]:
            response = client.post(
                f"/v3/workspaces/{test_workspace.name}/conclusions/list",
                json={"filters": filters},
            )
            assert response.status_code == 200, response.text
            return {item["content"] for item in response.json()["items"]}

        both = {"Scoped to a session", "Not scoped to any session"}

        # NOT and ne agree, and both keep the session-less conclusion.
        assert contents({"NOT": [{"session_id": other_session.name}]}) == both
        assert contents({"session_id": {"ne": other_session.name}}) == both

        # Requiring the field to be set is how you narrow to sessioned rows.
        assert contents(
            {
                "AND": [
                    {"session_id": {"ne": other_session.name}},
                    {"session_id": {"ne": None}},
                ]
            }
        ) == {"Scoped to a session"}
