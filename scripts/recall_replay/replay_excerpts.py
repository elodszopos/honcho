import asyncio
import json
import sys

from sqlalchemy import func, select

from src import models
from src.crud.message import search_messages
from src.dependencies import tracked_db
from src.embedding_client import embedding_client

WORKSPACE = "hermes"
OBSERVER = "elodszopos"
CURRENT = "agent-main-slack-dm-T0BDEEKMVLZ-D0BDBNRQ7L3-1790425373-655099-20260926_152253_e8e95dc2"
SEP19 = {"1I5XJS22a9dfQpcy0S4Cv": "sep19-user-question", "lSDQ6ti-_vEzqLKmzK9vf": "sep19-answer"}


async def exact_ranking(embedding: list[float], top: int) -> list[tuple[str, float, str]]:
    distance = models.MessageEmbedding.embedding.cosine_distance(embedding)
    async with tracked_db("replay.exact", read_only=True) as db:
        rows = await db.execute(
            select(
                models.MessageEmbedding.message_id,
                func.min(distance).label("d"),
                func.min(models.MessageEmbedding.session_name),
            )
            .where(models.MessageEmbedding.workspace_name == WORKSPACE)
            .where(models.MessageEmbedding.embedding.isnot(None))
            .group_by(models.MessageEmbedding.message_id)
            .order_by("d")
            .limit(top)
        )
        return [(r[0], float(r[1]), r[2]) for r in rows.all()]


async def main() -> None:
    queries = json.loads(sys.stdin.read())
    out = {}
    for name, text in queries.items():
        embedding = await embedding_client.embed(text)
        snippets = await search_messages(
            WORKSPACE, None, text, limit=8, context_window=1, embedding=embedding, observer=OBSERVER
        )
        ranking = await exact_ranking(embedding, 400)
        rank_of = {mid: (i + 1, d, s) for i, (mid, d, s) in enumerate(ranking)}
        dist_of = {mid: d for mid, d, _ in ranking}
        live = []
        for index, (matches, context) in enumerate(snippets, 1):
            session = context[0].session_name if context else None
            live.append({
                "excerpt": index,
                "current_chat": session == CURRENT,
                "session_tail": (session or "")[-34:],
                "matches": [
                    {
                        "id": m.public_id,
                        "peer": m.peer_name,
                        "date": m.created_at.strftime("%Y-%m-%d"),
                        "dist": round(dist_of[m.public_id], 4) if m.public_id in dist_of else None,
                        "chars": len(m.content),
                        "head": m.content[:90].replace("\n", " "),
                    }
                    for m in matches
                ],
            })
        current_above = {}
        for mid, label in SEP19.items():
            if mid in rank_of:
                rank = rank_of[mid][0]
                current_above[label] = {
                    "rank": rank,
                    "dist": round(rank_of[mid][1], 4),
                    "current_chat_messages_above": sum(1 for m, _, s in ranking[: rank - 1] if s == CURRENT),
                }
            else:
                current_above[label] = {"rank": ">400"}
        out[name] = {
            "chars": len(text),
            "live_search_excerpts": live,
            "sep19_exact_rank": current_above,
            "top10_exact": [
                {"rank": i + 1, "dist": round(d, 4), "current_chat": s == CURRENT, "session_tail": (s or "")[-34:], "id": mid}
                for i, (mid, d, s) in enumerate(ranking[:10])
            ],
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))


asyncio.run(main())
