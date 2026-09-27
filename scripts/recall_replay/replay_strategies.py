import asyncio
import json
import re
import sys

from sqlalchemy import func, select

from src import models
from src.dependencies import tracked_db
from src.embedding_client import embedding_client

WORKSPACE = "hermes"
CURRENT = "agent-main-slack-dm-T0BDEEKMVLZ-D0BDBNRQ7L3-1790425373-655099-20260926_152253_e8e95dc2"
GAP = {
    "1I5XJS22a9dfQpcy0S4Cv": "sep19-question",
    "lSDQ6ti-_vEzqLKmzK9vf": "sep19-answer",
    "zWzjGvYp_eJ0T1dkRTSa6": "sep01-job-setup",
    "xTf5_Dzo1jXXRRK6nqU_T": "sep13-shared-skill",
    "iaX8HmzK1FMm-mY67CVry": "sep05-honcho-job",
}


async def ranked(embedding: list[float], top: int) -> list[tuple[str, float, str]]:
    distance = models.MessageEmbedding.embedding.cosine_distance(embedding)
    async with tracked_db("replay.strategies", read_only=True) as db:
        rows = await db.execute(
            select(
                models.MessageEmbedding.message_id,
                func.min(distance).label("d"),
                func.min(models.MessageEmbedding.session_name),
            )
            .where(models.MessageEmbedding.workspace_name == WORKSPACE)
            .where(models.MessageEmbedding.embedding.isnot(None))
            .where(models.MessageEmbedding.session_name != CURRENT)
            .group_by(models.MessageEmbedding.message_id)
            .order_by("d")
            .limit(top)
        )
        return [(r[0], float(r[1]), r[2]) for r in rows.all()]


async def heads(ids: list[str]) -> dict[str, str]:
    async with tracked_db("replay.heads", read_only=True) as db:
        rows = await db.execute(
            select(models.Message.public_id, models.Message.created_at, models.Message.peer_name, models.Message.content)
            .where(models.Message.public_id.in_(ids))
        )
        return {
            r[0]: f"{r[1]:%m-%d} {r[2][:8]:8} {r[3][:80]!r}".replace("\\n", " ")
            for r in rows.all()
        }


def gap_ranks(ranking: list[tuple[str, float, str]]) -> dict[str, str]:
    where = {mid: (i + 1, d) for i, (mid, d, _) in enumerate(ranking)}
    return {label: (f"#{where[mid][0]} d={where[mid][1]:.3f}" if mid in where else "absent") for mid, label in GAP.items()}


async def main() -> None:
    text = json.loads(sys.stdin.read())["turn5_current_message_only"]
    sentences = [s.strip() for s in re.split(r"(?<=[.?!…])\s+", text) if len(s.strip()) >= 25]
    report: dict = {"sentences": {}}
    best: dict[str, float] = {}
    rrf: dict[str, float] = {}
    for sentence in sentences:
        emb = await embedding_client.embed(sentence)
        ranking = await ranked(emb, 300)
        for rank, (mid, d, _) in enumerate(ranking, 1):
            best[mid] = min(best.get(mid, 9.0), d)
            if rank <= 20:
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (60 + rank)
        report["sentences"][sentence[:70]] = {
            "top3_dist": [round(d, 3) for _, d, _ in ranking[:3]],
            "gap": gap_ranks(ranking),
        }
    whole = await ranked(await embedding_client.embed(text), 300)
    report["whole_message_current_chat_excluded_in_sql"] = gap_ranks(whole)
    maxsim = sorted(best.items(), key=lambda kv: kv[1])
    report["fused_min_distance"] = gap_ranks([(m, d, "") for m, d in maxsim])
    fused_rrf = sorted(rrf.items(), key=lambda kv: -kv[1])
    report["fused_rrf"] = gap_ranks([(m, -s, "") for m, s in fused_rrf])
    top_ids = [m for m, _ in maxsim[:8]]
    names = await heads(top_ids + [m for m, _ in fused_rrf[:8]] + [m for m, _, _ in whole[:8]])
    report["top8_min_distance"] = [f"{best[m]:.3f} {names.get(m, m)}" for m in top_ids]
    report["top8_rrf"] = [names.get(m, m) for m, _ in fused_rrf[:8]]
    report["top8_whole_message"] = [f"{d:.3f} {names.get(m, m)}" for m, d, _ in whole[:8]]
    print(json.dumps(report, ensure_ascii=False, indent=1))


asyncio.run(main())
