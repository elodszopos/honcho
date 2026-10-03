import glob
import gzip
import random
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src import models
from src.dependencies import tracked_db

WORKSPACE = "hermes"
TARGET = "elodszopos"
ARCHIVES = Path.home() / ".hermes" / "logs" / "honcho"
BATCH_START = re.compile(
    r"deriver\.batch start: workspace=hermes session=(\S+) observed=elodszopos .*?message_id_range=(\d+):(\d+)"
)
KEEP = {"LIVE"}
MAYBE = {"superseded_by_enrichment", "superseded", "duplicate_absorbed"}
THREAD_END = "[End of thread context]"
REPLYING_TO = re.compile(r'^\[Replying to: ".*?"\]\s*', re.DOTALL)
SPEAKER = re.compile(r"^\[Bossman(?: \| Slack user <@[^>]+>)?\]\s*")
DOCUMENT_NOTE = re.compile(
    r"^\[The user (?:sent|shared) (?:a document|an image)[^\]]*\]\s*"
)


def user_words(text: str) -> str:
    # Messages stored before late September carry gateway wrappers the deriver no longer sees
    if text.startswith("[IMPORTANT: Background process"):
        return ""
    if THREAD_END in text:
        text = text.rsplit(THREAD_END, 1)[1]
    text = REPLYING_TO.sub("", text.strip(), count=1)
    text = SPEAKER.sub("", text, count=1)
    return DOCUMENT_NOTE.sub("", text, count=1).strip()


def archive_windows() -> list[tuple[str, int, int]]:
    seen: list[tuple[str, int, int]] = []
    for path in sorted(glob.glob(str(ARCHIVES / "deriver-*.log.gz"))):
        with gzip.open(path, "rt", errors="replace") as log:
            for line in log:
                match = BATCH_START.search(line)
                if match:
                    window = (match.group(1), int(match.group(2)), int(match.group(3)))
                    if window not in seen:
                        seen.append(window)
    return seen


async def build(
    total: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    async with tracked_db("deriver_eval.cases", read_only=True) as db:
        docs = (
            await db.execute(
                select(
                    models.Document.session_name,
                    models.Document.internal_metadata,
                    models.Document.content,
                    models.Document.deleted_at,
                ).where(models.Document.workspace_name == WORKSPACE)
            )
        ).all()
        messages = (
            await db.execute(
                select(
                    models.Message.id,
                    models.Message.session_name,
                    models.Message.peer_name,
                    models.Message.created_at,
                    models.Message.content,
                )
                .where(models.Message.workspace_name == WORKSPACE)
                .order_by(models.Message.id)
            )
        ).all()

    by_session: dict[str, list[Any]] = {}
    for row in messages:
        by_session.setdefault(row.session_name, []).append(row)

    def window_rows(session: str, lo: int, hi: int, context: bool) -> list[Any]:
        rows = by_session.get(session, [])
        inside = [r for r in rows if lo <= r.id <= hi]
        if context and inside:
            before = [r for r in rows if r.id < inside[0].id]
            if before:
                inside = [before[-1], *inside]
        return inside

    fates = []
    for doc in docs:
        ids = doc.internal_metadata.get("message_ids") or []
        if not isinstance(ids, list) or not ids:
            continue
        removal = (doc.internal_metadata.get("removal") or {}).get("category")
        fate = "LIVE" if doc.deleted_at is None else (removal or "deleted")
        fates.append((doc.session_name, [int(i) for i in ids], fate, doc.content))

    windows: list[tuple[str, int, int, str]] = [
        (s, lo, hi, "archive") for s, lo, hi in archive_windows()
    ]
    for session, ids, _, _ in fates:
        windows.append((session, min(ids), max(ids), "conclusion"))

    taken: dict[str, list[tuple[int, int]]] = {}
    positives = []
    for session, lo, hi, origin in windows:
        spans = taken.setdefault(session, [])
        if any(not (hi < a or lo > b) for a, b in spans):
            continue
        rows = window_rows(session, lo, hi, context=origin == "conclusion")
        if not any(r.peer_name == TARGET and user_words(r.content) for r in rows):
            continue
        spans.append((rows[0].id, rows[-1].id))
        positives.append((session, rows, origin))

    rng = random.Random(seed)
    candidates = []
    for session, rows in by_session.items():
        spans = taken.get(session, [])
        for i, row in enumerate(rows):
            if row.peer_name != TARGET or any(a <= row.id <= b for a, b in spans):
                continue
            if not user_words(row.content):
                continue
            candidates.append((session, rows[max(0, i - 1) : i + 1], "negative"))
    rng.shuffle(candidates)
    negatives = candidates[: max(0, total - len(positives))]

    chosen = (positives + negatives)[:total]
    chosen.sort(key=lambda c: (c[1][0].created_at, c[0]))
    cases, drafts = [], []
    for n, (session, rows, origin) in enumerate(chosen):
        case_id = f"c{n:03d}"
        ids = {r.id for r in rows}
        must, may, rejected = [], [], []
        for doc_session, doc_ids, fate, content in fates:
            if doc_session != session or not set(doc_ids) & ids:
                continue
            if fate in KEEP:
                must.append(content)
            elif fate in MAYBE:
                may.append(content)
            else:
                rejected.append(f"{fate}: {content}")
        cases.append(
            {
                "case_id": case_id,
                "split": "heldout" if n % 3 == 2 else "tuning",
                "session": session,
                "range": [rows[0].id, rows[-1].id],
                "messages": [
                    {
                        "id": r.id,
                        "peer": r.peer_name,
                        "time": r.created_at.replace(
                            tzinfo=None, microsecond=0
                        ).isoformat(),
                        "content": user_words(r.content)
                        if r.peer_name == TARGET
                        else r.content,
                    }
                    for r in rows
                    if r.peer_name != TARGET or user_words(r.content)
                ],
                "tags": [origin],
            }
        )
        drafts.append(
            {
                "case_id": case_id,
                "must": must,
                "may": may,
                "rejected": rejected,
                "ruled_by": "draft",
            }
        )
    return cases, drafts
