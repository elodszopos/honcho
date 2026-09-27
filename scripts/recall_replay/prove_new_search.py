import asyncio
import json
import sys

from src.crud.document import embed_thoughts
from src.crud.message import search_message_snippets

CURRENT = "agent-main-slack-dm-T0BDEEKMVLZ-D0BDBNRQ7L3-1790425373-655099-20260926_152253_e8e95dc2"
SEP19 = {"1I5XJS22a9dfQpcy0S4Cv", "lSDQ6ti-_vEzqLKmzK9vf"}
GAP = {"1I5XJS22a9dfQpcy0S4Cv", "lSDQ6ti-_vEzqLKmzK9vf", "zWzjGvYp_eJ0T1dkRTSa6",
       "HHNFSYyjawDlO0AxJtu3b", "xTf5_Dzo1jXXRRK6nqU_T", "nZ6vzDfOrBT29EwK4LwO6",
       "iaX8HmzK1FMm-mY67CVry", "Zcmg1QBn2aHJqrxXrTftL", "6mCMzsWai78pQFXRGAd-F"}


async def main() -> None:
    queries = json.loads(open(sys.argv[1]).read())
    for name in ("turn5_current_message_only", "laneB_turn8_blend"):
        text = queries[name]
        embeddings = await embed_thoughts(text)
        snippets, distances = await search_message_snippets(
            "hermes", embeddings, limit=5, context_window=1, observer="elodszopos",
            exclude_session_name=CURRENT, min_chars=40,
        )
        print(f"== {name}: {len(embeddings)} thoughts, {len(snippets)} excerpts")
        for rank, (matches, context) in enumerate(snippets, 1):
            ids = {m.public_id for m in matches}
            tag = "SEP19" if ids & SEP19 else ("gap" if ids & GAP else "")
            best = min(distances[m.public_id] for m in matches)
            head = matches[0].content[:80].replace("\n", " ")
            print(f"  {rank}. {best:.3f} {matches[0].created_at:%m-%d} {tag:5} {context[0].session_name[-24:]} | {head!r}")


asyncio.run(main())
