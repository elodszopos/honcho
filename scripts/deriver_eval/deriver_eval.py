import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from build_cases import build
from pydantic import BaseModel

from src.config import settings
from src.deriver.prompts import (
    deriver_messages,
    deriver_system_prompt,
    format_deriver_message,
)
from src.llm import honcho_llm_call
from src.llm.errors import is_transient_llm_error
from src.utils.curated_memory import curated_memory_block
from src.utils.representation import (
    AdmissionRepresentation,
    ExtractedRepresentation,
    Representation,
)

EVAL_HOME = Path(
    os.environ.get("DERIVER_EVAL_HOME", "~/.honcho/deriver-eval")
).expanduser()
WORKSPACE = "hermes"
TARGET = "elodszopos"
REPLAY_CACHE_KEY = "deriver-eval"
JUDGE_CACHE_KEY = "deriver-eval-judge"
CONCURRENCY = 4
JUDGE_BATCH = 8
RUBRIC_VERSION = "3"
QUOTA_REACHED = asyncio.Event()


class QuotaReached(Exception):
    pass


JUDGE_RUBRIC = """
ROLE:
- Grade what a memory extractor stored from one chat batch against that batch's labels.

INPUT, PER CASE:
- `must`: facts that should be stored, numbered from 0.
- `may`: facts that are acceptable either way.
- `held`: facts the assistant already sees in every prompt; storing one repeats it.
- `stored`: what the extractor stored, numbered from 0.

GRADING:
- A stored item matches a label when it conveys the same fact; wording, framing, added or missing detail and a date anchor never make it a different fact.
- A stored item is `distorted` garbage only when what it says contradicts a label.
- Give every stored item exactly one verdict:
  - `must`, with `must_index`, when it matches a must fact.
  - `may` when it matches a may fact.
  - `duplicate` when it matches a held fact, or another stored item of the same case already covers its fact.
  - `garbage` otherwise, with the category that describes it best.
- List in `missed_must` every must index that no stored item covers.
- Judge meaning, never shared vocabulary.

GARBAGE CATEGORIES:
- `occasion`: true only for one occasion or only for now.
- `procedure`: how a task is carried out.
- `configuration`: a configured value.
- `mechanics`: how a system, tool or piece of code works or behaved.
- `memory_rule`: a rule about what memory should hold.
- `distorted`: contradicts a label.
- `other`: anything else.
""".strip()

GarbageCategory = Literal[
    "occasion",
    "procedure",
    "configuration",
    "mechanics",
    "memory_rule",
    "distorted",
    "other",
]


class JudgedItem(BaseModel):
    index: int
    verdict: Literal["must", "may", "duplicate", "garbage"]
    must_index: int | None
    category: GarbageCategory | None


class JudgedCase(BaseModel):
    case_id: str
    items: list[JudgedItem]
    missed_must: list[int]


class JudgeBatch(BaseModel):
    cases: list[JudgedCase]


class Usage:
    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    def add(
        self, pass_name: str, input_tokens: int, cached_tokens: int, output_tokens: int
    ) -> None:
        self.calls[pass_name].append((input_tokens, cached_tokens, output_tokens))

    def summary(self) -> dict[str, dict[str, Any]]:
        out = {}
        for pass_name, rows in self.calls.items():
            total = sum(r[0] for r in rows)
            cached = sum(r[1] for r in rows)
            out[pass_name] = {
                "calls": len(rows),
                "input_tokens": total,
                "output_tokens": sum(r[2] for r in rows),
                "cached_tokens": cached,
                "cached_share": round(cached / total, 3) if total else 0.0,
                "cold_calls": sum(1 for r in rows if r[1] == 0),
            }
        return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    path.chmod(0o600)


def model_config(cache_key: str) -> Any:
    base = settings.DERIVER.MODEL_CONFIG
    params = dict(base.overrides.provider_params)
    params["extra_body"] = {
        **params.get("extra_body", {}),
        "prompt_cache_key": cache_key,
    }
    return base.model_copy(
        update={
            "overrides": base.overrides.model_copy(update={"provider_params": params})
        }
    )


def max_tokens() -> int:
    return (
        settings.DERIVER.MODEL_CONFIG.max_output_tokens
        or settings.LLM.DEFAULT_MAX_TOKENS
    )


async def llm(
    usage: Usage,
    pass_name: str,
    cache_key: str,
    messages: list[dict[str, Any]],
    response_model: Any,
) -> Any:
    if QUOTA_REACHED.is_set():
        raise QuotaReached("not called: the provider's usage limit was reached")
    try:
        response = await honcho_llm_call(
            model_config=model_config(cache_key),
            prompt=messages[-1]["content"],
            messages=messages,
            max_tokens=max_tokens(),
            response_model=response_model,
            json_mode=True,
            max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
            enable_retry=True,
            retry_attempts=3,
            trace_name=f"deriver_eval_{pass_name}",
        )
    except Exception as exc:
        if not is_transient_llm_error(exc):
            QUOTA_REACHED.set()
        raise
    usage.add(
        pass_name,
        response.input_tokens,
        response.cache_read_input_tokens,
        response.output_tokens,
    )
    return response.content


async def primed_gather(items: list[Any], worker: Any) -> list[Any]:
    # The first item runs alone so its cached prefix exists before the rest fan out
    if not items:
        return []
    first = await worker(items[0])
    gate = asyncio.Semaphore(CONCURRENCY)

    async def bounded(item: Any) -> Any:
        async with gate:
            return await worker(item)

    rest = await asyncio.gather(*(bounded(item) for item in items[1:]))
    return [first, *rest]


def formatted_batch(case: dict[str, Any]) -> str:
    return "\n".join(
        format_deriver_message(
            idx,
            m["id"],
            m["peer"],
            TARGET,
            datetime.fromisoformat(m["time"]),
            m["content"],
        )
        for idx, m in enumerate(case["messages"])
    )


async def replay_case(
    case: dict[str, Any], curated: str, usage: Usage
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "candidates": [],
        "skipped": [],
        "stored": [],
        "stored_detail": [],
        "rejected": [],
    }
    try:
        messages = formatted_batch(case)
        extracted = await llm(
            usage,
            "extraction",
            REPLAY_CACHE_KEY,
            deriver_messages(peer_id=TARGET, messages=messages, curated_memory=curated),
            ExtractedRepresentation,
        )
        record["candidates"] = [item.content for item in extracted.explicit]
        record["skipped"] = [item.model_dump() for item in extracted.skipped]
        if not record["candidates"]:
            return record
        empty = Representation().format_as_markdown(include_ids=True)
        cases = [
            {
                "admission_case_id": i,
                "observer_id": TARGET,
                "candidate_observation": content,
                "searched_conclusion_ids": [],
            }
            for i, content in enumerate(record["candidates"])
        ]
        admitted = await llm(
            usage,
            "admission",
            REPLAY_CACHE_KEY,
            deriver_messages(
                peer_id=TARGET,
                messages=messages,
                existing_conclusions=json.dumps(
                    [
                        {"admission_case_id": i, "conclusions": empty}
                        for i in range(len(cases))
                    ],
                    indent=2,
                ),
                candidate_observation=json.dumps(cases, indent=2),
                curated_memory=curated,
            ),
            AdmissionRepresentation,
        )
        record["stored"] = [decision.content for decision in admitted.explicit]
        record["stored_detail"] = [
            {
                "content": decision.content,
                "reason": decision.reason_for_entry,
                "source_message_ids": decision.source_message_ids,
            }
            for decision in admitted.explicit
        ]
        record["rejected"] = [
            {
                "candidate": record["candidates"][item.admission_case_id]
                if item.admission_case_id < len(record["candidates"])
                else None,
                "reason": item.reason,
            }
            for item in admitted.rejected
        ]
    except Exception as exc:  # one failing case is recorded, never fatal to the run
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return record


def judge_key(label: dict[str, Any], stored: list[str]) -> str:
    blob = json.dumps(
        [
            RUBRIC_VERSION,
            label.get("must", []),
            label.get("may", []),
            label.get("held", []),
            stored,
        ]
    )
    return hashlib.sha256(blob.encode()).hexdigest()


async def judge_records(
    records: list[dict[str, Any]], labels: dict[str, dict[str, Any]], usage: Usage
) -> list[dict[str, Any]]:
    memo_path = EVAL_HOME / "judge_memo.jsonl"
    memo = {row["key"]: row["verdict"] for row in read_jsonl(memo_path)}
    verdicts: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for record in records:
        label = labels.get(record["case_id"], {"must": [], "may": []})
        key = judge_key(label, record["stored"])
        if not record["stored"]:
            verdicts[record["case_id"]] = {
                "items": [],
                "missed_must": list(range(len(label.get("must", [])))),
            }
        elif key in memo:
            verdicts[record["case_id"]] = memo[key]
        else:
            pending.append({"record": record, "label": label, "key": key})

    batches = [
        pending[i : i + JUDGE_BATCH] for i in range(0, len(pending), JUDGE_BATCH)
    ]

    async def grade(
        batch: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        payload = {
            "cases": [
                {
                    "case_id": p["record"]["case_id"],
                    "must": p["label"].get("must", []),
                    "may": p["label"].get("may", []),
                    "held": p["label"].get("held", []),
                    "stored": p["record"]["stored"],
                }
                for p in batch
            ]
        }
        try:
            result = await llm(
                usage,
                "judge",
                JUDGE_CACHE_KEY,
                [
                    {"role": "system", "content": JUDGE_RUBRIC},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=1),
                    },
                ],
                JudgeBatch,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            return [(p, {"error": error}) for p in batch]
        by_id = {case.case_id: case for case in result.cases}
        graded = []
        for p in batch:
            case = by_id.get(p["record"]["case_id"])
            verdict = (
                {
                    "items": [item.model_dump() for item in case.items],
                    "missed_must": case.missed_must,
                }
                if case
                else {"error": "judge returned no verdict"}
            )
            graded.append((p, verdict))
        return graded

    new_rows = []
    for graded in await primed_gather(batches, grade):
        for p, verdict in graded:
            verdicts[p["record"]["case_id"]] = verdict
            if "error" not in verdict:
                new_rows.append({"key": p["key"], "verdict": verdict})
    if new_rows:
        write_jsonl(memo_path, read_jsonl(memo_path) + new_rows)
    return [{"case_id": r["case_id"], **verdicts[r["case_id"]]} for r in records]


def run_metrics(
    records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_id = {v["case_id"]: v for v in verdicts}
    stored = garbage = duplicates = must_total = missed = empty_ok = empty_cases = 0
    errors = 0
    categories: Counter[str] = Counter()
    for record in records:
        label = labels.get(record["case_id"], {"must": [], "may": []})
        verdict = by_id.get(record["case_id"], {})
        if record.get("error") or verdict.get("error"):
            errors += 1
            continue
        items = verdict.get("items", [])
        stored += len(record["stored"])
        for item in items:
            if item["verdict"] == "garbage":
                garbage += 1
                categories[item.get("category") or "other"] += 1
            elif item["verdict"] == "duplicate":
                duplicates += 1
        must_total += len(label.get("must", []))
        missed += len(verdict.get("missed_must", []))
        if not label.get("must"):
            empty_cases += 1
            empty_ok += int(not any(i["verdict"] == "garbage" for i in items))
    return {
        "cases": len(records),
        "stored": stored,
        "garbage": garbage,
        "duplicates": duplicates,
        "garbage_rate": round(garbage / stored, 3) if stored else 0.0,
        "must": must_total,
        "missed": missed,
        "miss_rate": round(missed / must_total, 3) if must_total else 0.0,
        "clean_nothing_cases": f"{empty_ok}/{empty_cases}",
        "garbage_by_category": dict(categories.most_common()),
        "errors": errors,
    }


def flip_metrics(
    runs: list[list[dict[str, Any]]], labels: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if len(runs) < 2:
        return {}
    found: dict[tuple[str, int], int] = Counter()
    garbage_runs: dict[str, int] = Counter()
    for verdicts in runs:
        for v in verdicts:
            label = labels.get(v["case_id"], {"must": []})
            missed = set(v.get("missed_must", []))
            for i in range(len(label.get("must", []))):
                found[(v["case_id"], i)] += int(i not in missed)
            garbage_runs[v["case_id"]] += int(
                any(item["verdict"] == "garbage" for item in v.get("items", []))
            )
    n = len(runs)
    musts = list(found)
    flipping = [key for key in musts if 0 < found[key] < n]
    garbage_flips = [cid for cid, count in garbage_runs.items() if 0 < count < n]
    return {
        "runs": n,
        "must_flip_rate": round(len(flipping) / len(musts), 3) if musts else 0.0,
        "flipping_must": [f"{cid}#{i}" for cid, i in flipping],
        "garbage_flip_cases": garbage_flips,
    }


def write_report(
    run_dir: Path, name: str, labels: dict[str, dict[str, Any]], cases: dict[str, Any]
) -> None:
    run_dirs = sorted(
        p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
    )
    usage = (
        json.loads((run_dir / "usage.json").read_text())
        if (run_dir / "usage.json").exists()
        else {}
    )
    lines = [f"# Deriver eval: {name}", ""]
    all_verdicts = []
    for split in ("tuning", "heldout"):
        split_ids = {cid for cid, case in cases.items() if case.get("split") == split}
        per_run = []
        for rd in run_dirs:
            records = [
                r for r in read_jsonl(rd / "outputs.jsonl") if r["case_id"] in split_ids
            ]
            verdicts = [
                v
                for v in read_jsonl(rd / "verdicts.jsonl")
                if v["case_id"] in split_ids
            ]
            if records:
                per_run.append((records, verdicts))
        if not per_run:
            continue
        lines += [
            f"## {split}",
            "",
            "| run | cases | stored | garbage rate | duplicates | miss rate | clean nothing | errors |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k, (records, verdicts) in enumerate(per_run, 1):
            m = run_metrics(records, verdicts, labels)
            lines.append(
                f"| {k} | {m['cases']} | {m['stored']} | {m['garbage_rate']:.1%} | {m['duplicates']} | {m['miss_rate']:.1%} | {m['clean_nothing_cases']} | {m['errors']} |"
            )
            all_verdicts.append((split, k, records, verdicts, m))
        flips = flip_metrics([v for _, v in per_run], labels)
        if flips:
            lines += [
                "",
                f"Must-item flip rate across {flips['runs']} runs: {flips['must_flip_rate']:.1%}",
            ]
        lines.append("")
    lines += [
        "## Cache",
        "",
        "| pass | calls | input tokens | output tokens | cached share | cold calls |",
        "|---|---|---|---|---|---|",
    ]
    for pass_name, u in usage.items():
        lines.append(
            f"| {pass_name} | {u['calls']} | {u['input_tokens']} | {u.get('output_tokens', '-')} | {u['cached_share']:.1%} | {u['cold_calls']} |"
        )
    lines += ["", "## Wrong items (last run per split)", ""]
    for split, k, records, verdicts, _ in all_verdicts:
        if k != max(kk for s, kk, *_ in all_verdicts if s == split):
            continue
        by_id = {v["case_id"]: v for v in verdicts}
        for record in records:
            v = by_id.get(record["case_id"], {})
            label = labels.get(record["case_id"], {"must": []})
            said = {
                m["id"]: m["content"]
                for m in cases.get(record["case_id"], {}).get("messages", [])
            }
            details = record.get("stored_detail", [])
            for item in v.get("items", []):
                if item["verdict"] == "garbage" and item["index"] < len(
                    record["stored"]
                ):
                    lines.append(
                        f"- {split} {record['case_id']} garbage/{item.get('category')}: {record['stored'][item['index']]}"
                    )
                    if item["index"] < len(details):
                        detail = details[item["index"]]
                        cited = " | ".join(
                            said.get(mid, "?")[:160].replace("\n", " ")
                            for mid in detail["source_message_ids"]
                        )
                        lines.append(f"  - stored because: {detail['reason']}")
                        lines.append(f"  - cites: {cited}")
            missed = [
                label["must"][i]
                for i in v.get("missed_must", [])
                if i < len(label.get("must", []))
            ]
            for fact in missed:
                lines.append(f"- {split} {record['case_id']} missed: {fact}")
            if missed:
                for left in record.get("skipped", []):
                    lines.append(
                        f"  - skipped: {left['statement']} (because: {left['reason']})"
                    )
                for left in record.get("rejected", []):
                    lines.append(
                        f"  - rejected: {left['candidate']} (because: {left['reason']})"
                    )
                if not record.get("skipped") and not record.get("rejected"):
                    lines.append("  - nothing weighed or rejected was reported")
            if record.get("error"):
                lines.append(f"- {split} {record['case_id']} error: {record['error']}")
            elif v.get("error"):
                lines.append(f"- {split} {record['case_id']} ungraded: {v['error']}")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "report.md").chmod(0o600)


def load_cases(split: str, subset: int | None, seed: int) -> list[dict[str, Any]]:
    cases = read_jsonl(EVAL_HOME / "cases.jsonl")
    if split != "all":
        cases = [c for c in cases if c.get("split") == split]
    cases.sort(key=lambda c: c["case_id"])
    if subset and subset < len(cases):
        cases = sorted(
            random.Random(seed).sample(cases, subset), key=lambda c: c["case_id"]
        )
    return cases


async def cmd_run(args: argparse.Namespace) -> None:
    cases = load_cases(args.split, args.subset, args.seed)
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [
            c for c in read_jsonl(EVAL_HOME / "cases.jsonl") if c["case_id"] in wanted
        ]
    labels = {row["case_id"]: row for row in read_jsonl(EVAL_HOME / "labels.jsonl")}
    curated = curated_memory_block(WORKSPACE)
    if not curated:
        raise SystemExit("USER.md block is empty; check CURATED_MEMORY_PATHS")
    run_dir = EVAL_HOME / "runs" / args.name
    usage = Usage()
    for k in range(1, args.runs + 1):
        records = await primed_gather(
            cases, lambda case: replay_case(case, curated, usage)
        )
        write_jsonl(run_dir / f"run_{k}" / "outputs.jsonl", records)
        if not args.no_judge:
            verdicts = await judge_records(records, labels, usage)
            write_jsonl(run_dir / f"run_{k}" / "verdicts.jsonl", verdicts)
        print(
            f"run {k}/{args.runs}: {len(records)} cases, {sum(len(r['stored']) for r in records)} stored"
        )
        if QUOTA_REACHED.is_set():
            print(
                "stopped: the provider's usage limit was reached; later calls were skipped"
            )
            break
    (run_dir / "usage.json").write_text(json.dumps(usage.summary(), indent=1))
    write_report(
        run_dir,
        args.name,
        labels,
        {c["case_id"]: c for c in read_jsonl(EVAL_HOME / "cases.jsonl")},
    )
    print(f"report: {run_dir / 'report.md'}")
    print(json.dumps(usage.summary(), indent=1))


async def cmd_judge(args: argparse.Namespace) -> None:
    labels = {row["case_id"]: row for row in read_jsonl(EVAL_HOME / "labels.jsonl")}
    run_dir = EVAL_HOME / "runs" / args.name
    usage = Usage()
    for rd in sorted(
        p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
    ):
        verdicts = await judge_records(read_jsonl(rd / "outputs.jsonl"), labels, usage)
        write_jsonl(rd / "verdicts.jsonl", verdicts)
    prior = (
        json.loads((run_dir / "usage.json").read_text())
        if (run_dir / "usage.json").exists()
        else {}
    )
    prior.update({k: v for k, v in usage.summary().items()})
    (run_dir / "usage.json").write_text(json.dumps(prior, indent=1))
    write_report(
        run_dir,
        args.name,
        labels,
        {c["case_id"]: c for c in read_jsonl(EVAL_HOME / "cases.jsonl")},
    )
    print(f"report: {run_dir / 'report.md'}")


async def cmd_cases(args: argparse.Namespace) -> None:
    if (EVAL_HOME / "cases.jsonl").exists() and not args.rebuild:
        raise SystemExit(
            "cases.jsonl exists; pass --rebuild to replace it and reset the draft labels"
        )
    cases, drafts = await build(args.total, args.seed)
    write_jsonl(EVAL_HOME / "cases.jsonl", cases)
    write_jsonl(EVAL_HOME / "labels_draft.jsonl", drafts)
    tags = Counter(c["tags"][0] for c in cases)
    splits = Counter(c["split"] for c in cases)
    print(
        f"{len(cases)} cases {dict(tags)} {dict(splits)}; drafts with a must label: "
        f"{sum(1 for d in drafts if d['must'])}"
    )


def shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def cmd_mirror(_: argparse.Namespace) -> None:
    prompt = deriver_system_prompt()
    block = prompt.split("<examples>", 1)[1].split("</examples>", 1)[0]
    examples = [
        line[2:].split(" → ", 1)[0]
        for line in block.splitlines()
        if line.startswith("- ")
    ]
    flagged = 0
    for case in read_jsonl(EVAL_HOME / "cases.jsonl"):
        text = " ".join(m["content"] for m in case["messages"])
        case_sh = shingles(text)
        for example in examples:
            shared = shingles(example) & case_sh
            if len(shared) >= 2:
                flagged += 1
                print(
                    f"{case['case_id']}: {len(shared)} shared phrases with example: {example[:120]}"
                )
    print(f"{flagged} example/case pairs flagged")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay labelled chat batches through the deriver and score them."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--split", choices=("tuning", "heldout", "all"), default="tuning")
    run.add_argument("--subset", type=int)
    run.add_argument(
        "--cases", help="comma-separated case ids; overrides --split and --subset"
    )
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--no-judge", action="store_true")
    judge = sub.add_parser("judge")
    judge.add_argument("--name", required=True)
    sub.add_parser("mirror")
    cases = sub.add_parser("cases")
    cases.add_argument("--total", type=int, default=150)
    cases.add_argument("--seed", type=int, default=7)
    cases.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "judge":
        asyncio.run(cmd_judge(args))
    elif args.command == "cases":
        asyncio.run(cmd_cases(args))
    else:
        cmd_mirror(args)


if __name__ == "__main__":
    main()
