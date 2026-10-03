# Deriver eval

Replays real chat batches through the deriver and grades what it would store against labelled answers, so a
prompt change is measured before a rebuild ships it.

## Capabilities

Every command runs from the repo root as `bash scripts/deriver_eval/eval.sh <command>`.

| Task | Command |
|---|---|
| Measure the working tree's prompt | `run --name <run>`; `--split tuning` (default), `heldout` or `all`; `--runs <k>` repeats it; `--cases c001,c002` replays named cases only |
| Re-grade a finished run after a label change | `judge --name <run>`; no deriver calls |
| Check prompt examples against the cases | `mirror`; flags any example sharing phrases with a case |
| Build cases from archived deriver batches | `cases`; writes `cases.jsonl` and draft labels, refuses to overwrite without `--rebuild` |

## Contracts

- A replay is the production call: the working tree's prompt builders, the deriver's configured model and response
  formats, and the curated files in production order. The running containers are not involved.
- Admission sees no existing conclusions, so a duplicate of something already in Honcho is not measured; a duplicate
  of a curated file is.
- Data lives in `~/.honcho/deriver-eval/` (`DERIVER_EVAL_HOME` overrides it), owner-only and never in git: the cases
  hold the user's real chats.

| File | Holds |
|---|---|
| `cases.jsonl` | `case_id`, `split`, `session`, `range`, `messages` (`id`, `peer`, `time`, `content`), `tags`; `invented` tags a hand-written case |
| `labels.jsonl` | per case: `must` (should be stored), `may` (fine either way), `held` (a curated file already states it), `ruled_by` (`user`, `draft` or `invented`) |
| `labels_draft.jsonl` | the draft labels `cases` writes; `labels.jsonl` is maintained by hand |
| `judge_memo.jsonl` | verdicts keyed on the rubric version, the case's labels and what it stored |
| `runs/<run>/run_<k>/outputs.jsonl` | per case: what was stored with its reason and cited messages, what extraction left out and admission rejected, with reasons |
| `runs/<run>/run_<k>/verdicts.jsonl`, `report.md`, `usage.json` | the judge's verdicts, the report, calls and tokens per pass |

## Reading a report

- **Garbage rate**: stored items that match no label, over everything stored. Each carries a category.
- **Duplicates**: stored items matching a `held` fact or another stored item; counted apart from garbage.
- **Miss rate**: `must` facts nothing stored covers, over all `must` facts.
- **Clean nothing**: cases with no `must` label that stored no garbage.
- **Flip rate**: `must` facts found in some runs and missed in others.
- **Wrong items** (last run only): each garbage item with why the deriver stored it and the messages it cites; each
  miss with what extraction left out or admission rejected, and why.
- **Cache**: cached share per pass. The judge's prefix is too short to cache.

## Rules

- Read misses before garbage. The janitor removes garbage; nothing recovers a fact that was never stored.
- Run at least twice and compare per case. Two runs of the same prompt differ by more than one rule change moves, so a
  single rate is not a measurement.
- Tune on `tuning` and confirm on `heldout`. The held-out cases were read during the last tuning round and no longer
  count as unseen.
- Change a rule as an outcome, never as a list of the failing cases. Prompt examples are invented; run `mirror` after
  editing one.
- Only the user rules labels; a `draft` label is a guess.
- After a label change, run `judge`, not a new run: only cases whose labels or stored items changed are re-graded.
  Changing the rubric needs a new `RUBRIC_VERSION`, which re-grades everything.
- `cases --rebuild` assigns new case ids, so existing labels and runs no longer line up with the cases.
- A run spends real model calls: extraction for every case, admission for each case that extracted something, the
  judge for each case that stored something.
- When the provider reports its usage limit, no further call is made: the remaining cases of that run are recorded as
  errors, later runs are skipped, and the report covers what ran.
- A measured prompt reaches production only through `~/.hermes/scripts/honcho-rebuild.sh`.

## Operating details

- Code: `scripts/deriver_eval/` — `eval.sh` (environment), `deriver_eval.py` (commands), `build_cases.py` (case
  building).
- `eval.sh` loads `.env`, points provider URLs and the database at loopback, and lists the curated files from the
  Hermes home. Keep that list matching the deployed `CURATED_MEMORY_PATHS`.
- Cases come from the batch-start lines in the archived deriver logs under `~/.hermes/logs/honcho/` and the messages
  in the Honcho database; every case is a batch of the `hermes` workspace observing the user's peer.
