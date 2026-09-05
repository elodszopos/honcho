# Conclusion reinforcement — plan

Spun off from the `v3.0.12` merge. Scaffolding: delete once implemented and `PATCHES.md`
carries the resulting rules.

## Problem

The fork removed threshold-driven dedup on purpose and kept the machinery that *carries* a
reinforcement count, but nothing generates that count any more. `times_derived` is 1 on every
row the system writes, so every consumer of "most reinforced" is ordering by a constant.

## Goal

Restore a reinforcement signal without restoring a similarity threshold. Every change to a
conclusion's count must remain a decision an agent made and recorded, in line with the
admission contract. Counting is not judgement; the doctrine objects to the second, not the
first.

## Established

| Finding | Evidence |
|---|---|
| Upstream's three dedup mechanisms all live inside `create_documents` | exact-in-batch, exact-vs-existing (reinforces), semantic via `is_rejected_duplicate` |
| None of them run | `create_documents` has no production caller; `is_rejected_duplicate` is reached only from inside it. Verified before and after the merge |
| Two config keys are inert | `DERIVER.DEDUPLICATE` and `DERIVER.DEDUPLICATE_MAX_DISTANCE`; no `deduplicate=` call site exists in `src/` |
| The deriver never supplies a count | neither `agent_tools.py` nor `crud/representation.py` sets `times_derived` on a write |
| Enrichment destroys the count | `create_observations` writes `times_derived=obs.times_derived or 1`; the predecessor's value reaches `admission_history` and stops there |
| The synthesist preserves a constant | `jobs/synthesist.md` passes MAX of merged sources; MAX over a pool of ones is one |
| The janitor cannot preserve it at all | its tools are create plus soft-delete, so absorbed duplicates take their counts with them |
| Consumers are live, not decorative | `get_most_derived_observations` in `DREAMER_TOOLS`; `_query_documents_most_derived` inside `get_working_representation` |

## Design

Three changes, none of which reintroduce a threshold.

**A — enrichment carries the count forward.** `action: "enrich"` means the agent searched,
found the memory, and judged the new evidence a refinement of it. That is a re-derivation by
definition. The replacement takes `max(previous + 1, supplied)` instead of resetting. The
store records a decision already made and justified; it makes none of its own.

**C — a third admission verb, `reinforce`.** `target_id`, no content: bump the count, append an
`admission_history` entry carrying `reason_for_entry`, `search_query` and
`searched_conclusion_ids`. Serves two callers the current verbs fail:

- the janitor folds absorbed duplicates into the survivor before soft-deleting them, and
  reports it in the same audit row as every other action;
- the deriver answers "found this again, nothing to add" without either writing a twin or
  forcing an enrich that rewrites text and burns an embedding for no change.

**B — exact-content collision reinforces instead of inserting.** An admitted `create` whose
normalized content equals a live conclusion in the same collection increments rather than
inserting a twin. Identity, not similarity: no threshold, no ranking, nothing discarded. The
safety net for a pre-write search that missed on session boundary or embedding drift.

## Open questions

| # | Question | Notes |
|---|---|---|
| 1 | MAX or SUM when the synthesist merges | SUM is truthful for the explicit-duplicate bucket, inflates for the inseparable-fragments bucket. MAX is the safe default and only starts mattering once A/B/C create spread |
| 2 | Does `reinforce` count against the janitor's 20-or-10% cap | It removes nothing, but it is a mutation and the cap exists to bound autonomous change |
| 3 | Does the deriver's admission prompt need to learn `reinforce` | A alone covers the refinement case; C is what avoids pointless rewrites. Prompt work is behavioural surface and needs its own verdict |
| 4 | Do the four dedup fields on `RepresentationCompletedEvent` get repurposed | They default to 0 and are currently unpopulated. B produces something close to `exact_dup_existing_count` |
| 5 | Backfill or start fresh | Existing rows carry predecessor counts in `admission_history`. Reconstructing from it is possible; starting everything at its current value is simpler and loses only history that is already flat |
| 6 | What does the live distribution actually look like | A read-only count of `times_derived` values would show whether anything above 1 survives from before the admission rewrite. Not yet run |

## Touch points

| Change | Files |
|---|---|
| A | `src/crud/document.py` (`create_observations`, enrich branch) |
| B | `src/crud/document.py` (`create_observations`, create branch) |
| C | `src/schemas/api.py` (`action` literal), `src/crud/document.py`, `mcp/src/tools/conclusions.ts`, `sdks/python/src/honcho/conclusions.py`, `sdks/typescript/src/conclusions.ts` |
| Doctrine | `~/.hermes/skills/productivity/memory-honcho-steward/jobs/janitor.md`, `jobs/synthesist.md` |

## Out of scope

Restoring `create_documents` to a production path, semantic dedup by cosine distance, and any
mechanism that merges or discards a memory without a recorded agent decision.
