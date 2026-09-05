# Conclusion reinforcement — plan

Spun off from the `v3.0.12` merge. Scaffolding: delete once implemented and `PATCHES.md`
carries the resulting rules.

## Problem

The fork removed threshold-driven dedup on purpose and kept the machinery that *carries* a
reinforcement count, but nothing generates that count any more — and the paths that rewrite a
conclusion drop whatever it had already earned. Thirty percent of the live pool still carries
reinforcement from before the rewrite; every enrichment of one of those rows spends it.

## Goal

Restore a reinforcement signal without restoring a similarity threshold, and stop spending the
one that is left. Every change to a conclusion's count must remain a decision an agent made and
recorded, in line with the admission contract. Counting is not judgement; the doctrine objects
to the second, not the first.

## The invariant

**A reinforcement count is never lowered.** Every path that rewrites, replaces or absorbs a
conclusion carries forward what that conclusion already earned:

| Path | Carries forward by |
|---|---|
| enrich | `max(previous + 1, supplied)` — A |
| exact-content collision | incrementing the row that exists rather than inserting beside it — B |
| janitor absorbing a duplicate | `reinforce` on the survivor before the soft-delete — C |
| synthesist merge | sum for the duplicate bucket, max for fragments — decision 1 |

The root hazard is the default. `create_observations` writes `times_derived = obs.times_derived
or 1`, so any writer that does not supply a value silently resets the row to one. On these paths
supplying it is not an optimisation; omitting it is data loss.

## Established

| Finding | Evidence |
|---|---|
| Upstream's three dedup mechanisms all live inside `create_documents` | exact-in-batch, exact-vs-existing (reinforces), semantic via `is_rejected_duplicate` |
| None of them run | `create_documents` has no production caller, and `is_rejected_duplicate` now has none either — both it and `create_documents` call `_semantic_dup_decision`, which nothing else reaches |
| Two config keys are inert | `DERIVER.DEDUPLICATE` and `DERIVER.DEDUPLICATE_MAX_DISTANCE`; no `deduplicate=` call site exists in `src/` |
| The deriver never supplies a count | neither `agent_tools.py` nor `crud/representation.py` sets `times_derived` on a write |
| Enrichment destroys the count | `create_observations` writes `times_derived=obs.times_derived or 1`; the predecessor's value reaches `admission_history` and stops there |
| The synthesist carries what it is handed | `jobs/synthesist.md` passes MAX of merged sources, and those values are real; nothing replenishes them |
| The janitor cannot preserve it at all | its tools are create plus soft-delete, so absorbed duplicates take their counts with them |
| One consumer is live | `_query_documents_most_derived`, reached from `_get_working_representation_internal` when a caller passes `include_most_frequent=True` (peer and session routes). `get_most_derived_observations` sits only in `DREAMER_TOOLS`, which no loadout references — upstream dead code, not a fork regression |

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

## Read path — done

The value was already accepted, persisted and returned by Postgres; only the read models
dropped it. `Conclusion`, both SDK response models and both SDK conclusion classes now carry
it, and the MCP list and query tools include it in their output.

`jobs/synthesist.md` instructs the synthesist to fold into the more-reinforced conclusion by
reading `times_derived` from the injected pool listing. That worked only because
`~/.hermes/scripts/steward/shared.py` falls back to querying Postgres directly when the SDK omits the
field. The read path now supplies it, so that fallback short-circuits on its own and every
other consumer — the MCP tools, the TypeScript SDK, the agent's conclusion tool — gets the
value without a database connection of its own.

## Decisions

| # | Question | Ruling |
|---|---|---|
| 1 | Merge arithmetic | SUM for the explicit-duplicate bucket, MAX for inseparable fragments. The synthesist already names which bucket each delete falls in |
| 2 | Does `reinforce` count against the janitor's cap | Exempt, because it destroys nothing, but named in the audit row of the delete it accompanies |
| 3 | Does the deriver's admission prompt learn `reinforce` | Yes. Enrich-with-unchanged-content is a duplicate row in all but name, and costs a rewrite plus a re-embed |
| 4 | Backfill | Start fresh. Every row keeps its current value and counts accumulate from here |
| 5 | The four dedup counters on `RepresentationCompletedEvent` | B earns `exact_dup_existing_count`, so populate that one. The other three describe threshold dedup this fork refuses, and 0 is the correct reading rather than a gap. Upstream's event schema is left alone |

## The live pool, measured 2026-09-06

83 live conclusions. 25 of them carry `times_derived` above 1, peaking at 25, earned before the
admission rewrite retired the path that incremented it. The pool is not flat, so most-derived
recall does rank meaningfully today and the synthesist's MAX rule has real values to carry.

That reinforcement is being spent, not merely frozen. Enrichment writes `times_derived` from the
incoming conclusion, the deriver supplies none, so enriching a reinforced conclusion resets it to
1 and leaves the earned count in `admission_history` where nothing reads it. One row has already
lost its count this way.

Plan A is therefore the first thing to build: it stops an active loss before it restores anything.
The backfill ruling stands — rows keep the values they have, and nothing is reconstructed.

## Touch points

| Change | Files |
|---|---|
| A | `src/crud/document.py` (`create_observations`, enrich branch) |
| B | `src/crud/document.py` (`create_observations`, create branch) |
| C | `src/schemas/api.py` (`action` literal), `src/crud/document.py`, `mcp/src/tools/conclusions.ts`, `sdks/python/src/honcho/conclusions.py`, `sdks/typescript/src/conclusions.ts` |
| Doctrine | `~/.hermes/skills/productivity/memory-honcho-steward/jobs/janitor.md`, `jobs/synthesist.md` |

## Raised by review, parked until the plan is executed

None of these is load-bearing. Each was found in the post-merge review and judged not worth
doing now; bring them up once A, B and C have landed.

| Item | Why it waits |
|---|---|
| `internal_metadata["message_created_at"]` has six readers and no writer, so conclusion timestamps fall back to row-insert time and the dreamer's backdating never fires | Writing it means a new field on `ConclusionCreate` and its SDK ripple. Insert time tracks message time closely on a live deriver, so the visible error is small |
| `_latest_source_timestamp` runs a full document query on every dreamer write and always returns `None`, because of the above | Same fix, same cost. It is a wasted round-trip, not a wrong answer |
| A dreamer that deletes a searched candidate before writing loses the whole batch — `_validate_admission_references` requires every searched id to still be live | Needs a specific search-delete-write order in one run. Fix is to check liveness for `target_id` only, which is what the concurrency guard actually needs |
| `DREAMER_TOOLS` is defined and referenced nowhere | Upstream dead code. Either wire it into a loadout the dreamer uses or drop it; both are upstream's call more than ours |
| `ConclusionCreateParams` is defined a second time in each SDK's `api_types` with every admission field optional | Dormant — neither copy is exported or used. It documents a wire contract the server rejects |
| Two admission cases targeting the same `target_id` report "Enrichment target changed during admission", which is not what happened | The store fail-closes correctly; only the message misleads |
| `create_observations` filters blank content after the schema has already rejected it | Unreachable defensive code. The schema-level guarantee is tested |
| `track_deriver_input_tokens` counts the extraction call only, so `deriver_tokens_processed` understates input by roughly the admission call on any extracting batch | `DeriverComponents` has no bucket for it, and adding one means editing upstream's bounded-label init. Attempted and reverted: folding it into `PROMPT` inflates the scaffold metric, which a test correctly caught |

## Out of scope

Restoring `create_documents` to a production path, semantic dedup by cosine distance, and any
mechanism that merges or discards a memory without a recorded agent decision.
