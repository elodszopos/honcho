# The memory ledger — plan

One unit across three roots: the Honcho server and its SDKs, the Hermes Honcho plugin, and the
steward doctrine. Scaffolding: delete once implemented and `PATCHES.md` carries the rules.

## Problem

Every write to a conclusion is accountable. No removal is.

| Mutation | Recorded today |
|---|---|
| create | who, why, what it searched, what it saw, trace, model |
| enrich | all of the above, plus the predecessor's full envelope in `admission_history` |
| removal | a timestamp |

The fork's thesis is that no memory changes without a recorded agent decision. Half the
mutations are exempt, and three consequences follow:

- A conclusion leaves with no record of why or what carries it now. The janitor writes its audit
  row into a markdown report, outside Honcho, unreadable by anything that queries the memory.
- Reinforcement earned by an absorbed duplicate is not destroyed so much as never transferred.
  The survivor is never told it absorbed anything.
- In pgvector mode a reaper hard-deletes the row five minutes later, so even the timestamp goes.

Separately, `create_observations` writes `times_derived = obs.times_derived or 1` on every path,
so an enrichment that omits the value silently resets the row. That has happened once, from a
steward write, taking a conclusion from 6 to 1.

## Goal

One accounting model for every mutation. A memory's story is readable from the memory itself:
why it exists, what it used to say, what it absorbed, and — if it is gone — why. Reinforcement
stops being a number a caller must remember and becomes a consequence of recording the decision.

## The invariants

| # | Invariant | Enforced by |
|---|---|---|
| 1 | A reinforcement count is never lowered | the enrich branch takes `max(previous + 1, supplied)`, reading the predecessor under a row lock |
| 2 | No conclusion loses its `deleted_at` guard without a removal envelope | the envelope is a required argument of the only function that writes `deleted_at` |
| 3 | A removed conclusion's row lives as long as its session does | nothing reaps; session and workspace deletion remain terminal |
| 4 | An absorbed conclusion's count moves to its survivor | the server does the arithmetic |
| 5 | Removed conclusions never reach a deriving agent | `query_documents` and the representation reads stay live-only, unconditionally |

## What the codebase settles

**The reaper has no purpose here.** Upstream's `cleanup_soft_deleted_documents` drops a vector
from an external store and then the row. This deployment runs `VECTOR_STORE_TYPE=pgvector`,
taking `_cleanup_soft_deleted_documents_pgvector` in `src/reconciler/sync_vectors.py` — whose own
comment reads *"Hard delete directly (no vector store cleanup needed in pgvector mode)."* It
inherited the shape without the reason, and it removed the 34 enrich predecessors this pool has
produced.

**Removal is a state transition seven subsystems perform**, each writing `deleted_at` directly:

| Path | Site | Why it removes |
|---|---|---|
| conclusions API | `crud.delete_document_by_id` ← `routers/conclusions.py` | an operator or agent decided |
| agent delete tool | `crud.delete_documents` ← `utils/agent_tools.py` | an agent decided |
| single-document delete | `crud.delete_document` | an operator decided |
| queued observation delete | `crud.delete_document_by_id` ← `deriver/consumer.py` | a delete was enqueued |
| enrichment | `create_observations`, retiring its target | a replacement superseded it |
| semantic replacement | `_apply_document_row_updates`, the `replace` op | upstream's dead path, kept honest |
| scope removal | `deriver/scope_backfill.py`, two sites | the session left the scope |

Two more destroy rows outright and stay that way by ruling: `crud/session.py` batch-deletes every
Document for a session, and `crud/workspace.py` deletes them wholesale. Deleting a session is
meant to erase what it taught the system.

**The steward does not use the MCP server.** It calls the plugin's `honcho_conclusions` tool,
which reaches the Python SDK and the HTTP API directly. That tool already carries `times_derived`
on create — which is how 15 live rows hold counts — and its `delete` takes an id and nothing
else. Any contract designed against the MCP tool alone reaches nothing the janitor does.

## Design

### One chokepoint

`soft_delete_documents(db, workspace_name, document_ids, *, removal)` in `src/crud/document.py`
becomes the only code that writes `deleted_at`. The envelope is a required keyword argument, so a
path cannot omit it and compile. The three existing entry points are near-identical filter
variants over one operation and collapse into thin callers.

### The removal envelope

Stored at `internal_metadata.removal` on the removed row.

| Field | Required | Meaning |
|---|---|---|
| `category` | yes | one of the categories below |
| `reason` | yes | the audit row, prose, non-whitespace |
| `absorbed_into` | only for `duplicate_absorbed` | the surviving conclusion's id |
| `entry_origin` | yes | the admission envelope's enum, or the subsystem for a system removal |
| `agent_trace_id` | agent removals only | absent on system removals — never fabricated |
| `agent_model` | agent removals only | absent on system removals — never fabricated |
| `removed_at` | server | timestamp |

`absorbed_into` is optional by design. A conclusion that was never valid is removed outright;
inventing a link to satisfy the schema would corrupt the ledger worse than omitting one.

### Categories

Granular because the filter is the product: *show everything dropped as misderived last month* is
the question that finds pipeline defects.

**Agent-chosen:**

| Category | Meaning | Link |
|---|---|---|
| `duplicate_absorbed` | the same memory, folded into a survivor | required |
| `superseded` | was true, reality changed | optional |
| `contradicted` | was never true, evidence now shows it | optional |
| `misderived` | extraction defect — misread, wrong peer, invented detail | no |
| `out_of_scope` | valid content, wrong home; carrier verified | optional |
| `transient` | true in the moment, never durable memory | no |
| `low_value` | true and durable but not worth carrying | no |

**System-recorded**, never chosen by an agent, no fabricated actor:

| Category | Written by |
|---|---|
| `superseded_by_enrichment` | `create_observations` retiring an enrich target |
| `semantic_dup_replaced` | `_apply_document_row_updates`, the `replace` op |
| `scope_removed` | scope removal in `scope_backfill.py` |
| `queued_delete` | the enqueued observation delete in `deriver/consumer.py` |

### Absorption sums, and there is no second arithmetic

An earlier draft split absorption into a summing `duplicate_absorbed` and a max-taking
`fragment_merged`. That is order-dependent — starting at 1, sum-2-then-max-5 gives 5 while
max-5-then-sum-2 gives 7 — so the same three absorptions produce different totals depending on
call order. The whole argument for moving the arithmetic into the server was that it stops
depending on a caller behaving correctly, and an order-dependent rule fails that on its own terms.

Absorption always sums. Fragments of one fact over-count mildly; `times_derived` is a ranking
signal, and a consolidated memory ranking higher is not wrong.

### The absorb transfer

`_apply_document_row_updates` already locks target rows in sorted id order with
`populate_existing` so concurrent increments are visible. It is extended with a `sum` op rather
than reimplemented. Under that lock, `absorbed_into` causes two writes to the survivor:

1. `times_derived += the removed row's count`.
2. The removed conclusion's envelope is appended to `internal_metadata.absorbed`.

Self-absorption is rejected. The survivor must be live and in the same collection.

**The snapshot** carries `document_id`, `content`, `times_derived`, the source's own `admission`,
its `session_name`, `category`, `reason`, `absorbed_at`, `agent_trace_id`, `agent_model`, and
`survivor_count_before` / `survivor_count_after`. Recording both sides of the count makes the
ledger auditable without reconstructing anything.

It is a snapshot, not a tree. A conclusion that absorbed something which had itself absorbed
something does not carry the inner chain. The plan does not claim otherwise.

`session_name` is recorded because a collection spans sessions and the janitor legitimately
absorbs across them. Requiring same-session absorption would block its primary job; recording
which session the evidence came from keeps the transfer traceable instead.

### Consolidation counts as one derivation

A synthesist that creates a survivor and absorbs three sources of counts 3, 2 and 2 lands on 8,
not 7 — the survivor's own 1 is the consolidation act, which is itself a derivation. That is the
ruling, not a defect. The doctrine consequence is hard: **the synthesist supplies no
`times_derived` on the create.** Preloading the count and then absorbing the same source counts
it twice.

### The metadata constructor is one owner

`create_observations` rebuilds `internal_metadata` from scratch and carries forward only
`admission_history`, so the first enrichment after an absorption destroys `absorbed`. One
function owns building a revision's metadata and carries forward every ledger key —
`admission_history` and `absorbed` — with the predecessor's own snapshot appended.

### The vector stays

Nulling `embedding` on removal would break scope resurrection: `scope_backfill.py` un-deletes
rows with `deleted_at = None, sync_state = "pending"`, and in pgvector mode nothing re-embeds a
pending row. A resurrected copy would return with a null vector and drop out of semantic search
with no error. Every read filters `deleted_at IS NULL`, so a removed row's vector is inert where
it sits.

For external-store mode the vector is deleted but the row now survives, so cleanup needs a
completed state — otherwise the eligibility query re-selects the same rows forever and the
reconciler reports permanent work.

### Resurrection is category-scoped

Scope backfill restores only rows whose removal category is `scope_removed`. A copy the janitor
removed as `contradicted` must not come back when the session rejoins a scope. Before the ledger
this could not happen, because everything was reaped within five minutes.

### Two reference contracts, not one

`_validate_admission_references` conflates them today. They separate:

| Reference | Rule |
|---|---|
| `searched_conclusion_ids` — proof of look-before-write | may name retained rows; a search receipt is historical |
| `target_id` — the enrichment target | must be live |
| `source_ids` — evidence supporting a derived conclusion | must be live |

### The symmetry, complete

| Event | Recorded at | Answers |
|---|---|---|
| create | `admission` | why this exists, what was searched, who decided |
| enrich | `admission_history` | what this memory used to say |
| absorb | `absorbed`, on the survivor | what it swallowed, what each brought, and the count either side |
| removal | `removal`, on the row | why it went, which kind, who decided |
| — | `times_derived` | the running total of derivations and absorptions |

### Read surface

Routes are POST, not GET.

| Surface | Change |
|---|---|
| `POST /conclusions/list` | `include_deleted` filter, default false |
| `POST /conclusions/query` (semantic) | **unchanged.** Never searches removed memories |
| `GET /conclusions/{id}/lineage` | new — the chain for one conclusion, its own detail model |
| MCP | `include_deleted` on list, `times_derived` on create, **created ids in the create result**, new lineage tool |
| Python + TypeScript SDKs | mirror all of it, plus the new `delete` signature |

**Why the semantic route is exempt.** `get_documents_with_filters`, behind list, has exactly one
caller. `query_documents`, behind semantic query, has six — the agent search tool twice,
`_semantic_dup_decision`, and both the representation and admission candidate searches. A flag
there is one default away from feeding a removed memory to a deriving agent. Searching the
graveyard semantically buys nothing browsing and lineage do not.

**Lineage reports** the conclusion as it stands or its removal record, its `admission`, every
prior formulation newest-first, and every absorption with the count each brought and the
survivor's count either side. It does not decompose `times_derived` into derivations versus
absorptions: invariant 1 uses `max`, so an enrichment's contribution is not additive.

**List payloads shrink.** `Conclusion` returns `admission_history` on every list call, which
grows monotonically under retention. `admission_history` and `absorbed` move out of the list
model and are served only by lineage.

### Zero migrations

`internal_metadata` is JSONB and no column changes type or nullability, so nothing here adds an
Alembic revision — which keeps the fork free of one that every future upstream migration would
have to sequence around.

One deferral, anchored as a TODO at the list query: retention means the table only grows while
every live read filters `deleted_at IS NULL`. At this pool size that is free. The partial index
on `(workspace_name, observer, observed) WHERE deleted_at IS NULL` is the fix if it stops being.

## The client surface

The agent never touches the server directly. Every steward action goes through the plugin, and
the removal decision has to survive four layers that today drop everything but an id.

| Layer | File | Change |
|---|---|---|
| tool schema | `hermes_memory.py`, `CONCLUSIONS_SCHEMA` | `removal_category`, `reason_for_removal`, `absorbed_into` params; new `lineage` operation; description teaches that a removal without a reason is refused |
| handler | `hermes_memory.py`, `handle_conclusions_tool` | validate category and reason on `delete` the way `create` already validates admission; route `lineage`; forward the envelope |
| audit wrapper | `hermes_memory.py`, `_audited_delete` | record the category and `absorbed_into` in the local log alongside the id |
| session wrapper | `session.py`, `delete_conclusion` | take and forward the envelope; the "Use only for PII removal" docstring is wrong and goes |
| CLI | `cli.py` | `--delete` and `--replace` both require a category and reason. `--replace` is a delete plus a create and must not lose the count — it becomes an enrich |

**Two histories, named apart.** `history` stays what it is: the local, 60-day, metadata-only
mutation log, which records this instance's actions including failures the server never saw.
`lineage` is the new operation reading the server ledger for one conclusion — permanent,
content-bearing, and the same data any client sees. Neither replaces the other.

## Build order

One release. Nothing here ships alone, because making `category` required breaks the janitor the
moment the server has it and the plugin does not. Internal order only:

| Step | Work |
|---|---|
| 1 | Repoint the five stale plan references; drop the anchors for the two dropped designs |
| 2 | Enrich never lowers a count — predecessor read under `FOR UPDATE`, `max(previous + 1, supplied)` |
| 3 | The chokepoint, the envelope, the categories, and every one of the seven callers |
| 4 | Absorb transfer, the metadata constructor, resurrection scoping, reference contracts, reaper dropped |
| 5 | Server read surface: list flag, lineage endpoint and detail model |
| 6 | SDKs, MCP tool parity, created ids |
| 7 | The plugin: schema, handler, audit wrapper, session wrapper, CLI |
| 8 | Steward doctrine |

Step 2 is the only piece that is correct in isolation and stops an active loss. Say the word if
you want it landed ahead of the rest.

Roughly five days.

## Touch points

| Root | Files |
|---|---|
| honcho — server | `src/crud/document.py`, `src/schemas/api.py`, `src/routers/conclusions.py`, `src/deriver/consumer.py`, `src/deriver/scope_backfill.py`, `src/reconciler/sync_vectors.py`, `src/utils/agent_tools.py` |
| honcho — clients | `mcp/src/tools/conclusions.ts`, `sdks/python/src/honcho/{conclusions,aio,api_types}.py`, `sdks/typescript/src/{conclusions.ts,types/api.ts}`, `sdks/typescript/__tests__/conclusions.test.ts` |
| hermes-agent | `plugins/memory/honcho/{hermes_memory,session,cli,audit}.py` |
| hermes-home | `skills/productivity/memory-honcho-steward/jobs/{janitor,synthesist}.md` |

## Doctrine changes

| Job | Change |
|---|---|
| janitor | every removal supplies a category and a reason — the audit row moves into Honcho and the report references the same text. Dedup uses `duplicate_absorbed` with `absorbed_into`, so the count moves without the janitor computing anything |
| synthesist | stops computing merge arithmetic and **stops supplying `times_derived` on a consolidation create**. It creates the survivor, then removes each source as `duplicate_absorbed`; the server sums |

## Tests

`tests/test_fork_held_values.py` declares autouse fixtures that are empty stubs — a
source-assertion file with no database. Behaviour goes in the CRUD, route, reconciler, scope and
SDK suites that have real fixtures. A post-merge review already found three held values asserted
by substring checks that passed while testing nothing; these must be able to fail.

| Assertion | Home |
|---|---|
| no module writes `deleted_at` outside the chokepoint | held values |
| `get_documents_with_filters` is the only read builder taking `include_deleted` | held values |
| enrich carries the predecessor's count forward, raised by one | crud integration |
| a supplied count below the predecessor's does not lower it | crud integration |
| a concurrent absorb during an enrich does not lose the increment | crud integration |
| absorb sums; self-absorption and cross-collection absorption are rejected | crud integration |
| absorb → enrich → lineage still shows the absorption | crud integration |
| a removal without a reason or category is rejected | route integration |
| system removals carry no fabricated `agent_trace_id` | crud integration |
| a removed row survives a reconciler pass with its vector | reconciler integration |
| external-store cleanup does not re-select a completed row | reconciler integration |
| a resurrected scope copy still matches semantic search, and a `contradicted` copy is not resurrected | scope integration |
| removed rows never appear in the representation or admission candidates | crud integration |
| a search receipt may name a retained row; an enrich target may not | crud integration |
| the plugin refuses a delete without a category, and forwards the envelope it is given | plugin tests |
| the CLI `--replace` path preserves the count | plugin tests |

## Measured, 2026-09-06

84 live conclusions, all explicit. 25 carry `times_derived` above 1, peaking at 25.

| Writer / action | Rows | Reinforced | Peak |
|---|---|---|---|
| steward, enrich | 33 | 15 | 16 |
| deriver, create | 30 | 0 | 1 |
| deriver, enrich | 1 | 0 | 1 |
| legacy, pre-admission | 19 | 10 | 25 |

The steward is the only generator today, and it generates by remembering to pass a number. It
forgot once, out of 34 enrichments. The deriver has enriched once because until the admission
block began printing conclusion ids it could not reliably pick a target and defaulted to
`create`; that rate rises from here, and every one of those enrichments lands on invariant 1.

## Dropped

| Item | Reason |
|---|---|
| An exact-content collision reinforces instead of inserting | zero exact-content collisions across all 84 rows, and zero even ignoring the session key. It also fights `_dedup_key`'s session-purity rule |
| A third admission verb, `reinforce` | both callers are gone. The dreamer is permanently off and the janitor reaches the same outcome through absorption, which also records why |
| `fragment_merged` | order-dependent arithmetic, which defeats the reason for moving arithmetic into the server |
| Content erasure and a privacy category | single-tenant, self-hosted; no privacy boundary to serve. The by-value trace design defeats scrubbing anyway, and the source message keeps the text |
| Semantic search over removed conclusions | the only surface needing `include_deleted` on `query_documents`, whose five internal callers make it the one place invariant 5 breaks quietly |
| A retention job, config key, or tombstone table | rows are permanent for the life of their session |

## Parked

| Item | Why it waits |
|---|---|
| Removing a `misderived` or `contradicted` conclusion does not retire its deductive descendants | the dreamer is permanently off and all 84 conclusions are explicit. There are no descendants |
| `internal_metadata["message_created_at"]` has six readers and no writer, so timestamps fall back to row-insert time | a new field on `ConclusionCreate` and its SDK ripple. Insert time tracks message time closely on a live deriver |
| `_latest_source_timestamp` runs a full document query per dreamer write and always returns `None` | same fix, same cost. A wasted round-trip, not a wrong answer |
| `DREAMER_TOOLS` is defined and referenced nowhere | upstream dead code, and the dreamer is off |
| `ConclusionCreateParams` is defined twice in each SDK's `api_types` with every admission field optional | dormant — neither copy is exported. Resolve while changing those contracts if it is cheap |
| Two admission cases targeting the same `target_id` report "Enrichment target changed during admission" | only the message misleads |
| `track_deriver_input_tokens` counts the extraction call only | `DeriverComponents` has no bucket for it; folding into `PROMPT` inflates the scaffold metric, which a test caught |

The previously parked "`create_observations` filters blank content after the schema rejected it"
is struck: no such filter exists in the current code.

## Out of scope

The same ledger for messages and peer cards. Restoring `create_documents` to a production path.
Semantic dedup by cosine distance. Any mechanism that merges or discards a memory without a
recorded agent decision.
