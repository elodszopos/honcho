# The memory ledger

Every change to a conclusion is recorded on the conclusion itself. Nothing leaves the pool
without a reason, and a memory's whole history is readable from the memory.

A conclusion carries four records. `admission` says why it exists. `admission_history` holds
every earlier formulation of it. `absorbed` holds every conclusion folded into it, with the
count each brought. `removal` says why it was retired, and is absent while it is live.

## Lifecycle

| Event | What happens | Where it is recorded |
|---|---|---|
| create | a new row, count starts at one | `admission` |
| enrich | a replacement row is written and its predecessor retired in one transaction | predecessor's envelope appended to the replacement's `admission_history` |
| absorb | the retired conclusion's count moves onto a named survivor | snapshot appended to the survivor's `absorbed` |
| retire | the row stops being live but is kept | `removal` on the row |

A retired row is permanent and keeps its vector. Nothing reaps it. Session and workspace
deletion remain terminal — the ledger lives as long as its session does.

## Removal categories

Every removal names one. Seven are chosen by an agent:

| Category | Meaning | Requires |
|---|---|---|
| `duplicate_absorbed` | the same memory, folded into a survivor | `absorbed_into` |
| `superseded` | was true, reality changed | |
| `contradicted` | was never true | |
| `misderived` | an extraction defect — misread, wrong peer, invented detail | |
| `out_of_scope` | valid content, wrong home, carrier verified | |
| `transient` | true in the moment, never durable memory | |
| `low_value` | true and durable but not worth carrying | |

Four are written only by Honcho itself, carry `entry_origin: system`, and refuse agent
attribution rather than fabricate it: `superseded_by_enrichment`, `semantic_dup_replaced`,
`scope_removed`, `queued_delete`. The public route rejects a caller that claims one.

## Reinforcement

`times_derived` ranks most-derived recall. It only ever rises.

| Path | Effect on the count |
|---|---|
| enrich | `max(previous + 1, supplied)` — an admitted enrich is a re-derivation |
| `duplicate_absorbed` | the retired row's count is added to its survivor's |
| every other retirement | none; the count retires with the row |

Callers do not compute merge arithmetic. A consolidation creates its survivor without supplying
a count, then retires each source as `duplicate_absorbed`; supplying a count as well counts
those sources twice.

## Reading it

| Question | Surface |
|---|---|
| what does this peer know | conclusion list, or semantic query |
| what was retired here and why | conclusion list with `include_deleted` |
| what happened to this one memory | the lineage route for that conclusion |
| what did this Hermes instance do, including failures | the plugin's local mutation log |

Lineage returns prior formulations newest-first, every absorption with the survivor's count
either side, and the removal record if there is one. It is the only way to read a retired
conclusion — search never returns one.

## Invariants

| Invariant | Enforced by |
|---|---|
| a reinforcement count is never lowered | the enrich branch, with the batch's predecessors locked in one id-ordered statement |
| no conclusion loses its live status without an envelope | the envelope is a required argument of the only function that writes `deleted_at` |
| a retired conclusion never reaches a deriving agent | `include_deleted` exists on the list builder alone |
| an absorbed count is never lost | the transfer happens inside the retirement, under the survivor's lock |
| a revision inherits the whole ledger | one metadata owner starts from the predecessor and overwrites only what admission owns |

Held-value tests enforce the first three by reading source: one scans for any other writer of
`deleted_at`, one inventories every document query so a new one fails until it gets a verdict on
retired rows, and one pins the batch lock.

## Client surfaces

A contract change has to reach all five. The CLI inside the honcho repo is the one that gets
missed, because it sits away from its siblings.

| Surface | Path |
|---|---|
| MCP server | `mcp/src/tools/conclusions.ts` |
| Python SDK | `sdks/python/src/honcho/conclusions.py` |
| TypeScript SDK | `sdks/typescript/src/conclusions.ts` |
| standalone CLI | `honcho-cli/src/honcho_cli/commands/conclusion.py` |
| Hermes plugin | `~/.hermes/hermes-agent/plugins/memory/honcho/` |

Steward doctrine in `~/.hermes/skills/productivity/memory-honcho-steward/jobs/` is the sixth
place a contract lands: it tells the janitor and synthesist which category to use.

## Deliberately absent

| Not built | Why |
|---|---|
| content erasure | retirement keeps the text by design. A leaked credential is reported for rotation, never treated as handled by retiring the row |
| a purge or retention job | the removal record outliving the memory is the point |
| exact-content or threshold dedup on admission | the live pool holds no exact-content collisions, and similarity dedup discards memories without a recorded decision |
| a `reinforce` verb | absorption records why the count moved; a bare increment would not |
