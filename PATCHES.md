# PATCHES.md — upstream-merge companion

Companion to every upstream pull, read beside git history: deliberate omissions,
values chosen against upstream, and non-obvious adaptations.

This fork is the memory backend a Hermes deployment runs against. Its Python SDK is an
editable install in that agent's venv, so a merge here reaches a live runtime the moment
it lands on disk.

## Entry format — enforced

One entry, one verdict: what the fork does, why upstream's version loses.

- Keep the rule; drop the bug story, the counterfactual, the reverted attempt.
- Numbers belong in the Baseline and Values tables, never in prose. Name the config key.
- No per-merge history. Baseline is current state, overwritten by the next pull.
- Nothing the source already states. Delete it instead.
- Length is earned only by named symbols a merge could silently drop.

## Baseline

| Ref | Value |
|---|---|
| Upstream | `plastic-labs/honcho` `main` |
| Fork branch | `hermes` |
| Fork point | `0936ee2e` — upstream's tip; the backlog is closed |
| Last upstream merge | `0936ee2e`, 2026-10-02 |
| Carried surface | 99 files, +6,625 / -2,133 against `0936ee2e`, measured on the merge tree; 42 further files are fork-only additions |
| Collides with upstream | the next pull's manifest comes from the gap report; nothing is outstanding |
| Schema | unchanged by the fork; `migrations/` is byte-identical to `0936ee2e` |

## Deliberately not carried

- `create_documents` as a write path. The deriver, the conclusion route and the agent tools
  all admit through `create_observations`. Upstream's version decides create, merge and
  discard from a cosine threshold with no recorded reason, which the admission contract
  exists to prevent. Its exact-content reinforcement went with it and is not restored; that
  call was made against a pool that held no exact-content collisions at the time, which is a
  data observation and not a property the tree can assert, so re-check it before relying on it.
- The soft-delete reapers. `_cleanup_soft_deleted_documents_pgvector` and its reconciler batch
  are dropped outright, and `cleanup_soft_deleted_documents` drops the external vector but keeps
  the row, marking it `sync_state='purged'` so the eligibility query does not re-select it. A
  retired conclusion carries the removal ledger; hard-deleting it destroys the record of why it
  went. Upstream will keep maintaining both, and every merge that touches them re-decides this.
- `CreateDocumentsResult` as the return of `save_representation`. A count is reported instead,
  so an empty save stays falsy where the deriver tallies successful observers. Asserted in
  `tests/test_fork_held_values.py`.
- The four dedup counters on `RepresentationCompletedEvent`. They keep their schema default
  because the admission path produces none of them.
- Upstream's tests that patch `crud.create_documents` — the summed-dedup-counts test, the
  batch-embed fallback pair, and the `DERIVER_DEDUPLICATE` forwarding pair. They exercise a
  function no production path reaches.
- Upstream's `PromptRepresentation` conversion tests. `ExtractedRepresentation` replaced that
  model and `from_prompt_representation` with it.
- The dreamer rule telling the model to "delete outdated observations - don't leave
  duplicates". Retirement is permanent here and nothing reaps, so a model acting on that
  instruction mints removals with no ledger reason. The fork's rule in that slot is to use
  `enrich`, which absorbs the duplicate and moves its derivation count, and not to delete the
  target separately. Upstream's neighbouring rules in the same list are carried.
- Per-observation embedding inside `agent_tools.create_observations`. The admission path lets
  `crud.create_observations` embed, so upstream's batch-with-single-item-fallback belongs to
  the write path this fork does not use.
- Upstream's deriver instruction to write the peer id as the subject, and its test. The fork's
  deriver writes "the user"; see the adaptation below.
- Upstream's example-free deriver scaffold. The fork's selective-extraction rules keep their
  fabricated EXTRACT and NOTHING examples; when this was decided the live pool held no conclusion
  copied from one. Upstream's scaffold test is kept for the structured-output schemas and its
  legacy leaked facts only.
- Upstream's deriver rules for accepted proposals, atomic facts and reworded duplicates. The fork's
  acceptance rule, one-fact rule and writing contract already cover them, the acceptance rule more
  broadly.

## Values set against upstream's

Each is asserted against the **declared** default, not the resolved runtime value — `.env`
overrides say nothing about what a merge would revert. The assertion lives in
`tests/test_fork_held_values.py` unless the row names another file.

| Value | Ours | Upstream's |
|---|---|---|
| `DERIVER.WORK_UNIT_TIMEOUT_SECONDS` | 300, and six times that for dream, deletion and scope work | key does not exist |
| `DERIVER.DEDUPLICATE` | never forwarded to a write path | `True`, forwarded from the deriver and the agent tools |
| `DERIVER.DEDUPLICATE_MAX_DISTANCE` | 0.05, configurable — inert, see caveats | 0.05, a module constant |
| `DERIVER.MAX_OBSERVATIONS_PER_SESSION` | 0, off | key does not exist |
| `MAX_CONCLUSION_CHARS` / `CONCLUSION_TARGET_CHARS` | 800 / 500, declared in `src/writing_contract.py` and asserted in `tests/test_llm_writing_contract.py` | 65535, storage ceiling only |
| Grounded sources required of an inductive conclusion | 2, mirroring `ConclusionCreate` | 1 |

## Non-obvious adaptations

- Conclusions are admitted with evidence. Every write carries who decided, why, what was
  searched first, and what it rests on; `crud/document` verifies each reference against the
  store and a rejected reference fails the whole batch. Enrichment is a revision — the
  replacement is created and its predecessor retired in one transaction, with the
  predecessor's admission appended to `admission_history`, and a batch that enriches one
  conclusion twice is refused before any write. An admission citing messages records the
  latest one's send time; a dreamer admission records its sources'. A revision and an
  absorbing survivor list every message their predecessors cited. Explicit creates are
  never deduplicated behind the agent's back.
- Retirement is an admission too. `soft_delete_documents` is the only writer of
  `Document.deleted_at` and takes a required envelope — category, prose reason, actor —
  recorded at `internal_metadata.removal`. Seven paths route through it, four of them system
  categories that carry `entry_origin: system` and refuse agent attribution rather than
  fabricate it. A held-value test scans `src/` and fails on any other writer.
- Retired rows are permanent and keep their vector. Retention is what the removal record is
  for, so nothing reaps; `include_deleted` exists only on the conclusion list builder, and never
  on the deriver, dialectic or dreamer search paths — the held-value test binds exactly that, by
  scanning `agent_tools` and `crud.representation`. It is deliberately opt-in and default-false
  on the list surface above that builder: the conclusions route, both SDKs, and the MCP tool,
  which is agent-facing by design so a retired row can be read back with its removal reason.
  Every `select(models.Document)` site is inventoried in the
  held-value tests, so a new one fails the suite until it gets a verdict on retired rows.
- Reinforcement follows the memory. Enrichment writes `max(previous.times_derived + 1,
  supplied)` with the whole batch's predecessors locked in one id-ordered statement — per-target
  locking deadlocks two batches enriching the same pair in opposite order. `duplicate_absorbed`
  sums the retired row's count onto its survivor and appends a snapshot carrying the count
  either side, so callers never compute merge arithmetic themselves.
- A search receipt may name a retired conclusion, because a receipt is history. The evidence a
  derived conclusion rests on, and any enrichment target, must still be live.
- The message response carries `internal_id` alongside the public `id`. Conclusions cite the
  internal one in `source_message_ids`, so upstream already publishes that id space while
  offering no way to resolve it — a caller could read that a memory came from message N and
  never learn which message N is. Both SDKs carry it. Drop this only if upstream stops citing
  internal ids in the admission envelope.
- `src/writing_contract.py` owns the writing rules and the conclusion bounds. The deriver,
  the dialectic agent, both summarizers and the dreamer's specialists embed the shared
  block; the bound is published in the tool JSON schema the model plans against and
  validated on every write path including both SDKs. The TypeScript SDK counts Unicode
  code points and refuses before HTTP.
- The deriver writes about "the user" rather than spelling the peer id into prose, which
  removes the misspelled-subject variants and the near-duplicate churn they caused.
- `ConclusionCreate` carries `times_derived` and `source_ids`, so a consolidation merge
  keeps the accumulated reinforcement and provenance of what it merged.
- Provenance lives in upstream's `document_sources` edge table, adopted whole rather than kept
  as the fork's JSONB column. `Document.legacy_source_ids` maps the old `source_ids` column and
  a `source_ids` property coalesces table, column and legacy `internal_metadata` keys, so reads
  work mid-drain. The migration is DDL only; `reconciler.backfill_document_sources` drains the
  column asynchronously, and every traversal (`get_child_observations`, the scope cascade)
  matches the table `or` the legacy column until it finishes; linkage that only ever lived in
  `internal_metadata` reads through the property but not through those walks until it drains.
  The column and its GIN index cannot be dropped until upstream's follow-up migration, so do
  not treat them as dead.
- Cited `source_ids` are grounded before admission. Upstream's `_filter_ungrounded_source_ids`
  runs on the fork's admission path: fabricated ids are stripped, and an observation left below
  its level's source requirement is returned as an `ObservationFailure` rather than stored.
  `create_observations` reports those in `failed`, which it previously hardcoded empty. The
  dreamer prompts carry upstream's matching rule that `[id:xxx]` must be copied exactly and
  `search_messages` results cannot be cited. It is a layer in front of `crud`'s reference guard,
  not a duplicate of it: the filter rescues an observation by dropping only its fabricated ids,
  and the guard still refuses the batch over anything unresolved, search receipts and
  `source_message_ids` included.
- `ConclusionCreateParams` is defined twice in each SDK. The exported one beside the scoped client
  is the admission contract this server enforces; upstream's in the generated api types mirrors
  upstream's looser server, reaches no caller, and is kept so the writing-contract bound is
  asserted on it too.
- A pair-scoped conclusions view answers for its own pair only, `lineage` included; the
  workspace-level view is where a cross-pair read belongs. Upstream scopes `get` this way and
  the fork's lineage route follows it.
- Two collision points re-decide retirement on every pull, because upstream writes
  `Document.deleted_at` inline where the fork routes through `soft_delete_documents`: the
  semantic-duplicate replacement in `crud.document`, and the scope cascade in
  `deriver.scope_backfill`, where upstream turns the cascade select into an
  `UPDATE ... RETURNING`. The fork keeps both as selects, because the shared tail feeds every
  id to `soft_delete_documents` and an inline update would retire the row twice with no ledger.
- `GET /conclusions/{id}` and `GET /conclusions/{id}/lineage` are both live: upstream's reads
  the live pool and 404s on a retired row, the fork's returns the retired row with its ledger.
  Upstream's own soft-delete route test asserts the 404, so it also pins that split.
- A 429 whose body says `usage_limit_reached` is never retried: `is_transient_llm_error` in
  `src/llm/errors.py` screens every `retry()` site under `src/llm`, the tool loop's included,
  and the OpenAI clients are built with `max_retries=0` so the tenacity wrapper is the only
  retry policy.
- `honcho conclusion create` takes the JSON admission payload whole and refuses plain text;
  upstream's content-only create cannot pass the admission contract.
- `ExtractedRepresentation` and `AdmissionRepresentation` are the two response models; the
  structured-output repair path treats both as representation models. Extraction also returns
  what it weighed and left out, and admission what it refused, each with the excluding rule, and
  the deriver logs both per batch; upstream returns only what it keeps, so an over-excluding rule
  leaves no trace.
- Every batch message reaches the deriver inside upstream's `<message>` tag carrying one extra
  attribute, `message_id`. The admission pass cites database ids in `source_message_ids`, and
  upstream's `idx` is batch position, which resolves against nothing.
- The deriver extracts what stays true about the user, how they want things done from now on and
  what they rely on, from the user's own words or an assistant statement the user explicitly
  agreed with or asked to remember. What holds for one occasion, and what another owner holds by
  nature, is never a conclusion; upstream extracts any atomic fact. Rules state outcomes and
  examples are invented, never the user's own data. Custom instructions only narrow extraction.
- The deriver and the automatic answer read the assistant's own memory files, listed per workspace
  in `CURATED_MEMORY.PATHS`, each in a `<file>` tag after the rules; the deriver never extracts
  what they state and the answer never restates it. Upstream has no notion of what the assistant
  already carries, so it re-extracts that into the pool.
- The dialectic prompt is written for machine consumption: front-loaded answer, no second
  person, no questions, `Unknown: <specific gap>` for missing evidence, contradictions
  stated rather than asked about.
- Deriver resilience: a per-work-unit timeout frees the worker slot instead of deadlocking
  the pool, and stale work-unit cleanup runs independently of pool capacity.
- `has_pending_work` counts a retired row only behind an external vector store and only until
  it is `purged`; upstream's unguarded check is always true against permanent retirement. Held
  by `tests/reconciler/test_sync_vectors_enqueue_gate.py`.
- A local-memory seed block creates no queue work of its own; a summary triggered by a later
  message still reads it. Hermes no longer uploads these blocks when a session opens.

## Standing caveats for future pulls

- An SDK version bump runs `uv lock` in hermes-agent in the same pull; a major bump widens
  the agent's `honcho-ai` range first.
- `src/config.py` calls `load_dotenv(override=True)` at import, so `.env` beats the process
  environment regardless of the precedence its own settings sources declare.
  `PYTHON_DOTENV_DISABLED=1` turns it off.
- The suite does not run from a host shell without four corrections; they live in
  `~/.hermes/scripts/honcho-run-tests.sh`, which the daily gap job also uses so a hand run
  and the recorded baseline cannot diverge. Three are environment traps `CLAUDE.md` explains;
  the fourth forces `DREAM_ENABLED=true`, because `.env` carries the deployment's `false` and
  upstream's scope tests schedule dreams.
- A representation work unit becomes claimable at
  `REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS` and is then capped per LLM call by
  `REPRESENTATION_BATCH_TARGET_INPUT_TOKENS`; below the first it waits for the
  `REPRESENTATION_BATCH_MAX_AGE_SECONDS` flush. Anything asserting on derivation must exceed
  the claim threshold rather than poll briefly.
- The deriver makes two LLM calls on any turn that extracts — the extraction pass and the
  admission pass — where upstream makes one; the admission pass counts its input under the
  `admission` token component. `CLAUDE.md`'s single-call description is upstream's.
- `mcp/package.json` points `@honcho-ai/sdk` at `file:../sdks/typescript` rather than a
  published version, which is how the fork's conclusion fields reach the MCP tools. Its
  typecheck reads `dist/`, so the SDK must be built first.
- `mock_tracked_db` in `tests/conftest.py` is one call into `tests/tracked_db_patch.py`, which
  the live tests also take. An upstream edit to that fixture's body — session lifecycle, a new
  `tracked_db` import site — is ported into `_tracked_db` and `TRACKED_DB_TARGETS`, never
  resolved away with the conflict: an unpatched import site writes to whatever database
  settings resolve to. `tests/test_tracked_db_targets.py` fails on a module-level import site
  missing from the list.
- `src/routers/messages.py` widens upstream's enqueue payload with the message metadata, which
  is how `is_seeded_memory_message` sees a local-memory seed block.
- Upstream tests that create a conclusion arrive without the admission envelope, because
  upstream has no such contract: they fail on `action`, `reason_for_entry`, `search_query`,
  `searched_conclusion_ids`, `entry_origin`, `agent_trace_id`, `agent_model` long before
  reaching what they mean to assert, and a route-level delete fails for want of a removal
  envelope. Adapt them rather than drop them — the helpers exist (`_operator_conclusion` in
  `tests/sdk`, `operatorConclusion` in the TS tests) and the behaviour under test is usually
  one the fork wants covered. Watch for the inverse too: an auto-merge that keeps the fork's
  fixture and appends upstream's assertions produces a test that contradicts itself.
- Python 3.13 for the server: `.python-version`, the root `requires-python` and the project
  environment all match the `python:3.13-slim` containers and the agent's own interpreter,
  adopted ahead of the merge rather than during it. The suite passes on 3.13 unchanged, so
  upstream's own move to that floor is already reconciled and needs no verdict. `honcho-cli`
  is the exception and stays at `>=3.11`, matching upstream; it is not a carried value.
- `create_documents` and `is_rejected_duplicate` have no production caller; every write enters
  through `create_observations`. Their conflicts resolve toward upstream at no behavioural cost.
  `DERIVER.DEDUPLICATE_MAX_DISTANCE` is read at both sites beneath them — the candidate resolve
  in `create_documents` and the query in `_semantic_dup_decision` — so its assertion guards the
  declaration and not behaviour. `DERIVER.DEDUPLICATE` reaches neither; restoring either
  forwarding call re-enables threshold dedup on a live write path, which is why a test asserts
  the forwarding stays absent.
- `AutomaticDialecticAgent._prepare_query` overrides `DialecticAgent._prepare_query` and ends the
  run span the base opened before refusing an empty context. An upstream change to that signature
  or its returned tuple merges clean and fails only when an automatic chat runs;
  `tests/dialectic/test_automatic.py` is the tripwire.
- Prompt and instruction text is behavioural and security surface. An upstream edit to any
  prompt this fork rewrote gets read and given a verdict, never merged on the diff alone.

## Standing rules

- Upstream wins on implementation, never on behaviour. A changed default, threshold, flag
  or enum member is functionality; taking upstream's silently is a value regression.
- Every held value needs an assertion before the merge that could revert it, because
  upstream rewrites its own tests around sentinels on every pull.
- A carried patch that upstream now covers is dropped, not reconciled — but coverage is
  proven at merge time, never assumed from a gap brief.
