# PATCHES.md: upstream-merge companion

Companion to every upstream pull, read beside git history: deliberate omissions,
values chosen against upstream, and non-obvious adaptations.

This fork is the memory backend a Hermes deployment runs against. Its Python SDK is an
editable install in that agent's venv, so a merge here reaches a live runtime the moment
it lands on disk.

## Entry format (enforced)

One entry, one verdict: what the fork does, why upstream's version loses.

- Keep the rule. Drop the bug story, the counterfactual and the reverted attempt.
- Numbers belong in the Baseline and Values tables, never in prose. Name the settings key.
- No per-merge history. Baseline is current state, overwritten by the next pull.
- Nothing the source already states. Delete it instead.
- Length is earned only by named symbols a merge can silently drop.

## Baseline

| Ref | Value |
|---|---|
| Upstream | `plastic-labs/honcho` `main` |
| Fork branch | `hermes` |
| Fork point | `3922af85`, upstream's tip. The backlog is closed. |
| Last upstream merge | `3922af85`, 2026-10-10 |
| Carried surface | 102 files, +6,775 / -2,213 against `3922af85`, measured on the merge tree. 48 further files are fork-only additions. |
| Collides with upstream | The next pull's manifest comes from the gap report. Nothing is outstanding. |
| Schema | Unchanged by the fork. `migrations/` is byte-identical to `3922af85`. |

## Deliberately not carried

- `create_documents` as a write path. The deriver, the conclusion route and the agent tools
  all admit through `create_observations`. Upstream's version decides create, merge and
  discard from a cosine threshold with no recorded reason, which the admission contract
  exists to prevent. Its exact-content reinforcement went with it and is not restored. That
  ruling rests on the pool at the time, which held no exact-content collisions. This is a data
  observation, not a property the tree can assert. Re-check the pool before you rely on the ruling.
- The soft-delete reapers. `_cleanup_soft_deleted_documents_pgvector` and its reconciler batch
  are dropped outright. `cleanup_soft_deleted_documents` drops the external vector but keeps the
  row and marks it `sync_state='purged'`, so the eligibility query does not re-select it. A
  retired conclusion carries the removal ledger. A hard delete destroys the record of why it
  went. Upstream will keep maintaining both, and every merge that touches them re-decides this.
- `CreateDocumentsResult` as the return of `save_representation`. A count is reported instead,
  so an empty save stays falsy where the deriver tallies successful observers. Asserted in
  `tests/test_fork_held_values.py`.
- The four dedup counters on `RepresentationCompletedEvent`. They keep their schema default
  because the admission path produces none of them.
- Upstream's tests that patch `crud.create_documents`: the summed-dedup-counts test, the
  batch-embed fallback pair and the `DERIVER_DEDUPLICATE` forwarding pair. They exercise a
  function that no production path reaches.
- Upstream's `PromptRepresentation` conversion tests. `ExtractedRepresentation` replaced that
  model and `from_prompt_representation` with it.
- The dreamer rule telling the model to "delete outdated observations - don't leave
  duplicates". Retirement is permanent here and nothing reaps, so a model acting on that
  instruction mints removals with no ledger reason. The fork's rule in that slot is to use
  `enrich`, which absorbs the duplicate and moves its derivation count. The same rule says not
  to delete the target separately. Upstream's neighbouring rules in the same list are carried.
- Per-observation embedding inside `agent_tools.create_observations`. The admission path lets
  `crud.create_observations` embed, so upstream's batch-with-single-item-fallback belongs to
  the write path this fork does not use.
- Upstream's deriver instruction to write the peer id as the subject, and its test. The fork's
  deriver writes "the user". The adaptation below explains it.
- Upstream's example-free deriver scaffold. The fork's selective-extraction rules keep their
  fabricated EXTRACT and NOTHING examples. When this was decided, the live pool held no
  conclusion copied from one. Upstream's scaffold test is kept for the structured-output schemas
  and its legacy leaked facts only.
- Upstream's deriver rules for accepted proposals, atomic facts and reworded duplicates. The fork's
  acceptance rule, one-fact rule and writing contract already cover them, the acceptance rule more
  broadly.
- Upstream's `CLAUDE.md` instruction to read its PR and issue-filing rules before opening a PR or
  filing an issue. This fork never contributes upstream.

## Values set against upstream's

Each is asserted against the **declared** default, not the resolved runtime value. A `.env`
override says nothing about what a merge can revert. Unless a row names another file, the
assertion lives in `tests/test_fork_held_values.py`.

| Value | Ours | Upstream's |
|---|---|---|
| `DERIVER.WORK_UNIT_TIMEOUT_SECONDS` | 300, and six times that for dream, deletion and scope work | key does not exist |
| `DERIVER.DEDUPLICATE` | never forwarded to a write path | `True`, forwarded from the deriver and the agent tools |
| `DERIVER.DEDUPLICATE_MAX_DISTANCE` | 0.05, a settings key, inert (see caveats) | 0.05, a module constant |
| `DERIVER.MAX_OBSERVATIONS_PER_SESSION` | 0, off | key does not exist |
| `MAX_CONCLUSION_CHARS` / `CONCLUSION_TARGET_CHARS` | 800 / 500, declared in `src/writing_contract.py` and asserted in `tests/test_llm_writing_contract.py` | 65535, storage ceiling only |
| Grounded sources required of an inductive conclusion | 2, mirroring `ConclusionCreate` | 1 |

## Non-obvious adaptations

- Conclusions are admitted with evidence. Every write carries who decided, why, what was
  searched first, and what it rests on. `crud/document` verifies each reference against the
  store, and a rejected reference fails the whole batch.
- Enrichment is a revision. The replacement is created and its predecessor retired in one
  transaction, with the predecessor's admission appended to `admission_history`. A batch that
  enriches one conclusion twice is refused before any write.
- An admission citing messages records the latest one's send time. A dreamer admission records
  its sources' send time. A revision and an absorbing survivor list every message their
  predecessors cited. Explicit creates are never deduplicated behind the agent's back.
- Retirement is an admission too. `soft_delete_documents` is the only writer of
  `Document.deleted_at`. It takes a required envelope of category, prose reason and actor,
  recorded at `internal_metadata.removal`. Seven paths route through it. Four of them are system
  categories that carry `entry_origin: system` and refuse agent attribution rather than
  fabricate it. A held-value test scans `src/` and fails on any other writer.
- Retired rows are permanent and keep their vector. Retention is what the removal record is
  for, so nothing reaps. `include_deleted` exists only on the conclusion list builder, never on
  the deriver, dialectic or dreamer search paths. The held-value test scans `agent_tools` and
  `crud.representation` and binds exactly that.
- On the list surface above that builder, `include_deleted` is deliberately opt-in and
  default-false: the conclusions route, both SDKs, and the MCP tool. The MCP tool is agent-facing
  by design, so a retired row can be read back with its removal reason.
- Every `select(models.Document)` site is inventoried in the held-value tests. Each new site fails
  the suite and needs a verdict on retired rows.
- Reinforcement follows the memory. Enrichment writes `max(previous.times_derived + 1,
  supplied)` with the whole batch's predecessors locked in one id-ordered statement. Per-target
  locking deadlocks two batches that enrich the same pair in opposite order. `duplicate_absorbed`
  sums the retired row's count onto its survivor and appends a snapshot carrying the count
  either side, so callers never compute merge arithmetic themselves.
- A search receipt can name a retired conclusion, because a receipt is history. The evidence a
  derived conclusion rests on, and any enrichment target, must still be live.
- The message response carries `internal_id` alongside the public `id`. Conclusions cite the
  internal one in `source_message_ids`. Upstream publishes that id space but offers no way to
  resolve it. A caller then reads that a memory came from message N and never learns which
  message N is. Both SDKs carry `internal_id`. The one reason to drop it is upstream no longer
  citing internal ids in the admission envelope.
- `src/writing_contract.py` owns the writing rules and the conclusion bounds. The deriver,
  the dialectic agent, both summarizers and the dreamer's specialists embed the shared block.
  The bound is published in the tool JSON schema that the model plans against. Every write path
  verifies it, both SDKs included. The TypeScript SDK counts Unicode code points and refuses
  before HTTP.
- The deriver writes about "the user" rather than spelling the peer id into prose, which
  removes the misspelled-subject variants and the near-duplicate churn they caused.
- `ConclusionCreate` carries `times_derived` and `source_ids`, so a consolidation merge
  keeps the accumulated reinforcement and provenance of what it merged.
- Provenance lives in upstream's `document_sources` edge table, adopted whole rather than kept
  as the fork's JSONB column. `Document.legacy_source_ids` maps the old `source_ids` column. A
  `source_ids` property coalesces table, column and legacy `internal_metadata` keys, so reads
  work mid-drain.
- The provenance migration is DDL only. `reconciler.backfill_document_sources` drains the column
  asynchronously. Until the drain finishes, every traversal (`get_child_observations`, the scope
  cascade) matches the table `or` the legacy column. Until it drains, linkage that only ever lived
  in `internal_metadata` reads through the property but not through those walks.
- Until upstream's follow-up migration, the legacy column and its GIN index cannot be dropped. Do
  not treat them as dead.
- Cited `source_ids` are grounded before admission. Upstream's `_filter_ungrounded_source_ids`
  runs on the fork's admission path. Fabricated ids are stripped. An observation left below its
  level's source requirement is returned as an `ObservationFailure` rather than stored.
  `create_observations` reports those in `failed`, which it previously hardcoded empty. The
  dreamer prompts carry upstream's matching rule that `[id:xxx]` must be copied exactly and
  `search_messages` results cannot be cited.
- The grounding filter is a layer in front of `crud`'s reference guard, not a duplicate of it.
  The filter rescues an observation by dropping only its fabricated ids. The guard still refuses
  the batch over anything unresolved, search receipts and `source_message_ids` included.
- `ConclusionCreateParams` is defined twice in each SDK. The exported one beside the scoped client
  is the admission contract that this server enforces. Upstream's, in the generated api types,
  mirrors upstream's looser server and reaches no caller. It is kept so that the writing-contract
  bound is asserted on it too.
- A pair-scoped conclusions view answers for its own pair only, `lineage` included. The
  workspace-level view is where a cross-pair read belongs. Upstream scopes `get` this way and
  the fork's lineage route follows it.
- Two collision points re-decide retirement on every pull. Upstream writes `Document.deleted_at`
  inline where the fork routes through `soft_delete_documents`. The first point is the
  semantic-duplicate replacement in `crud.document`. The second is the scope cascade in
  `deriver.scope_backfill`, where upstream turns the cascade select into an
  `UPDATE ... RETURNING`.
- The fork keeps both collision points as selects, because the shared tail feeds every id to
  `soft_delete_documents`. With an inline update as well, each row is retired twice, once with no
  ledger.
- Both `GET /conclusions/{id}` and `GET /conclusions/{id}/lineage` are live. Upstream's route
  reads the live pool and answers 404 on a retired row. The fork's route returns the retired row
  with its ledger. Upstream's own soft-delete route test asserts the 404, so it also pins that split.
- A 429 whose body says `usage_limit_reached` is never retried. `is_transient_llm_error` in
  `src/llm/errors.py` screens every `retry()` site under `src/llm`, the tool loop's included. The
  OpenAI clients are built with `max_retries=0`, so the tenacity wrapper is the only retry policy.
- `honcho conclusion create` takes the JSON admission payload whole and refuses plain text.
  Upstream's content-only create cannot pass the admission contract.
- `ExtractedRepresentation` and `AdmissionRepresentation` are the two response models. The
  structured-output repair path treats both as representation models.
- Extraction also returns what it weighed and left out, and admission returns what it refused,
  each with the excluding rule. The deriver logs both per batch. Upstream returns only what it
  keeps, so an over-excluding rule leaves no trace.
- Every batch message reaches the deriver inside upstream's `<message>` tag carrying one extra
  attribute, `message_id`. The admission pass cites database ids in `source_message_ids`, and
  upstream's `idx` is batch position, which resolves against nothing.
- The deriver extracts what stays true about the user, how they want things done from now on,
  and what they rely on. Its sources are the user's own words and any assistant statement that
  the user explicitly agreed with or asked to remember.
- What holds for one occasion is never a conclusion. Neither is what another owner holds by
  nature. Upstream extracts any atomic fact. Rules state outcomes, and examples are invented,
  never the user's own data. Custom instructions only narrow extraction.
- The deriver and the automatic answer read the assistant's own memory files, listed per workspace
  in `CURATED_MEMORY.PATHS`, each in a `<file>` tag after the rules. The deriver never extracts
  what they state, and the answer never restates it. Upstream has no notion of what the assistant
  already carries, so it re-extracts that into the pool.
- The dialectic prompt is written for machine consumption: front-loaded answer, no second
  person, no questions, `Unknown: <specific gap>` for missing evidence, contradictions
  stated rather than asked about.
- Automatic mode applies the dialectic custom instructions that upstream added for chat. They sit
  before the curated-memory files in its prompt, so the files still close the cached prefix.
  `tests/dialectic/test_automatic.py` holds the wiring and the order.
- Deriver resilience: a per-work-unit timeout frees the worker slot instead of deadlocking
  the pool, and stale work-unit cleanup runs independently of pool capacity.
- `has_pending_work` counts a retired row only behind an external vector store. Once that row is
  `purged`, the check skips it. Upstream's unguarded check is always true against permanent
  retirement. Held by `tests/reconciler/test_sync_vectors_enqueue_gate.py`.
- A local-memory seed block creates no queue work of its own. A summary that a later message
  triggers still reads it. Hermes no longer uploads these blocks at session open.

## Standing caveats for future pulls

- Hermes installs the Python SDK as an editable path dependency, so it runs the new source after
  a gateway restart. Its lock picks up the new SDK version at the next hermes-agent relock. A pull
  needs no Hermes step for it, except that a major SDK bump widens the agent's `honcho-ai` range.
- `uv.lock` differs from upstream's only by `pysbd`, and its `exclude-newer` line stays upstream's.
  Whenever the local uv changes the lock, it writes its own form of that line. Upstream rewrites
  the line on every relock. After any fork relock, set the line back to upstream's value.
- `src/config.py` calls `load_dotenv(override=True)` at import, so `.env` beats the process
  environment regardless of the precedence its own settings sources declare.
  `PYTHON_DOTENV_DISABLED=1` turns it off.
- The suite does not run from a host shell without five corrections. They live in
  `~/.hermes/scripts/honcho-run-tests.sh`, which the daily gap job also uses, so a hand run
  and the recorded baseline cannot diverge. Four are environment traps that `CLAUDE.md` explains.
  The fifth forces `DREAM_ENABLED=true`, because `.env` carries the deployment's `false` and
  upstream's scope tests schedule dreams.
- A representation work unit becomes claimable at
  `REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS` and is then capped per LLM call by
  `REPRESENTATION_BATCH_TARGET_INPUT_TOKENS`. Below the first threshold, it waits for the
  `REPRESENTATION_BATCH_MAX_AGE_SECONDS` flush. Anything asserting on derivation must exceed
  the claim threshold rather than poll briefly.
- The deriver makes two LLM calls on any turn that extracts: the extraction pass and the
  admission pass. Upstream makes one. The admission pass counts its input under the `admission`
  token component.
- `mcp/package.json` points `@honcho-ai/sdk` at `file:../sdks/typescript` rather than a
  published version, which is how the fork's conclusion fields reach the MCP tools. Its
  typecheck reads `dist/`, so the SDK must be built first.
- `mock_tracked_db` in `tests/conftest.py` is one call into `tests/tracked_db_patch.py`, which
  the live tests also take. An upstream edit to that fixture's body, such as session lifecycle or
  a new `tracked_db` import site, is ported into `_tracked_db` and `TRACKED_DB_TARGETS`. It is
  never resolved away with the conflict. An unpatched import site writes to whatever database
  settings resolve to.
- `tests/test_tracked_db_targets.py` fails on a module-level import site missing from the list.
  An import site that upstream removes leaves the list too, because a stale target errors every
  test that takes the fixture.
- `src/routers/messages.py` widens upstream's enqueue payload with the message metadata, which
  is how `is_seeded_memory_message` sees a local-memory seed block.
- Upstream tests that create a conclusion arrive without the admission envelope, because
  upstream has no such contract. They fail on `action`, `reason_for_entry`, `search_query`,
  `searched_conclusion_ids`, `entry_origin`, `agent_trace_id` and `agent_model` long before they
  reach what they mean to assert. A route-level delete fails for want of a removal envelope.
- Adapt those upstream tests rather than drop them. The helpers exist (`_operator_conclusion` in
  `tests/sdk`, `operatorConclusion` in the TS tests), and the behaviour under test is usually
  one the fork wants covered. Watch for the inverse too. An auto-merge that keeps the fork's
  fixture and appends upstream's assertions produces a test that contradicts itself.
- Python 3.13 runs the server. `.python-version`, the root `requires-python` and the project
  environment all match the `python:3.13-slim` containers. This was adopted ahead of a merge, not
  during one. The suite passes on 3.13 unchanged, so upstream's own move to that floor is already
  reconciled and needs no verdict. `honcho-cli` is the exception. It stays at `>=3.11`, as
  upstream has it, and is not a carried value.
- `create_documents` and `is_rejected_duplicate` have no production caller. Every write enters
  through `create_observations`. Their conflicts resolve toward upstream at no behavioural cost.
- `DERIVER.DEDUPLICATE_MAX_DISTANCE` is read at every site beneath them: the external and pgvector
  candidate resolves in `create_documents`, and the query in `_semantic_dup_decision`. Its
  assertion therefore guards the declaration and not behaviour. Upstream's new read sites use its
  module constant, and a held-value test bans that constant, so each new site reads the setting.
- `DERIVER.DEDUPLICATE` reaches neither function. Restoring either forwarding call re-enables
  threshold dedup on a live write path, which is why a test asserts that the forwarding stays
  absent.
- `AutomaticDialecticAgent._prepare_query` overrides `DialecticAgent._prepare_query` and ends the
  run span the base opened before refusing an empty context. An upstream change to that signature
  or its returned tuple merges clean. It fails only at the first automatic chat, and
  `tests/dialectic/test_automatic.py` is the tripwire.
- Prompt and instruction text is behavioural and security surface. An upstream edit to any
  prompt this fork rewrote gets read and given a verdict, never merged on the diff alone.
- `CLAUDE.md` is rewritten in plain English throughout, upstream's sections included. An upstream
  edit to it conflicts on most pulls. Keep the fork's wording and port upstream's new facts into it.

## Standing rules

- Upstream wins on implementation, never on behaviour. A changed default, threshold, flag
  or enum member is functionality. Taking upstream's silently is a value regression.
- Every held value needs an assertion before the merge that can revert it, because
  upstream rewrites its own tests around sentinels on every pull.
- A carried patch that upstream now covers is dropped, not reconciled. Coverage is proven at
  merge time, never assumed from a gap brief.
