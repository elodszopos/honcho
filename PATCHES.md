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
| Fork point | `93dcf59c` (`v3.0.12`) |
| Last upstream merge | `v3.0.12`, 2026-09-05 |
| Carried surface | 75 files, +6,059 / -2,297 (2026-09-05) |
| Collides with upstream | 43 of those paths, ranked in the dated gap report |
| Schema | unchanged; `migrations/` is byte-identical from the fork point through upstream's tip |

## Deliberately not carried

- `create_documents` as a write path. The deriver, the conclusion route and the agent tools
  all admit through `create_observations`. Upstream's version decides create, merge and
  discard from a cosine threshold with no recorded reason, which the admission contract
  exists to prevent. Its exact-content reinforcement went with it; `PLAN-reinforcement.md`
  covers restoring that half without the threshold.
- `CreateDocumentsResult` as the return of `save_representation`. A count is reported instead,
  so an empty save stays falsy where the deriver tallies successful observers. Asserted in
  `tests/test_fork_held_values.py`.
- The four dedup counters on `RepresentationCompletedEvent`. They keep their schema default
  because the admission path produces none of them.
- Upstream's tests that patch `crud.create_documents` and the summed-dedup-counts test. They
  exercise a function no production path reaches.

## Values set against upstream's

Each is asserted in `tests/test_fork_held_values.py` against the **declared** default in
`src/config.py`, not the resolved runtime value — `.env` overrides say nothing about what
a merge would revert.

| Value | Ours | Upstream's |
|---|---|---|
| `DERIVER.WORK_UNIT_TIMEOUT_SECONDS` | 300 | key does not exist |
| `DERIVER.DEDUPLICATE_MAX_DISTANCE` | 0.05, configurable — inert, see caveats | 0.05, hardcoded at the call site |
| `DERIVER.MAX_OBSERVATIONS_PER_SESSION` | 0, off | key does not exist |
| `MAX_CONCLUSION_CHARS` / `CONCLUSION_TARGET_CHARS` | 800 / 500 | 65535, storage ceiling only |
| SDK version | 2.3.1, the version Hermes pins | upstream's own release cadence |

## Non-obvious adaptations

- Conclusions are admitted with evidence. Every write carries who decided, why, what was
  searched first, and what it rests on; `crud/document` verifies each reference against the
  store and a rejected reference fails the whole batch. Enrichment is a revision — the
  replacement is created and its predecessor retired in one transaction, with the
  predecessor's admission appended to `admission_history`. Explicit creates are never
  deduplicated behind the agent's back.
- `src/writing_contract.py` owns the writing rules and the conclusion bounds. The deriver,
  the dialectic agent, both summarizers and the dreamer's specialists embed the shared
  block; the bound is published in the tool JSON schema the model plans against and
  validated on every write path including both SDKs. The TypeScript SDK counts Unicode
  code points and refuses before HTTP.
- The deriver writes about "the user" rather than spelling the peer id into prose, which
  removes the misspelled-subject variants and the near-duplicate churn they caused.
- `ConclusionCreate` carries `times_derived` and `source_ids`, so a consolidation merge
  keeps the accumulated reinforcement and provenance of what it merged.
- A 429 whose body says `usage_limit_reached` is never retried (`is_transient_llm_error`),
  and the OpenAI clients are built with `max_retries=0` so the tenacity wrapper is the only
  retry policy.
- `ExtractedRepresentation` and `AdmissionRepresentation` are the two response models; the
  structured-output repair path treats both as representation models.
- The deriver prompt is a four-criteria selective-extraction contract with an enumerated
  exclusion taxonomy, an outcome-over-process rule, and travel-scope exclusions. Zero
  extractions is the expected output for an ordinary turn. Custom instructions may narrow
  extraction and never relax the exclusions.
- The dialectic prompt is written for machine consumption: front-loaded answer, no second
  person, no questions, `Unknown: <specific gap>` for missing evidence, contradictions
  stated rather than asked about.
- Deriver resilience: a per-work-unit timeout frees the worker slot instead of deadlocking
  the pool, and stale work-unit cleanup runs independently of pool capacity.
- Local-memory seed blocks stay stored and searchable while staying out of representation,
  summary, dream and queue generation.

## Standing caveats for future pulls

- The SDK is installed editable into the agent's venv. A merge that bumps its version
  breaks the agent's pin; that pin is asserted from the other side in the agent repo's
  `tests/test_fork_held_values.py`.
- `src/config.py` calls `load_dotenv(override=True)` at import, so `.env` beats the process
  environment regardless of the precedence its own settings sources declare.
  `PYTHON_DOTENV_DISABLED=1` turns it off.
- The suite does not run from a host shell without three fixes; they live in
  `~/.hermes/scripts/honcho-run-tests.sh`, which the daily gap job also uses so a hand run
  and the recorded baseline cannot diverge. `CLAUDE.md` explains each.
- A representation work unit under `REPRESENTATION_BATCH_MAX_TOKENS` waits for the
  `REPRESENTATION_BATCH_MAX_AGE_SECONDS` flush, which is why anything asserting on
  derivation must exceed the token threshold rather than poll briefly.
- Python 3.13 everywhere: `.python-version`, `requires-python` and the project environment
  all match the `python:3.13-slim` containers and the agent's own interpreter, adopted
  ahead of the merge rather than during it. The suite passes on 3.13 unchanged, so
  upstream's own move to that floor is already reconciled and needs no verdict.
- `create_documents` and `is_rejected_duplicate` have no production caller; every write enters
  through `create_observations`. Their conflicts resolve toward upstream at no behavioural cost.
  `DERIVER.DEDUPLICATE_MAX_DISTANCE` is read only there, so its assertion guards the declaration
  and not behaviour.
- Prompt and instruction text is behavioural and security surface. An upstream edit to any
  prompt this fork rewrote gets read and given a verdict, never merged on the diff alone.

## Standing rules

- Upstream wins on implementation, never on behaviour. A changed default, threshold, flag
  or enum member is functionality; taking upstream's silently is a value regression.
- Every held value needs an assertion before the merge that could revert it, because
  upstream rewrites its own tests around sentinels on every pull.
- A carried patch that upstream now covers is dropped, not reconciled — but coverage is
  proven at merge time, never assumed from a gap brief.
