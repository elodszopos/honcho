# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## Fork-owned docs

`.claude/docs/` holds documentation that this fork owns. Upstream never touches it, so it carries no
merge surface. Read the relevant doc before you work in its area.

| Doc | Covers |
|---|---|
| `.claude/docs/memory-ledger.md` | conclusion lifecycle, removal categories, how reinforcement accumulates, which read surface answers which question, the invariants and what enforces them |

## The Memory System Is One Unit, Across Three Roots

Honcho work is never only this repo. The server, its clients and its doctrine ship together.
A contract changed in one root and not the others is a broken feature, not a phased rollout.

| Root | Holds |
|---|---|
| `~/Projects/honcho/` | server, routers, crud, schemas, plus four client surfaces: `mcp/`, `sdks/python/`, `sdks/typescript/`, `honcho-cli/` |
| `~/.hermes/hermes-agent/plugins/memory/honcho/` | the tool the agent actually calls: `hermes_memory.py` (schema + handler), `session.py` (wrapper), `client.py`, `cli.py`, `audit.py` (local mutation log) |
| `~/.hermes/skills/productivity/memory-honcho-steward/` | janitor and synthesist doctrine: the prompts that drive every steward write |

**Trace inward from the caller, never outward from the server**. The steward reaches Honcho
through the plugin's `honcho_conclusions` tool and the Python SDK, not through the MCP server.
A change to the MCP tool alone reaches nothing the steward does. Before you design any contract
change, find every caller in all three roots and list them. Database evidence about who wrote a
row is not proof of which code path wrote it.

**Enumerate all five client surfaces, every time**. MCP server, Python SDK, TypeScript SDK,
`honcho-cli/`, and the Hermes plugin. `honcho-cli/` is the one that gets missed. It sits inside
this repo rather than beside the SDKs, and it broke silently on a signature change that every
other surface got. Search the method name with `rg` across all of them before you call a contract
done.

Scope, plan and commit all three together.

## Agent Writing Contract

- Read `$HOME/.hermes/hermes-agent/agent/agent-writing-contract.md` first.
- Apply it to every agent-consumed prompt, template, task body, and instruction.
- Use `src/writing_contract.py` as the runtime owner for Honcho-authored LLM text.
- Keep LLM-facing text compact, atomic, and structured.
- Keep human-facing text natural and plainspoken unless the user requests another style.
- Preserve raw messages, quotes, search queries, and machine data exactly. They are inputs, not authored prose.

# Honcho Overview

## What is Honcho?

Honcho is an infrastructure layer for building AI agents with memory and social cognition. Its primary purposes include:

- Imbuing agents with a sense of identity
- Personalizing user experiences through understanding user psychology
- Providing a Chat Endpoint (the Dialectic agent) that injects personal context just-in-time
- Supporting development of LLM-powered applications that adapt to end users
- Enabling multi-peer sessions where multiple participants (users or agents) can interact

Honcho uses the reasoning capabilities of LLMs to build coherent models of user psychology over time.
These models make AI interactions more personal and more effective.

## Core Concepts

### Peer Model

Honcho uses a peer-based model where both users and agents are represented as "peers". This unified approach enables:

- Multi-participant sessions with mixed human and AI agents
- Observation configuration (which peers observe which others)
- Flexible identity management for all participants

### Key Primitives

- **Workspace** (formerly App): The root organizational unit containing all resources
- **Peer** (formerly User): Any participant in the system (human or AI)
- **Session**: A conversation context that can involve multiple peers
- **Message**: Data units that can represent communication between peers OR arbitrary data that a peer ingests to extend its global representation
- **Collections & Documents**: Internal vector storage for peer representations. Collections are keyed by `(observer, observed)` peer pairs. Collections/Documents are not directly exposed via API, but the observations stored within them are exposed as **Conclusions** (see `/v3/.../conclusions` endpoints).

## Architecture Overview

### API Structure

All API routes follow the pattern: `/v3/{resource}/{id}/{action}`. Most "list/search" endpoints are `POST` so they can accept rich filter bodies.

- **Workspaces**: Create, list, update, search
- **Peers**: Create, list, update, chat (dialectic), messages, representation
- **Sessions**: Create, list, update, delete, clone, manage peers, get context
- **Messages**: Create (batch up to 100), upload (file), list, get, update
- **Conclusions**: Create, list, query (semantic search), delete. "Conclusions" is the API-facing name for observations stored in `(observer, observed)` collections.
- **Keys**: Create scoped JWTs
- **Webhooks**: Register endpoint, list, delete, test

### Key Features

#### Chat Endpoint (Dialectic agent) (`/peers/{peer_id}/chat`)

- Provides bespoke responses informed by the representation
- Integrates long-term facts from vector storage
- Supports streaming responses
- Configurable LLM providers

#### Message Processing Pipeline

1. Messages created via API (batch or single)
2. Enqueued for background processing:
   - `representation`: Update peer's context
   - `summary`: Create session summaries
3. Session-based queue processing ensures order
4. Results stored internally in vector DB

### Configuration

- Hierarchical configuration: `config.toml` + environment variables
- Database configuration with connection pooling
- Multiple LLM provider support
- Background worker (deriver) configuration
- Authentication can be toggled on/off

## Development Guide

### Commands

- Setup: `uv sync`
- Run server: `uv run fastapi dev src/main.py`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/path/to/test_file.py::test_function`
- Linting: `uv run ruff check src/`
- Typechecking: `uv run basedpyright`
- Format code: `uv run ruff format src/`

### Local Test Environment

For `uv run pytest tests/` to pass, `.env` must name a database the host can reach and the vector width the fixtures use. On a machine running the Docker stack, neither holds, and the three `.env` traps below report as collection errors rather than failures. The fourth trap comes from the shell that starts the run. It stops pytest at conftest import, before pytest collects a test.

| Trap | Symptom | Resolution |
|---|---|---|
| `.env` outranks the process environment | An inline `DB_CONNECTION_URI=...` changes nothing. The old value still resolves | `src/config.py` calls `load_dotenv(override=True)` at import, ahead of the `env > .env` order its own `settings_customise_sources` returns. `PYTHON_DOTENV_DISABLED=1` turns it off, and every value then has to come from the environment |
| Compose hostname reached from the host | `failed to resolve host 'database'` | `.env` carries the in-network `database:5432`. Compose publishes the same Postgres on `127.0.0.1:18732` and Redis on `127.0.0.1:18733` |
| Embedding width | `expected 1024 dimensions, not 1536` | Fixtures hardcode 1536-wide vectors (`tests/utils/test_agent_tools.py`). A deployment on another embedding model sets `EMBEDDING_VECTOR_DIMENSIONS` to its own width, and every test that writes an embedding then errors |
| A Hermes-launched shell | `No module named 'pydantic_core._pydantic_core'` at conftest import | Hermes exports `PYTHONPATH` with its own site-packages, built for another Python. Unset `PYTHONPATH` before the run |

`$HOME/.hermes/scripts/honcho-run-tests.sh` applies all four and passes its arguments
through to pytest: call it with no arguments for the whole suite, or
`$HOME/.hermes/scripts/honcho-run-tests.sh tests/utils/test_agent_tools.py -q` for one
file. It is the same
runner the daily upstream-gap job uses, so a hand run and the recorded baseline cannot
diverge.

Prompt, schema and SDK-shape tests need none of that. `tests/test_llm_writing_contract.py` and `tests/deriver/test_agent_writing_contract.py` run against a bare `uv run pytest`.

### End-to-end against real models

`$HOME/.hermes/scripts/honcho-run-tests.sh --live` runs `tests/live_llm/` against the configured
provider and embedding model. It spends real tokens, so nothing schedules it: the daily
upstream-gap job runs the same script with no arguments and the marker gate skips every live test.
Pass a path to narrow it.

`tests/live_llm/test_live_memory_contract_e2e.py` is the full memory contract in one journey. It is
the only test anywhere that proves the pipeline works with a real model, not a mocked one:

| Step | What it proves |
|---|---|
| A real message through `process_representation_tasks_batch` | extraction and the admission pass both run. The conclusion lands with a complete envelope: `entry_origin`, `reason_for_entry`, `search_query`, and `source_message_ids` that resolve to the message that produced it |
| A derived conclusion citing it, with a fabricated id alongside | grounding strips the invented id, the real one becomes a `document_sources` edge, and reverse traversal finds the child |
| Retiring the source | the row keeps its removal record and leaves the live pool |
| Citing the retired source | the admission reference guard refuses it |

Run it after any change to the deriver, the admission path, the prompt, or the conclusion schema.
A mocked suite cannot catch a prompt that stopped carrying what the model needs to cite.

What the deriver stores from real chats is measured by the deriver eval: `.claude/docs/deriver-eval.md`.

The `--live` mode exists because two things differ from a hermetic run. The provider URLs in `.env`
name the compose network, so a host run has to reach them on loopback. The fixture vector width
must stay unset, or the real embedding model's answer is rejected. `tests/live_llm/` also sits in
`_RUNTIME_MOCK_TEST_BLOCKLIST_PREFIXES`, which unhooks `tracked_db` along with the provider mocks.
A live test that reaches background code writes to whatever database the configuration resolves
to. Such a test must take `patch_tracked_db` from `tests/tracked_db_patch.py` itself.

The suite builds and discards its own `test_db*` database per xdist worker, and backs the cache
with fakeredis. A run leaves the live `postgres` database and the deriver's queue alone.

### SDK Testing

#### TypeScript SDK

**🚨 DO NOT RUN `bun test` DIRECTLY. IT WILL NOT WORK. 🚨**

The TypeScript SDK tests require a running Honcho server with database and Redis. Running `bun test` alone fails at once, because there is no server. Pytest orchestrates the tests and handles all the infrastructure setup.

**The ONLY way to run TypeScript SDK tests is from the monorepo root**.

```bash
# From the monorepo root (not from sdks/typescript/)
uv run pytest tests/ -k typescript
```

**The TypeScript SDK typecheck is safe to run directly**.

```bash
cd sdks/typescript && bun run tsc --noEmit
```

### Code Style

- Follow isort conventions with absolute imports preferred
- Use explicit type hints with SQLAlchemy mapped_column annotations
- snake_case for variables and functions. PascalCase for classes.
- Line length: 88 chars (Black compatible)
- Explicit error handling with appropriate exception types
- Docstrings: Use Google style docstrings
- **Never hold a DB session during external calls** (LLM, embedding, HTTP). If a function needs both a DB session and an external call result, compute the external result first and pass it as a parameter. This avoids tying up DB connections during slow network I/O. Use `tracked_db` for short-lived, DB-only operations. When multiple DB-only calls can reuse one connection, pass a shared session.
- **Never write through a read-only session** (`tracked_db(..., read_only=True)`, `get_read_db`, `ReadSessionLocal`). These sessions run in AUTOCOMMIT mode with no transaction. The database does NOT block their writes. A write commits silently and at once, and `begin_nested()` savepoints break. No runtime guard exists, and only convention enforces this. Use `read_only=True` strictly for SELECT-only windows. Anything that mutates, get-or-create paths included, must use a regular write session.

#### Multi-row locking and deadlocks

Two tables take concurrent writes from more than one worker: `documents` (deriver, dreamer, scope backfill/removal, reconciler) and `queue` (every deriver replica). When two writers touch an overlapping row set in different orders, these tables deadlock. Rules:

- **A multi-row `SELECT ... FOR UPDATE` MUST carry an explicit `ORDER BY <pk>`**. Without it Postgres locks in scan order, which differs per plan, so two writers with overlapping sets can cycle. `_apply_document_row_updates` in `src/crud/document.py` is the reference implementation.
- **`WHERE id IN (...)` does NOT impose an order**, so sorting the Python list is a no-op. The planner discards the list order and picks `Bitmap Heap Scan` (ctid order), `Index Scan` (id order) or `Seq Scan` per invocation. Deterministic ordering requires either a preceding `SELECT ... ORDER BY id FOR UPDATE` or `WHERE id IN (SELECT id ... ORDER BY id FOR UPDATE)`.
- **`Document.id` is a random nanoid** (`models.py`), so id order is uncorrelated with physical order. Per row pair, an unordered predicate `UPDATE`/`DELETE` is roughly a coin flip against an id-ordered locker, not a rare edge case. `QueueItem.id` is an integer identity, so there id order is also chronological.
- **Prefer no lock at all**. A single `UPDATE ... WHERE <predicate>` acquires row locks as it writes, with no separate lock phase to get wrong. Use `FOR UPDATE` only for a read-modify-write: a value read, computed in Python and written back. That read-modify-write is the only reason `_apply_document_row_updates` locks, and it replaced a server-side `func.greatest()`. The lock requires `populate_existing=True`, so the identity map does not serve a stale pre-lock value. Server-side expressions (`func.greatest`, the JSONB `-` operator) avoid the lock entirely. `_clear_work_unit_retry_attempts` in `src/deriver/queue_manager.py` is the example.
- `FOR UPDATE SKIP LOCKED` (the reconciler's claim pattern) never waits, so it cannot be a deadlock partner. But a claim held across an external call still stalls other writers. See the "never hold a DB session during external calls" rule above.

#### Auth scoping

- **`allow_member_read=True` (in `require_auth(...)`) is read-only. NEVER set it on a route that mutates state**. When a peer-scoped key's peer is an active member of a session, the flag lets that key reach the session route. On a mutating route, the flag gives any session member write access: message injection, configuration mutation, deletion. HTTP method is not a reliable read/write signal here, because some read routes use POST for a richer body. An explicit allowlist in `tests/routes/test_auth_route_policy.py` enforces this. Until you add a new route to `EXPECTED_MEMBER_READ_ROUTES` on purpose, the flag on that route fails the test. Never add a mutating method there.
- **A member-read route can be keyed by another sub-resource**, for example `peers/{peer_id}/config`. Then the handler must also make sure that a peer-scoped caller reads only its OWN resource. The condition is `jwt_params.p == peer_id`, and any other caller gets `AuthenticationException`. Membership grants session access, not access to a co-member's data. See `get_peer_config` in `src/routers/sessions.py`.

### Runtime Architecture

Honcho runs as two cooperating processes that share a Postgres database and Redis cache:

- **API server** (`uv run fastapi dev src/main.py`): handles HTTP, enqueues background work and returns immediately. It hosts the **Dialectic** agent inline, as a synchronous tool loop during chat requests.
- **Deriver worker** (`uv run python -m src.deriver`): long-running queue consumer (uvloop). It runs the **Deriver**, **Summarizer** and **Dreamer** off the queue, and can run as multiple instances (`DERIVER_WORKERS`). It also hosts an in-process **Reconciler scheduler** (`src/reconciler/`). The scheduler periodically embeds messages with `sync_state='pending'` in `MessageEmbedding`, cleans up stale queue items and runs registered backfills. Embedding generation is decoupled from message creation by design.

### Agent Architecture

Honcho uses several specialized LLM agents. They share tool definitions and the LLM client abstraction in `src/utils/agent_tools.py` + `src/llm/`.

> **Terminology:** users see **conclusions**, the public API term that the documentation uses. Code symbols call them **observations**, for example `create_observations`, `delete_observations` and `get_observation_context`. Doc prose below says "conclusions", and code symbol names keep "observations".

#### 1. Deriver (`src/deriver/`)

**Role**: Memory formation through content ingestion.

The Deriver processes batches of incoming messages and extracts conclusions about peers. The current architecture is "minimal deriver": structured output, not an agentic tool loop. Upstream makes one LLM call per batch. This fork adds a second, the admission pass, on any batch that extracts candidates. This trades flexibility for cost and predictability.

- **Trigger**: `src/deriver/enqueue.py` enqueues messages on message create. They are consumed by `src/deriver/queue_manager.py` → `consumer.process_item()` → `deriver.process_representation_tasks_batch()`.
- **Output**: Explicit conclusions (direct facts) and deductive conclusions (inferences) saved to `(observer, observed)` collections.
- **What it leaves out**: extraction returns the statements it weighed and did not extract. Admission returns the candidates it refused. Each comes with the excluding rule. Neither is stored, and each batch logs them as `deriver.batch left_out` and `deriver.batch rejected`.
- **Entry point**: `src/deriver/__main__.py` → `queue_manager.main()`.
- **Prompts**: `src/deriver/prompts.py` (`deriver_messages`): the rules go out as the system turn and the batch as the user turn, so the rules cache across calls. The workspace's curated files follow the rules, each inside a `<file>` tag named after it. Their order is the order in which `CURATED_MEMORY_PATHS` lists the files mounted into the container, steadiest first. The deriver never extracts what they state. The admission pass appends its own rules after them, so extraction and admission of one batch share one cached prefix. The same block closes the automatic answer's system prompt. The Codex backend caches a prefix only from about 2048 tokens. A workspace without a curated file has a 1.8k-token extraction prefix and never hits.
- **Custom instructions**: workspace, session and message configuration can add guidance through `reasoning.custom_instructions`. Workspace and session configuration can also set the shared `custom_instructions`. `DERIVER__MAX_CUSTOM_INSTRUCTIONS_TOKENS` caps the addition (default 2000). `DERIVER__MAX_INPUT_TOKENS` defaults to 25000 to make room.

#### 2. Dialectic (`src/dialectic/`)

**Role**: Analysis and recall for answering queries.

The Dialectic answers questions about peers and gathers context from memory to do it. It is the only tool-using agent on the synchronous request path. It loops over `DIALECTIC_TOOLS` until it has enough context to answer. The Dreamer specialists also use tools, but run off the queue.

- **Trigger**: API call to `POST /v3/.../peers/{peer_id}/chat`.
- **Tools** (see `DIALECTIC_TOOLS` in `src/utils/agent_tools.py`): `search_memory`, `search_messages`, `get_observation_context`, `grep_messages`, `get_messages_by_date_range`, `search_messages_temporal`, `get_reasoning_chain`. At the `minimal` reasoning level, a reduced set (`DIALECTIC_TOOLS_MINIMAL`) is used: just `search_memory` + `search_messages`.
- **Reasoning levels**: 5 tiers (`minimal`, `low`, `medium`, `high`, `max`), each with its own model configuration (see `DialecticLevelSettings` in `src/config.py`).
- **Output**: Natural language response grounded in gathered context. Supports SSE streaming.
- **Entry point**: `src/dialectic/chat.py` → `agentic_chat()` / `agentic_chat_stream()` → `DialecticAgent` (in `src/dialectic/core.py`).
- **Automatic mode** (`DialecticOptions.automatic`, `src/dialectic/automatic.py`): one server-side prefetch on the caller's `search_text`, one model call with no tools and no peer card, short bullets or `NONE`. Never streams. The Hermes plugin's lane recall calls it. Dialectic custom instructions sit before the curated files in its prompt.
  - The search runs one thought at a time (lines, then sentences) over past messages, and ranks each message at its best distance. The query filters out the caller's own session, one-line acknowledgements and messages past `excerpt_max_distance`.
  - The context is those excerpts with their thread summaries. The messages behind the conclusion ids that the caller already holds stay out of the search and out of every excerpt's context. Conclusions are never prefetched here. An excerpt message longer than the excerpt cap is cut at its last sentence end within the cap.
  - A failed embed or read raises out of the run, so the caller sees a failed call rather than `NONE`. A prefetch with no excerpt makes no model call at all and answers empty (`outcome=no_context`).
  - Knobs: `EMBEDDING.THOUGHT_MIN_CHARS` and `EMBEDDING.THOUGHT_CACHE_SECONDS` (a thought's vector is reused across the conclusion query and the answer prefetch of one message), `DIALECTIC.AUTOMATIC_EXCERPT_MESSAGE_CHARS`, `DIALECTIC.AUTOMATIC_EXCERPT_MIN_MESSAGE_CHARS`. Search text has no thought cap beyond the API's length limit.
  - Each run logs `dialectic.automatic prefetch` (thought, excerpt, summary and cut counts, rendered chars), `dialectic.automatic excerpts` (rank, session, message ids and distances) and `dialectic.automatic answer`. `run_id` joins them to the run's `dialectic.prepare` and `dialectic.answer` lines.
- **Per-thought conclusion query** (`per_thought` on the conclusion query): the same thought split, with one vector query per thought and each conclusion at its best distance. Every query applies the distance floor and `exclude_ids`, and the run logs `documents.query per_thought`.
- **pgvector only**: both per-thought searches need the pgvector store and pgvector 0.8 or newer (iterative HNSW scans). A migrated external store is refused with `VectorStoreError`, and startup fails on an older extension.

#### 3. Dreamer (`src/dreamer/`)

**Role**: Consolidation and self-improvement of memory.

The Dreamer is an orchestrated multi-specialist system that runs during scheduled "dreams" to consolidate conclusions and build reasoning trees.

- **Trigger**: Scheduled via `DreamScheduler` (`src/dreamer/dream_scheduler.py`) or explicit dream task on the queue.
- **Strategy**: Surprisal-based prioritization (`src/dreamer/surprisal.py`) selects which conclusions to expand. The orchestrator (`orchestrator.run_dream`) runs two specialist phases:
  1. **DeductionSpecialist** (`specialists.py`): produces deductive conclusions from explicit conclusions. Tools: `get_recent_observations`, `search_memory`, `search_messages`, `create_observations_deductive`, `delete_observations`, `update_peer_card`.
  2. **InductionSpecialist**: produces inductive conclusions from explicit + deductive conclusions. Tools: same discovery set + `create_observations_inductive`.
- **Reasoning trees** (`src/dreamer/trees/`, migration `f1a2b3c4d5e6_add_reasoning_tree_columns`): each conclusion links to its premises and downstream conclusions, so `get_reasoning_chain` can traverse them at recall time.
- **Output**: Deductive/inductive conclusions, consolidated redundancies, updated peer cards.
- **Entry point**: `src/dreamer/orchestrator.py` → `process_dream()` (the package-level export from `src/dreamer/__init__.py`), which wraps `run_dream()`.

#### 4. Summarizer (`src/utils/summarizer.py`)

**Role**: Two-tier session summarization (direct LLM call, no agentic tools).

- **Trigger**: Runs as part of the queue pipeline alongside representation tasks.
- **Tiers**: a short summary every `SUMMARY_MESSAGES_PER_SHORT_SUMMARY` messages (default 20), and a long summary every `SUMMARY_MESSAGES_PER_LONG_SUMMARY` messages (default 60). `SUMMARY_MAX_TOKENS_SHORT` and `SUMMARY_MAX_TOKENS_LONG` set the token caps.

#### Shared Agent Infrastructure

- **Tool definitions** (`src/utils/agent_tools.py`): one `TOOLS` dict and the per-agent lists (`DIALECTIC_TOOLS`, `DIALECTIC_TOOLS_MINIMAL`, `WORKSPACE_DIALECTIC_TOOLS`, `WORKSPACE_TOOLS_MINIMAL`, `DEDUCTION_SPECIALIST_TOOLS`, `INDUCTION_SPECIALIST_TOOLS`, `CARD_REFRESH_SPECIALIST_TOOLS`).
- **LLM subsystem** (`src/llm/`): provider-agnostic `honcho_llm_call()`. Backends in `src/llm/backends/` (`anthropic.py`, `gemini.py`, `openai.py`). It includes prompt caching (`caching.py`), structured output (`structured_output.py`) and the tool loop (`tool_loop.py`). It also has history adapters for cross-provider message formats and a model registry. An `AttemptPlan` pins per-retry provider selection. After the tool loop settles on a fallback, stream-final retries stay on it and do not bounce back to the primary.
- **Per-agent model configuration**: each agent has its own `MODEL_CONFIG` in `src/config.py` with fallback chains (see `ConfiguredModelSettings`, `FallbackModelSettings`).
- **Telemetry**: cloudevents in `src/telemetry/events/` cover API routes, dialectic, dream, deletion, reconciliation, representation and per-call LLM accounting. In `llm.py`, `LLMCallCompletedEvent` fires once per provider hit with full cost-attribution context. `TelemetrySettings.HIGH_VOLUME_SAMPLE_RATE` samples high-volume events deterministically per `run_id`.
- **Prometheus metrics** (`src/telemetry/prometheus/`): every metric carries a `namespace` label. Every recorder is fail-soft: a metrics error never propagates into a request or a worker loop. Counter children with a *bounded* label domain are zero-initialized per process at startup. `initialize_bounded_metrics(instance_type=...)` does this, called from the `src/main.py` lifespan (`api`) and from `src/deriver/__main__.py` (`deriver`). An absent series therefore means a broken scrape, not "nothing happened". Before you touch telemetry, know two consequences:
  - **A new `BaseEvent` subclass needs its `_event_type` in `ALL_EVENT_TYPES`** (`src/telemetry/events/__init__.py`). When `_volume_class == "high_volume"`, it also goes in `HIGH_VOLUME_EVENT_TYPES`. The drift guards in `tests/telemetry/test_metric_zero_init.py` enforce this, and assert set-equality against the discovered subclasses.
  - **Every replica must refresh a service-wide, non-additive gauge on its own timer**. Aggregate such a gauge with `max()` or `avg()`, never `sum()`. `message_embeddings_pending` is the example. It reports a DB-global count, so `ReconcilerScheduler._scheduler_loop` drives it on all replicas, not the work-unit-deduped reconciliation cycle. A cycle-driven gauge with the zero-init reads a confident permanent `0` on every replica that never wins the work unit.

### Project Structure

```
src/
├── main.py              # FastAPI app: middleware, routers, lifespan, exception handlers
├── models.py            # SQLAlchemy ORM models (Workspace/Peer/Session/Message/
│                        #   MessageEmbedding/Collection/Document/QueueItem/...)
├── config.py            # Pydantic-settings configuration (very large; see README)
├── db.py                # Engine + session/context management (request_context var)
├── dependencies.py      # FastAPI DI (tracked_db, etc.)
├── exceptions.py        # Custom exception types (HonchoException + subclasses)
├── security.py          # JWT authentication
├── embedding_client.py  # Embedding provider client (configurable dimensions
│                        #   via EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE)
├── schemas/             # Pydantic schemas
│   ├── api.py            # Public API request/response schemas
│   ├── configuration.py  # Per-resource configuration schemas
│   └── internal.py       # Internal-only schemas (queue payloads, etc.)
├── crud/                # Per-resource DB operations
│   ├── collection.py, deriver.py, document.py, message.py
│   ├── peer.py, peer_card.py, representation.py  (RepresentationManager)
│   ├── session.py, webhook.py, workspace.py
├── routers/             # FastAPI route handlers (all under /v3)
│   ├── workspaces.py, peers.py (dialectic /chat lives here), sessions.py
│   ├── messages.py, conclusions.py, keys.py, webhooks.py
├── dialectic/           # Dialectic agent — runs inline per chat request
│   ├── chat.py           # agentic_chat() / agentic_chat_stream()
│   ├── core.py           # DialecticAgent (the tool-loop driver)
│   └── prompts.py
├── deriver/             # Background queue consumer (separate process)
│   ├── __main__.py       # `python -m src.deriver` entry point
│   ├── queue_manager.py  # QueueManager + main() loop
│   ├── consumer.py       # process_item dispatcher (representation / deletion / reconciler)
│   ├── deriver.py        # "minimal deriver" batch processor: extraction call, then admission call
│   ├── enqueue.py        # API → queue producer
│   └── prompts.py
├── dreamer/             # Memory consolidation (runs off the queue)
│   ├── orchestrator.py   # run_dream() / process_dream()
│   ├── specialists.py    # DeductionSpecialist + InductionSpecialist
│   ├── dream_scheduler.py
│   ├── surprisal.py      # Surprisal-based conclusion prioritization
│   └── trees/            # Reasoning-tree primitives
├── reconciler/          # In-process scheduler hosted by the deriver worker
│   ├── scheduler.py      # ReconcilerScheduler (started from queue_manager.py)
│   ├── sync_vectors.py   # Embeds MessageEmbedding rows with sync_state='pending'
│   ├── backfill.py       # Registered predicate-driven backfills
│   └── queue_cleanup.py  # Removes stale queue items
├── llm/                 # Provider-agnostic LLM client subsystem
│   ├── api.py, backend.py, executor.py, runtime.py, registry.py
│   ├── caching.py, structured_output.py, tool_loop.py, conversation.py
│   ├── history_adapters.py, request_builder.py, credentials.py, types.py
│   └── backends/         # anthropic.py, gemini.py, openai.py
├── cache/               # Redis cache abstraction (cashews-backed)
│   └── client.py
├── vector_store/        # Optional external vector stores (pgvector is default,
│   │                    #   implemented via MessageEmbedding/Document in models+crud)
│   ├── lancedb.py
│   └── turbopuffer.py
├── telemetry/           # Observability
│   ├── emitter.py        # CloudEvents emitter
│   ├── logging.py        # Logging helpers + route-template extraction
│   ├── metrics_collector.py, reasoning_traces.py, sentry.py
│   ├── events/           # Event type definitions
│   └── prometheus/       # Prometheus metric definitions
├── utils/               # Cross-cutting utilities
│   ├── agent_tools.py    # Tool definitions + per-agent tool lists
│   ├── summarizer.py     # Two-tier session summarizer
│   ├── representation.py # Representation formatting (distinct from crud/representation.py)
│   ├── search.py, filter.py, formatting.py
│   ├── tokens.py         # tiktoken-based counting
│   ├── pagination.py     # List pagination: opt-in cursor mode + the cached offset shim
│   ├── work_unit.py, queue_payload.py
│   ├── config_helpers.py, json_parser.py, files.py
│   └── types.py
└── webhooks/            # Webhook delivery
    ├── events.py
    └── webhook_delivery.py
```

- Tests run in pytest with fixtures in tests/conftest.py. Subdirectories mirror src/ (`tests/deriver/`, `tests/dialectic/` and the rest). There are also `tests/bench/` (perf benchmarks), `tests/integration/`, `tests/live_llm/` (gated by `--live-llm`) and `tests/unified/` (the unified runner).
- Use environment variables via python-dotenv (`.env`). Declared configuration precedence: env > `.env` > `config.toml` > defaults. In practice `.env` outranks the process environment (see Local Test Environment).

### Database Design

- All tables use text IDs (nanoid format) as primary keys
- Composite foreign keys for multi-tenant relationships
- Feature flags on workspace, peer, and session levels
- Token counting on messages for usage tracking
- JSONB metadata fields for extensibility
- HNSW indexes for vector similarity search
- **A `Select` that feeds a list endpoint MUST have a total ORDER BY that ends in a unique column** (for example `created_at, id`). `paginate_offset_or_cursor` (`src/utils/pagination.py`) seeks by the sort key in both cursor mode and the offset shim. Without a unique tail, rows that share a sort key are skipped or repeated across pages.

### Key Architectural Decisions

1. **Peer model**: humans and AI agents are unified as "Peers", many-to-many with Sessions. Internal vector storage (Collections/Documents) is keyed by `(observer, observed)` peer pairs. The same mechanism powers self-representation (`observer == observed`) and cross-peer modeling.
2. **Multi-Peer Sessions**: Sessions can have multiple participants with different observation configuration.
3. **API server / worker split**: API enqueues, deriver worker process consumes. Never block HTTP on LLM work. The Reconciler runs as an in-process scheduler inside the deriver. It handles async embedding sync, queue cleanup and backfills.
4. **"Minimal" deriver**: memory formation is structured output rather than an agentic tool loop: one call to extract and, in this fork, a second to admit. Predictable cost, lower latency. The Dialectic is the one true tool-using agent.
5. **Provider-agnostic LLM layer** (`src/llm/`): all model calls go through `honcho_llm_call()`. Backends (`anthropic`, `gemini`, `openai`) sit behind a registry. Per-agent `MODEL_CONFIG` with fallback chains is resolved at call time.
6. **Dialectic reasoning tiers**: 5 levels (`minimal` → `max`). Each level has its own model configuration and tool set, and `minimal` uses a reduced toolset.
7. **Hybrid search**: Postgres FTS plus vector similarity. The FTS side is a GIN index on `to_tsvector('english', content)`, combined by OR with a `content ILIKE` fallback. A `pg_trgm` GIN index backs that fallback. Without pg_trgm, the index is skipped and the ILIKE scans. Vector similarity is HNSW on `MessageEmbedding.embedding`. `MessageEmbedding` is a separate table from `Message`, with its own `sync_state`, so embedding is decoupled from message creation.
8. **Pluggable external vector stores**: defaults to pgvector inline, and can swap to turbopuffer or lancedb (`VECTOR_STORE_*` configuration, `src/vector_store/`).
9. **Composite-FK multi-tenancy**: `workspace_name` participates in nearly every composite FK. Cross-workspace data leakage is structurally impossible at the schema level.
10. **Scoped Authentication**: JWTs can be scoped to workspace, peer, or session level.
11. **Batch Operations**: Bulk message creation up to 100 messages per request.
12. **Session History**: Two-tier summarization: short every `SUMMARY_MESSAGES_PER_SHORT_SUMMARY` (default 20), long every `SUMMARY_MESSAGES_PER_LONG_SUMMARY` (default 60).

### Error Handling

- Custom exceptions defined in src/exceptions.py
- Use specific exception types, for example ResourceNotFoundException and ValidationException
- Proper logging with context instead of print statements
- Global exception handlers defined in main.py
- See docs/contributing/error-handling.mdx for details

### Notes

- Always use `uv run` or `uv` to prefix any commands related to python to ensure you use the virtual environment
