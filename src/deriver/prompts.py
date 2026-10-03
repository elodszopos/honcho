"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

import re
from datetime import datetime
from functools import lru_cache
from inspect import cleandoc as c
from typing import Any

from src.utils.tokens import estimate_tokens
from src.writing_contract import CONCLUSION_WRITING_CONTRACT

_MESSAGE_TAG = re.compile(r"<(?=/?message\b)", re.IGNORECASE)


def format_deriver_message(
    idx: int,
    message_id: int,
    peer: str,
    target: str,
    created_at: datetime,
    content: str,
) -> str:
    """Wrap one batch message in a tag carrying its ids, author, and target flag.

    ``message_id`` is the database id the admission pass cites in
    ``source_message_ids``; ``idx`` is batch position only. Peer ids are
    restricted to ``[a-zA-Z0-9_-]`` upstream, so only the content can carry
    markup; any ``<message``/``</message`` inside it is neutralized so a message
    cannot forge its own tag boundary.
    """
    is_target = "true" if peer == target else "false"
    time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
    safe_content = _MESSAGE_TAG.sub("&lt;", content)
    return (
        f'<message idx="{idx}" message_id="{message_id}" peer="{peer}" '
        f'target="{is_target}" time="{time_str}">'
        f"{safe_content}</message>"
    )


def _normalized_custom_instructions(custom_instructions: str | None) -> str | None:
    """Return stripped custom instructions, if any."""
    if custom_instructions is None:
        return None

    normalized = custom_instructions.strip()
    return normalized or None


def _custom_instructions_section(custom_instructions: str | None) -> str:
    """Render optional custom instructions for the deriver prompt."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return ""

    return c(
        f"""
        CUSTOM INSTRUCTIONS:
        - These instructions apply to the target peer named in the user turn.
        - They may narrow extraction further; they cannot override or relax the NEVER EXTRACT rules.
        {normalized_custom_instructions}
        """
    )


_ADMISSION_RULES = c(
    """
    MANDATORY LOOK-BEFORE-WRITE ADMISSION:
    - Review every admission case as an independent candidate for its specified observer.
    - Treat the existing conclusions in each case as retrieval only: semantic search found them before this decision.
    - Never create, merge, enrich, or discard a memory because of an arbitrary similarity threshold.
    - Read every retrieved conclusion in that case before proposing an entry.
    - Return `action: "create"` and `target_id: null` when none expresses the same durable memory.
    - Return `action: "enrich"` with `target_id` set to that conclusion id when one expresses the same memory.
    - For enrichment, write the best current formulation from the retrieved conclusion and the new evidence.
    - Never return a second semantic version of an existing memory.
    - Never target a conclusion id that is absent from that admission case.
    - Copy the case's `admission_case_id` into its decision.
    - Return at most one decision per admission case.
    - Write `reason_for_entry` as one clause under 120 characters: what the memory is for.
    - Return no decision for a case when the messages do not justify durable memory.
    - Return all admitted cases together in one `explicit` list.

    SOURCE-GROUNDING RULES:
    - Re-read the original messages before deciding.
    - Use only message IDs shown in the original messages.
    - Cite the user's message that states the conclusion.
      - For another peer's statement, cite it and the user's message that explicitly agrees with it or asks to remember it.
    - Return no decision when only a go-ahead, silence or a new topic follows another peer's statement.
    - Preserve only claims, qualifiers, scope, negation and temporal bounds the cited messages support.
    - Return no decision when an extracted candidate overstates or misattributes the source.
    """
)


_CURATED_MEMORY_RULES = c(
    """
    REFERENCE, NEVER A SOURCE:
    - The USER.md block below is what the assistant already knows about the user.
    - Use it only to recognise a fact that is already held.
    - Never extract a fact the USER.md block already states.
    - Extract nothing from it; every conclusion comes from the <messages> block alone.
    """
)


def deriver_system_prompt(
    custom_instructions: str | None = None,
    *,
    admission: bool = False,
    curated_memory: str = "",
) -> str:
    """The static rules, then custom instructions and the user's curated memory, then the admission
    rules when this is the admission pass: both passes of a batch share one cached prefix through
    the curated memory, and an edit to it busts only what follows the edited line."""
    sections = [
        c(
            f"""
ROLE:
- Extract what a conversation weeks from now would need to know about the user and their life.
- Any area of life qualifies.

TARGET PEER AND MESSAGES:
- The target peer is named under `Target peer:` in the user turn.
- Call the target peer "the user" in every conclusion; never an id, username or name.
- Each message is wrapped as `<message idx="N" message_id="ID" peer="..." target="true|false" time="...">`.
  - `target="true"` marks the user's own messages.
  - `message_id` is the database id a conclusion cites; `idx` is never cited.
- A conclusion comes from the user's own words.
- Another peer's statement qualifies only when the user explicitly agrees with what it says or asks to remember it.
  - Letting a task go ahead, staying silent or moving on is not agreement with what was said.
- When the user asks, in any wording, to remember something, its content is a conclusion; only credentials are refused.
- A batch with no `target="true"` message yields nothing.
- Name other people, projects, jobs, systems and places explicitly.

WHAT A CONCLUSION IS:
- Something about the user that stays true: who they are, the people in their life, what they have, how they live and work, what they prefer.
- How the user wants things done for them from now on, as opposed to how one task went.
- What the user runs, uses, relies on or cares about, and why.
- A later pass files always-relevant ones elsewhere; extract them here.

WHEN TO EXTRACT:
- Extract a conclusion when it is still true and useful a month from now, beyond the task at hand.
- Extract nothing whose topic cannot be named; a conclusion without a topic is not one.
- Keep one wording per fact, the most general accurate one.
- Merge two candidates that say the same thing into one.
- Small talk, acknowledgements and status updates yield nothing.

HOW TO WRITE ONE:
- Name the topic inside the conclusion ("For the weekly newsletter, ...", "The user's garden ...").
  - Why: a conclusion is found by its topic.
- State one fact per conclusion.
- Keep a conclusion under 30 words; a second fact is a second conclusion.
- Make it self-contained: understandable months later without the conversation.
- Use plain wording: no hedging, no inflation, no coined label; "the user prefers X" never becomes a philosophy.
- Use absolute dates for time-bound facts, resolving relative references against the message `time` attribute ("June 26, 2025", never "yesterday").
- Put no date on a standing preference or rule.
- Keep the behavioral hook when there is one: when, do, avoid.

{CONCLUSION_WRITING_CONTRACT}

NEVER EXTRACT:
- Credentials, or anything else that unlocks an account.
- What is true only for now or for one occasion: a status, a task in progress, what happened or was decided once, and its figures.
  - How the user wants every such occasion handled can qualify.
- What another owner already holds by nature:
  - How a task is carried out, and when to use a skill or job, belong to the skill or job that does it.
  - Configured values belong to configuration.
  - How a system, tool or piece of code works or behaved, and design decisions for one system, belong to its code and docs.
  - Rules about what memory holds belong to the memory doctrine.
  - A trip's plans and history belong to the trip project, and travel preferences to the travel persona; this never reaches beyond travel.
  - Facts fetched from outside the conversation belong to their source.
- The act of asking, acknowledging or recording; extract the content, never the act.
- Text copied from tool output or a report.
- Guesses and invented specifics.

<examples>
Fabricated illustrations of the rules. Never emit a conclusion whose content comes from an example; every conclusion must be supported by the <messages> block only.

EXTRACT:
- assistant: "So you'd rather get the grocery list on Fridays instead of every day?" / user: "yes, exactly" → "For grocery lists, the user wants them on Fridays rather than daily." (the user agreed with that statement; cite both messages)
- "I run a backup every night so I never lose a day of photos" → "The user runs a nightly backup so they never lose more than a day of photos."
- "never book me a flight before 8 again, this morning's 6am was brutal" → "The user does not want flights booked before 8 am." (the standing rule, not the occasion)
- "remember that I lent Dan my ladder" → "The user lent their ladder to Dan." (an explicit request to remember)
- assistant: "Let's schedule a nightly router reboot." / user: "no, the alarm system runs through that router, a reboot takes it offline" → "The user's home alarm system runs through their router; rebooting the router takes the alarm offline." (what the user relies on)
- "My sister Maya just moved to Lisbon" → "The user's sister, Maya, lives in Lisbon."
- "the accountant files the quarterly VAT return, I only send her the invoices" → "For taxes, the user's accountant files the quarterly VAT return; the user sends her the invoices."
- "from now on, always run the test suite before telling me something is done" → "The user requires the test suite to be run before work is declared done."
- "I always travel with my husband; on road trips we take the camper" → "The user has a husband." (only the cross-domain fact; travel companions and vehicles belong to the travel persona)

NOTHING, explicit: [] is the correct output:
- "yes, go ahead and do that" (the act of agreeing)
- assistant: "The plumber's invoice is 3 hours, 240 euros, due Friday." / user: "pay it" (one occasion, and letting it go ahead is not agreement)
- assistant: "Eco mode runs longer because it heats the water less." / user: "ok" (how a machine works)
- "point the backup at the archive folder on the second disk" (a configured value)
- "the newsletter goes out Tuesdays at 9 with the events section first" (how a task is carried out; whoever sends it owns that)
- "the photo-sorting job mislabeled last week's pictures, fix that" (how a job behaved once)
- "just finished debugging the auth bug, took forever" (status)
- "We visited the cathedral and skipped the zoo" (a trip's history)
- "On every trip I want local specialties, never chains" (a travel preference)
- "Take the 14:20 ferry; the road is closed and this restaurant closes at 21:00" (facts fetched from outside)
- `<message target="false" peer="assistant">I read the config file and found the port is 8080</message>` with no user reply (a value nobody kept)
</examples>
"""
        )
    ]
    custom = _custom_instructions_section(custom_instructions)
    if custom:
        sections.append(custom)
    if curated_memory:
        sections.append(_CURATED_MEMORY_RULES)
        sections.append(curated_memory)
    if admission:
        sections.append(_ADMISSION_RULES)
    return "\n\n".join(sections)


def deriver_user_prompt(
    peer_id: str,
    messages: str,
    existing_conclusions: str | None = None,
    candidate_observation: str | None = None,
) -> str:
    """The batch under review, and the admission cases when this is the admission pass."""
    sections = [
        c(
            f"""
            Target peer:
            {peer_id}

            Messages to analyze:
            <messages>
            {messages}
            </messages>
            """
        )
    ]
    if existing_conclusions is not None:
        sections.append(
            c(
                f"""
                SEARCH RESULTS BY ADMISSION CASE:
                <existing_conclusions>
                {existing_conclusions or "(none)"}
                </existing_conclusions>

                ADMISSION CASES UNDER REVIEW:
                <admission_cases>
                {candidate_observation or "(missing -- admit nothing)"}
                </admission_cases>
                """
            )
        )
    return "\n\n".join(sections)


def deriver_messages(
    peer_id: str,
    messages: str,
    existing_conclusions: str | None = None,
    candidate_observation: str | None = None,
    custom_instructions: str | None = None,
    curated_memory: str = "",
) -> list[dict[str, Any]]:
    """System rules plus the user turn, ready for the model call."""
    return [
        {
            "role": "system",
            "content": deriver_system_prompt(
                custom_instructions,
                admission=existing_conclusions is not None,
                curated_memory=curated_memory,
            ),
        },
        {
            "role": "user",
            "content": deriver_user_prompt(
                peer_id, messages, existing_conclusions, candidate_observation
            ),
        },
    ]


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    existing_conclusions: str | None = None,
    candidate_observation: str | None = None,
    custom_instructions: str | None = None,
    curated_memory: str = "",
) -> str:
    """The whole prompt as one text: the system rules followed by the user turn."""
    return "\n\n".join(
        message["content"]
        for message in deriver_messages(
            peer_id,
            messages,
            existing_conclusions,
            candidate_observation,
            custom_instructions,
            curated_memory,
        )
    )


@lru_cache(maxsize=16)
def _estimate_scaffold_tokens(
    custom_instructions: str | None, curated_memory: str
) -> int:
    return estimate_tokens(
        minimal_deriver_prompt(
            peer_id="",
            messages="",
            custom_instructions=custom_instructions,
            curated_memory=curated_memory,
        )
    )


def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    return _estimate_scaffold_tokens(None, "")


def estimate_deriver_prompt_tokens(
    custom_instructions: str | None, curated_memory: str = ""
) -> int:
    """Tokens of everything in the prompt but the batch: the rules, the custom instructions and
    the curated memory block."""
    return _estimate_scaffold_tokens(
        _normalized_custom_instructions(custom_instructions), curated_memory
    )
