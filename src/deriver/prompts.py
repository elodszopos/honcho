"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

import re
from datetime import datetime
from functools import cache
from inspect import cleandoc as c

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
        These instructions apply to the target peer identified below.
        Custom instructions may narrow extraction further, but they cannot override or relax the NEVER EXTRACT rules.
        {normalized_custom_instructions}
        """
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    existing_conclusions: str | None = None,
    candidate_observation: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast, selective observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: Batch messages, each wrapped by ``format_deriver_message``.
        existing_conclusions: Semantically retrieved candidate conclusions for agent review.
        candidate_observation: One extracted candidate being evaluated for admission.
        custom_instructions: Optional narrowing instructions for the target peer.

    Returns:
        Formatted prompt string for observation extraction.
    """
    custom_instructions_section = _custom_instructions_section(custom_instructions)
    admission_section = ""
    if existing_conclusions is not None:
        candidate_section = c(
            f"""
            ADMISSION CASES UNDER REVIEW:
            <admission_cases>
            {candidate_observation or "(missing -- admit nothing)"}
            </admission_cases>

            SOURCE-GROUNDING RULES:
            - Re-read the original messages below before deciding.
            - Use only message IDs shown in the original messages.
            - Cite the message that states the conclusion and the user's message that asked for it, confirmed it or built on it; at least one cited message is the user's.
            - Preserve only claims, qualifiers, scope, negation and temporal bounds the cited messages support.
            - Return no decision when an extracted candidate overstates or misattributes the source.
            """
        )
        admission_section = c(
            f"""
            MANDATORY LOOK-BEFORE-WRITE ADMISSION:
            - Review every admission case as an independent candidate for its specified observer.
            - The existing conclusions in each case were retrieved by semantic search before this decision.
            - Cosine similarity is retrieval only.
            - Never create, merge, enrich, or discard a memory because of an arbitrary similarity threshold.
            - Read every retrieved conclusion in that case before proposing an entry.
            - If none expresses the same durable memory, return `action: "create"` and `target_id: null`.
            - If one expresses the same memory, return `action: "enrich"` and set `target_id` to that conclusion id.
            - For enrichment, write the best current formulation using the retrieved conclusion and the new evidence.
            - Never return a second semantic version of an existing memory.
            - Never target a conclusion id that is absent from that admission case.
            - Copy the case's `admission_case_id` into its decision.
            - Return at most one decision per admission case.
            - Supply a specific `reason_for_entry` that names what the conclusion is for and why it lasts.
            - Return no decision for a case when the messages do not justify durable memory.
            - Return all admitted cases together in one `explicit` list.

            SEARCH RESULTS BY ADMISSION CASE:
            <existing_conclusions>
            {existing_conclusions or "(none)"}
            </existing_conclusions>

            {candidate_section}
            """
        )
    return c(
        f"""
Extract the conclusions this conversation reached that a later conversation on the same topic would need, and the facts it established about the target peer.

TARGET PEER AND MESSAGES:
- The target peer is named under `Target peer:` below and is called "the user" in every conclusion; never an id, username or name.
- Each message is wrapped as `<message idx="N" message_id="ID" peer="..." target="true|false" time="...">`. `target="true"` marks the user's own messages. `message_id` is the database id a conclusion cites; `idx` is never cited.
- A conclusion may come from any peer's message. One another peer stated counts once the user accepted it: agreed, built on it, or continued without contradicting it.
- A batch with no `target="true"` message yields nothing: nothing in it was accepted.
- Name other people, projects, jobs, systems and places explicitly.

WHAT A CONCLUSION IS, extract every one that qualifies:
- A fact the conversation established about a topic: what exists, how something works, a finding, a root cause, an outcome.
- A decision, ruling or requirement for a stream of work: how the user wants a specific thing done, what was agreed, what was rejected.
- A pointer: where something lives, a job, skill, file, channel, path or name, when a later conversation would look for it.
- The user's relationship to a topic: what they run, maintain, use or care about, and why.
- A fact about the user: a preference, trait, relationship or circumstance. A later pass files always-relevant ones elsewhere; extract them here.
- Any topic qualifies: work, projects, home, health, money, people, hobbies.

HOW TO WRITE ONE:
- Name the topic inside the conclusion so it is found when the topic comes up ("For the upstream gap jobs, ...", "The user's home network ...").
- One fact per conclusion, self-contained, understandable months later without the conversation.
- Plain wording, no hedging, no inflation; "the user prefers X" never becomes a philosophy or a coined label.
- Absolute dates for time-bound facts ("June 26, 2025", never "yesterday"); no date on a standing preference or rule.
- Keep the behavioral hook when there is one: when, do, avoid.

{CONCLUSION_WRITING_CONTRACT}

NEVER EXTRACT:
- Passwords, API keys, tokens, cookies, private keys, recovery codes and other authenticators.
- Transient state: a status ("finished X", "checked Y"), progress, an in-progress task, a scheduled event, a debugging observation; anything true only for days.
- Values that change often: counts, run results, run timestamps, version numbers, current numeric settings. The stable name, path or job that holds them may be a pointer.
- The act of asking, acknowledging or recording: "asked to check", "sounds good", "asked to remember". Extract the content, never the act.
- Text copied from tool output or a report; extract the finding, not the transcript.
- Guesses: anything that needs "likely" or "probably"; invented specifics, entities or affiliations; a mindset label the user never used.
- Travel-trip-instance history or execution state: visited, skipped, completed, scheduled, or planned places; day order; itinerary, route, lodging, booking, current vehicle or party, current location, trip-only decisions. This holds after the trip ends: durable travel history belongs to the authoritative trip project. This exclusion is specific to travel trips and never covers a software, home or other non-travel outcome.
- Travel-persona doctrine: travel-specific preferences or directives about itinerary pacing, route order, maps, navigation, parking, ferries, attractions, food, weather, photographic light, hiking, vehicles, lodging, location sharing, or travel-answer/message behavior. Even when durable, standing, or cross-trip, these belong only in the owning travel persona.
- Live or country-operational findings: timetables, fares, prices, opening hours, weather, incidents, availability, fuel, parking, road/ferry status, operator behavior, country terminology, source/API mechanics, or other fetched answers.

<examples>
Fabricated illustrations of the rules. Never emit a conclusion whose content comes from an example; every conclusion must be supported by the <messages> block only.

EXTRACT:
- user: "Do the gap jobs read the right PATCHES.md?" / assistant: "Yes: the Hermes job reads its repo's PATCHES.md and the Honcho job reads its own." / user: "good" → "For the upstream gap jobs, each daily job reads its own repository's PATCHES.md." (assistant-stated, user accepted; cite both messages)
- "I run the gap reports daily for both forks so I know what a merge costs" → "The user runs daily upstream-gap reports for the Hermes and Honcho forks to judge what a merge costs."
- "from now on a Honcho answer that misses its wait attaches on the next turn, even an 'ok'" → "For Hermes lane recall, a Honcho answer that misses its first-turn wait attaches on the next message, trivial or not." (a ruling for a stream of work)
- "the media server is live on my homelab now, everything streams from there" → "The user runs a media server on their homelab that handles their streaming."
- "My sister Maya just moved to Lisbon" → "The user's sister, Maya, lives in Lisbon."
- "the accountant files the quarterly VAT return, I only send her the invoices" → "For taxes, the user's accountant files the quarterly VAT return; the user sends her the invoices."
- "from now on, always run the test suite before telling me something is done" → "The user requires the test suite to be run before work is declared done."
- "I always travel with my wife; on own-car trips I use my BMW X5" → "The user has a wife." (only the cross-domain fact; travel-companion and vehicle doctrine belongs to the travel persona)

NOTHING, explicit: [] is the correct output:
- "yes, go ahead and do that" (the act of agreeing)
- "the server's running on port 8080 right now" (a current value)
- "just finished debugging the auth bug, took forever" (status)
- "We visited Brandenburg Gate and skipped Berlin Zoo" (trip instance)
- "We skipped the museum today. On every trip, I prefer renowned local specialties" (the visit is trip state and the standing food preference belongs to the travel persona)
- "For flexible base-camp travel days, prioritize forecast-weighted experience quality over route efficiency" (travel persona)
- "I do not want guided hikes included" (travel persona)
- "Choose parking by proximity to the actual planned attractions, not generic venue labels" (travel persona)
- "Take the 14:20 ferry; the road is closed and this restaurant closes at 21:00" (live findings)
- "Norway's operator uses this API field and the current fare is NOK 735" (country mechanics and a fetched answer)
- `<message target="false" peer="assistant">I read the config file and found the port is 8080</message>` with no user reply (a value nobody kept)
</examples>

OUTPUT:
- Small talk, acknowledgements and pure status updates yield nothing.
- A working discussion usually yields one to a few conclusions; extract each one that qualifies, never pad and never drop a qualifying one to stay short.

{admission_section}

{custom_instructions_section}

Target peer:
{peer_id}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=None,
    )
    return estimate_tokens(prompt)


def estimate_deriver_prompt_tokens(custom_instructions: str | None) -> int:
    """Estimate minimal deriver prompt tokens, including custom instructions if present."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return estimate_minimal_deriver_prompt_tokens()

    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=normalized_custom_instructions,
    )
    return estimate_tokens(prompt)
