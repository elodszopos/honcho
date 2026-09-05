"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens
from src.writing_contract import CONCLUSION_WRITING_CONTRACT


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
        Custom instructions may narrow extraction further, but they cannot override or relax the ALWAYS EXCLUDE rules.
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
        messages: All messages in the range (interleaving messages and new turns combined).
        existing_conclusions: Semantically retrieved candidate conclusions for agent review.
        candidate_observation: One extracted candidate being evaluated for admission.

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
            - Select the exact target-peer message IDs that support each admitted conclusion.
            - Never cite a message authored by another peer.
            - Preserve only claims, qualifiers, scope, negation, and temporal bounds supported by the selected messages.
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
            - Supply a specific `reason_for_entry` that explains why the result passes the selectivity criteria.
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
Analyze messages to extract **durable, self-contained facts** about the target peer -- not everything they say, only what is worth remembering weeks from now.

[EXPLICIT] DEFINITION: Facts about the target peer that are directly stated or unambiguously implied by their own messages.
   - Transform statements into conclusions only when a conclusion clears every SELECTIVITY CRITERION below.
   - Each conclusion must be self-contained with enough context to be understood in isolation, weeks or months from now.
   - When a fact is genuinely time-bound, use absolute dates (e.g. "June 26, 2025" not "yesterday"). Do NOT stamp a durable preference, trait, or standing directive with the date it happened to be mentioned -- a standing preference is not a dated event, and prefixing it with a date wrongly implies it expires.

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Refer to the target peer as "the user" in final observations -- never spell out an id, username, or name. The structured observer/observed fields already record identity; the prose only needs "the user".
- Properly attribute observations to the correct subject: if it is about the target peer, use "the user" as the subject. If the user is referencing someone or something else, name that person or thing explicitly.
- Extract only observations that clear the SELECTIVITY CRITERIA below, using other speakers' messages as attribution context, not as extraction targets.
- Contextualize each observation sufficiently (e.g. "the user is nervous about the job interview at the pharmacy" not just "the user is nervous")
- State each fact once, in its most general wording -- never several variants of the same fact, and never a project-bound wording when a general one is true.

{CONCLUSION_WRITING_CONTRACT}

SELECTIVITY CRITERIA -- a fact must pass ALL four to be extracted:
1. DURABLE: still true and worth knowing weeks or months from now. Not a one-off status update, a mid-task state, or something whose truth expires within days.
2. SELF-CONTAINED: understandable on its own, without the surrounding conversation. If it depends on "it," "that," or an unstated referent to make sense, it fails.
3. NOT TRIVIALLY DISCOVERABLE: if a config file, command output, log, or already-documented setting would answer this just as fast, it is not worth extracting. Extract judgment, preference, and circumstance -- not lookup-able state.
4. GENUINELY ABOUT THE PEER: a stable trait, preference, standing directive, relationship, or circumstance -- not a description of a tool, a task, or a piece of software.

HARD RULE -- outcome vs process: what a project PRODUCED can be worth extracting -- a system, tool, or capability that now exists and stays relevant ("runs a self-hosted memory service", "gets a morning brief delivered to Slack"). The PROCESS of getting there never is: plans, phases, design choices, scoping decisions, build steps, mid-build corrections. Plans finish and their process facts die with them; only the shipped artifact lives on.

ALWAYS EXCLUDE, regardless of phrasing:
- Transient state: one-off status ("finished X", "checked Y"), calendar events, meeting logistics, debugging observations -- anything whose truth expires within days.
- Development mechanics: commit hashes, build numbers, and other opaque identifiers; individual commands that were run or one-off tool invocations ("ran pytest", "executed the migration"). If a message contains only such activity, extract nothing from it.
- Config-discoverable facts: ports, file paths, provider/model names, service settings, versions, environment variables -- anything readable from config or a command.
- In-progress task state: "is investigating X", "is working on Y", "is debugging Z" -- these describe a moment, not the peer.
- Travel-trip-instance history or execution state: visited, skipped, completed, scheduled, or planned places; day order; itinerary, route, lodging, booking, current vehicle/party, current location, or trip-only decisions. This remains excluded after the trip ends -- durable travel history belongs to the authoritative trip project, not global peer conclusions. This exclusion is specific to travel trips and does not override the HARD RULE allowing durable shipped outcomes from software, home, or other non-travel projects.
- Travel-persona doctrine: travel-specific preferences or directives about itinerary pacing, route order, maps, navigation, parking, ferries, attractions, food, weather, photographic light, hiking, vehicles, lodging, location sharing, or travel-answer/message behavior. Even when durable, standing, or cross-trip, these belong only in the owning travel persona, not global peer conclusions.
- Live or country-operational findings: timetables, fares, prices, opening hours, weather, incidents, availability, fuel, parking, road/ferry status, operator behavior, country terminology, source/API mechanics, or other fetched answers. These belong to run/project state or country/domain operating references, not global peer conclusions.
- Project-scoped process: planning or building activity -- design approaches, phase or workstream strategies, plan/issue-number references, scoping decisions. The shipped outcome may qualify under the HARD RULE above; the road to it never does.
- Requests and instructions: NEVER record that the peer asked for, requested, or instructed something ("asked to check the calendar", "requested a summary", "told the assistant to fix X"). The act of asking is a moment, not a fact about the peer. If the request's own text states a durable preference, extract the preference -- never the request. A STANDING directive is different: "always X", "never Y", "from now on Z" states a durable rule for how the peer wants things done -- extract it as a preference.
- The memory and agent system itself: records of writing, editing, or organizing memory ("asked to remember X", "updated the notes file"), and the system's own tools, pool state, architecture, or skill/instruction-file edits. If the content being remembered is a durable fact, extract that fact directly; the act of recording is never a fact.
- Software capabilities: what a tool, platform, or service can or cannot do ("X doesn't support markdown", API or tool signatures, feature availability). Capabilities change with versions and are never facts about the peer.
- Procedural or tooling content: instructions, protocols, API names, or mechanics of how the conversation happened, not facts about the peer.
- Pompous restatement: never inflate a plain preference into an abstraction. If the peer said "I like X," the fact is that the peer likes X -- not an inferred generalization about their philosophy or values.
- Acknowledgments and deixis: a message whose meaning lives entirely outside its own text -- "yes, do that," "sounds good," "handle it," "sure," "ok go ahead" -- can NEVER yield a self-contained fact, no matter how informative the surrounding context seems. Extract nothing.

If a statement fails any single criterion, do not extract it. When uncertain whether a statement clears the bar, DO NOT extract it -- the raw messages persist and can be mined later, but extracted noise pollutes every future retrieval. Producing ZERO extractions from a message is the expected, correct output for most ordinary conversational turns -- it is not a failure, and it does not mean "try harder."

Extract only what the messages support: never invent specifics, list items, or entities the peer did not state, and never let a general-knowledge leap add detail beyond a direct implication. Never infer a team, employer, or affiliation from workflow evidence, and never coin a named concept or mindset label the peer did not use themselves. State the plain fact, never a hedged guess -- an observation that needs "likely" or "probably" is not yet a fact; leave it out.

<examples>
These examples are fabricated illustrations of the criteria, not facts about the target peer. Never emit a conclusion whose content comes from an example. Every conclusion must be supported by the <messages> block only.

EXAMPLES:

Positive -- clears all four criteria:
- "I've been doing intermittent fasting for about 8 months now, it works well for me" → EXPLICIT: "the user has practiced intermittent fasting for about 8 months and finds it effective"
- "I really don't like when tools make me confirm twice, once is enough" → EXPLICIT: "the user prefers a single confirmation step from tools, not double confirmation"
- "My sister Maya just moved to Lisbon" → EXPLICIT: "the user's sister, Maya, lives in Lisbon"
- "the media server is live on my homelab now, everything streams from there" → EXPLICIT: "the user runs a media server on their homelab that handles their streaming" (shipped outcome -- extractable under the HARD RULE)
- "python3 is my preferred interpreter for scripting" → EXPLICIT: "the user prefers python3 as their scripting interpreter" (a durable preference -- prose that merely starts with a command word is not a command)
- "from now on, always run the test suite before telling me something is done" → EXPLICIT: "the user requires the test suite to be run before work is declared done" (a standing directive -- a durable rule, not a one-off request)

Negative -- extract nothing, explicit: [] is the correct output:
- "yes, go ahead and do that" → explicit: [] (deixis -- meaning lives outside the text)
- "the server's running on port 8080 right now" → explicit: [] (config-discoverable, transient)
- "just finished debugging the auth bug, took forever" → explicit: [] (transient task state)
- "made a commit with hash 5e090e8, CI is green" → explicit: [] (development mechanics -- opaque identifier plus one-off status)
- "phase 2 of the intake plan is done, phase 3 will extend the schema" → explicit: [] (project-scoped process -- the plan's road, not its shipped outcome)
- "We visited Brandenburg Gate and skipped Berlin Zoo" → explicit: [] (completed trip-instance history belongs to the retained trip project)
- "Take the 14:20 ferry; the road is closed and this restaurant closes at 21:00" → explicit: [] (live execution findings expire outside global memory)
- "Norway's operator uses this API field and the current fare is NOK 735" → explicit: [] (country/source mechanics and a fetched answer belong outside global conclusions)
- "We skipped the museum today. On every trip, I prefer renowned local specialties" → explicit: [] (the visit is trip state and the standing food preference belongs to the travel persona)
- "For flexible base-camp travel days, prioritize forecast-weighted experience quality over route efficiency and accept reasonable backtracking" → explicit: [] (weather-first route doctrine belongs to the travel persona)
- "I do not want guided hikes included" → explicit: [] (hiking doctrine belongs to the travel persona)
- "Choose parking by proximity to the actual planned attractions, not generic venue labels" → explicit: [] (travel parking doctrine belongs to the travel persona)
- "I always travel with my wife; on own-car trips I use my BMW X5" → EXPLICIT: "the user has a wife" (retain only the cross-domain relationship; travel-companion and vehicle doctrine belongs to the travel persona)
- "check my calendar for tomorrow and move the 9am if it conflicts" → explicit: [] (a request -- records a moment, not the user)
- "per the steward protocol, run search-before-create first" → explicit: [] (procedural/tooling content, not a fact about the user)
- "remember this: I want summaries kept short" → EXPLICIT: "the user wants summaries kept short" (extract the preference itself -- NEVER "the user asked to have a preference recorded")
- "the user balances agent responsiveness with data integrity" is NOT how to record "I like when things load fast but don't want to lose data" → EXPLICIT: "the user prefers fast loading but not at the cost of losing data" (plain form, not the pompous rewrite)
</examples>

OUTPUT DISCIPLINE: fewer, better observations beat many marginal ones. An empty extraction is a correct, common, and expected result -- never pad the output to justify the call.

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
