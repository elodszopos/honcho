"""Shared writing rules for text produced or consumed by Honcho LLMs."""

from inspect import cleandoc as c

CONCLUSION_TARGET_CHARS = 500
MAX_CONCLUSION_CHARS = 800

LLM_CONSUMED_WRITING_CONTRACT = c(
    """
    ## LLM-CONSUMED WRITING CONTRACT

    - Assume another LLM consumes this text unless the prompt explicitly identifies a human reader.
    - Lead with the result, fact, decision, contradiction, or gap.
    - Use compact headings, atomic bullets, checklists, or tables when they improve scanning.
    - Keep one fact, instruction, or decision per sentence or bullet.
    - Preserve material facts, bounds, exceptions, risks, uncertainty, and required actions.
    - Remove repetition, filler, narrative, process commentary, and meta-explanation.
    - Use natural prose only when the prompt identifies a human reader or the user explicitly requests another style.
    - Stop when the requested output is complete.
    """
)

CONCLUSION_WRITING_CONTRACT = c(
    f"""
    {LLM_CONSUMED_WRITING_CONTRACT}

    ## CONCLUSION TEXT

    - Keep one durable retrieval need per conclusion.
    - Target {CONCLUSION_TARGET_CHARS} characters.
    - Never exceed {MAX_CONCLUSION_CHARS} characters.
    - Consolidate through semantic compression, not concatenation.
    - Reword freely; remove repetition, scaffolding, stale phrasing, and redundant examples.
    - Preserve supported decisions, bounds, exceptions, negative directives, and behavioral hooks.
    - Treat historical revisions as evidence; never restore one wholesale.
    - Split independent facts into separate conclusions.
    """
)
