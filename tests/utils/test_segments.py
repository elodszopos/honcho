"""A query splits into the thoughts it carries: one per line, then one per sentence."""

import pytest

from src.utils.segments import MAX_SEGMENTS, split_thoughts


@pytest.fixture(autouse=True)
def clean_queue_tables():
    """Splitting is DB-free; override the package's DB fixture."""
    yield


def test_lines_then_sentences():
    text = (
        "it should attach on the next turn, no matter what. and what about the cut?\n"
        "I need some ideas on what you think is wrong here."
    )

    assert split_thoughts(text) == [
        "it should attach on the next turn, no matter what.",
        "and what about the cut?",
        "I need some ideas on what you think is wrong here.",
    ]


def test_a_short_fragment_joins_the_thought_before_it():
    thoughts = split_thoughts("cutting off results with 0.533?! that's a bug?? ok")

    assert thoughts == ["cutting off results with 0.533?! that's a bug?? ok"]


def test_a_short_leading_fragment_stands_alone():
    assert split_thoughts(
        "1 done\n2 maybe this is a lack of conclusions after all."
    ) == [
        "1 done",
        "2 maybe this is a lack of conclusions after all.",
    ]


def test_blank_input_yields_nothing_and_a_bare_word_yields_itself():
    assert split_thoughts("  \n ") == []
    assert split_thoughts("ok") == ["ok"]


def test_thoughts_are_capped():
    text = "\n".join(f"thought number {i} stands on its own line." for i in range(30))

    assert len(split_thoughts(text)) == MAX_SEGMENTS
