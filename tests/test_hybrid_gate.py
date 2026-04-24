# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for hybrid_gate pure functions.

These do not call the judge or require the interpreter container — they
exercise only the response parsing, per-item synthesis, and reward combination.
The client no longer parses rubric text; the judge reads the raw rubric and
echoes each item's weight inline.
"""
from __future__ import annotations

import pytest

from hypotest.env.hybrid_gate import (
    hybrid_reward,
    parse_hybrid_response,
    synthesize_per_item_awards,
)


def _response(per_item: list[tuple[int, int, str]], summary: tuple[int, int, int, int] | None) -> str:
    """Build a mock judge response. per_item: [(idx, weight, PRESENT/ABSENT)]."""
    lines = [
        f"item {i} ({w} point{'s' if w != 1 else ''}): {v} \u2014 reason {i}"
        for i, w, v in per_item
    ]
    body = "\n".join(lines)
    if summary is None:
        return body
    return body + (
        f"\n\n<summary>\nprocedural_points_present={summary[0]} of {summary[1]}\n"
        f"conclusion_points_present={summary[2]} of {summary[3]}\n</summary>"
    )


# ---------- parse_hybrid_response ----------

def test_parse_hybrid_response_happy_path() -> None:
    resp = _response(
        [(1, 1, "PRESENT"), (2, 1, "PRESENT"), (3, 1, "ABSENT"), (4, 1, "ABSENT"), (5, 5, "PRESENT")],
        summary=(2, 4, 5, 5),
    )
    parsed = parse_hybrid_response(resp)
    assert parsed["proc_max_pts"] == 4
    assert parsed["concl_max_pts"] == 5
    assert parsed["proc_present_pts"] == 2
    assert parsed["concl_present_pts"] == 5
    assert len(parsed["per_item"]) == 5


def test_parse_hybrid_response_missing_summary_falls_back_to_per_item() -> None:
    resp = _response(
        [(1, 1, "PRESENT"), (2, 1, "ABSENT"), (3, 1, "ABSENT"), (4, 1, "ABSENT"), (5, 5, "ABSENT")],
        summary=None,
    )
    parsed = parse_hybrid_response(resp)
    assert parsed["proc_present_pts"] == 1
    assert parsed["concl_present_pts"] == 0


def test_parse_hybrid_response_mismatched_summary_max_falls_back() -> None:
    # Summary maxes don't agree with what we got per-item; ignore summary totals.
    resp = _response(
        [(1, 1, "PRESENT"), (2, 1, "PRESENT"), (3, 1, "ABSENT"), (4, 1, "ABSENT"), (5, 5, "ABSENT")],
        summary=(3, 99, 0, 99),
    )
    parsed = parse_hybrid_response(resp)
    assert parsed["proc_present_pts"] == 2  # from per-item
    assert parsed["concl_present_pts"] == 0


def test_parse_hybrid_response_dedupe_duplicate_items() -> None:
    resp = "item 1 (1 point): PRESENT\nitem 1 (1 point): ABSENT\nitem 5 (5 points): PRESENT\n"
    parsed = parse_hybrid_response(resp)
    verdicts = {idx: v for idx, _, v, _ in parsed["per_item"]}
    assert verdicts[1] == "PRESENT"
    assert verdicts[5] == "PRESENT"


def test_parse_hybrid_response_accepts_pt_abbreviation() -> None:
    resp = "item 1 (1 pt): PRESENT\nitem 2 (2 pts): ABSENT\nitem 3 (5 points): PRESENT\n"
    parsed = parse_hybrid_response(resp)
    weights = {idx: w for idx, w, _, _ in parsed["per_item"]}
    assert weights == {1: 1, 2: 2, 3: 5}


def test_parse_hybrid_response_accepts_markdown_bold_on_verdict() -> None:
    resp = "item 1 (1 point): **PRESENT** — loaded CSV\nitem 2 (5 points): **ABSENT** — no test run"
    parsed = parse_hybrid_response(resp)
    verdicts = {idx: v for idx, _, v, _ in parsed["per_item"]}
    assert verdicts == {1: "PRESENT", 2: "ABSENT"}


def test_parse_hybrid_response_empty_returns_no_items() -> None:
    parsed = parse_hybrid_response("garbage output that doesn't match anything")
    assert parsed["per_item"] == []
    assert parsed["proc_max_pts"] == 0
    assert parsed["concl_max_pts"] == 0


def test_parse_hybrid_response_extracts_reason_text() -> None:
    resp = "item 1 (1 point): PRESENT — loaded the CSV via pd.read_csv"
    parsed = parse_hybrid_response(resp)
    assert parsed["per_item"][0][3].startswith("loaded the CSV")


# ---------- synthesize_per_item_awards ----------

@pytest.mark.parametrize(
    ("total", "expected_awards"),
    [
        (0, [False, False, False, False, False]),
        (1, [True, False, False, False, False]),
        (3, [True, True, True, False, False]),
        (4, [True, True, True, True, False]),
        (5, [False, False, False, False, True]),  # total == conclusion weight → conclusion only
        (6, [True, False, False, False, True]),
        (9, [True, True, True, True, True]),
    ],
)
def test_synthesize_per_item_awards(total: int, expected_awards: list[bool]) -> None:
    weights = [1, 1, 1, 1, 5]
    awards = synthesize_per_item_awards(total, weights)
    assert [a for _, _, a in awards] == expected_awards


def test_synthesize_per_item_awards_empty() -> None:
    assert synthesize_per_item_awards(3, []) == []


def test_synthesize_per_item_awards_mixed_procedural_weights() -> None:
    # Rubric with 2-point procedural items (real pattern in the dataset).
    weights = [1, 2, 2, 5]  # total = 10
    # total=8 -> concl (5) + one 2pt procedural + one 1pt procedural (greedy by index)
    awards = synthesize_per_item_awards(8, weights)
    # Procedural allocation is greedy lowest-index: 1 (1pt), 2 (2pt), then no more leftover (8-5-1-2=0)
    assert [a for _, _, a in awards] == [True, True, False, True]


# ---------- hybrid_reward ----------

def test_hybrid_reward_perfect_rollout() -> None:
    weights = [1, 1, 1, 1, 5]
    awards = synthesize_per_item_awards(9, weights)
    faith = parse_hybrid_response(
        _response(
            [(1, 1, "PRESENT"), (2, 1, "PRESENT"), (3, 1, "PRESENT"), (4, 1, "PRESENT"), (5, 5, "PRESENT")],
            summary=(4, 4, 5, 5),
        ),
    )
    score, breakdown = hybrid_reward(awards, faith, max_points=9)
    assert score == pytest.approx(1.0)
    assert breakdown["proc_pts_credited"] == 4
    assert breakdown["concl_pts_credited"] == 5
    assert breakdown["strip_reason"] == "none"


def test_hybrid_reward_conclusion_stripped_by_faith() -> None:
    weights = [1, 1, 1, 1, 5]
    awards = synthesize_per_item_awards(9, weights)
    faith = parse_hybrid_response(
        _response(
            [(1, 1, "PRESENT"), (2, 1, "PRESENT"), (3, 1, "PRESENT"), (4, 1, "PRESENT"), (5, 5, "ABSENT")],
            summary=(4, 4, 0, 5),
        ),
    )
    score, breakdown = hybrid_reward(awards, faith, max_points=9)
    assert score == pytest.approx(4 / 9)
    assert breakdown["concl_pts_credited"] == 0
    assert breakdown["strip_reason"] == "faith_absent"


def test_hybrid_reward_procedural_stripped_by_faith() -> None:
    weights = [1, 1, 1, 1, 5]
    awards = synthesize_per_item_awards(9, weights)
    faith = parse_hybrid_response(
        _response(
            [(1, 1, "ABSENT"), (2, 1, "PRESENT"), (3, 1, "PRESENT"), (4, 1, "PRESENT"), (5, 5, "PRESENT")],
            summary=(3, 4, 5, 5),
        ),
    )
    score, breakdown = hybrid_reward(awards, faith, max_points=9)
    assert score == pytest.approx(8 / 9)
    assert breakdown["proc_pts_credited"] == 3
    assert breakdown["concl_pts_credited"] == 5


def test_hybrid_reward_rubric_not_awarded_never_credited() -> None:
    weights = [1, 1, 1, 1, 5]
    awards = synthesize_per_item_awards(5, weights)  # only conclusion awarded
    faith = parse_hybrid_response(
        _response(
            [(1, 1, "PRESENT"), (2, 1, "PRESENT"), (3, 1, "PRESENT"), (4, 1, "PRESENT"), (5, 5, "PRESENT")],
            summary=(4, 4, 5, 5),
        ),
    )
    score, breakdown = hybrid_reward(awards, faith, max_points=9)
    assert score == pytest.approx(5 / 9)
    assert breakdown["proc_pts_credited"] == 0
    assert breakdown["concl_pts_credited"] == 5


def test_hybrid_reward_missing_faith_entry_fails_open() -> None:
    weights = [1, 1, 1, 1, 5]
    awards = synthesize_per_item_awards(9, weights)
    # Judge dropped item 2 entirely.
    faith = parse_hybrid_response(
        _response(
            [(1, 1, "PRESENT"), (3, 1, "PRESENT"), (4, 1, "PRESENT"), (5, 5, "PRESENT")],
            summary=(3, 4, 5, 5),
        ),
    )
    score, breakdown = hybrid_reward(awards, faith, max_points=9)
    assert score == pytest.approx(1.0)
    assert breakdown["proc_pts_credited"] == 4


def test_hybrid_reward_zero_max_points() -> None:
    score, _ = hybrid_reward([], {"per_item": []}, max_points=0)
    assert score == 0.0
