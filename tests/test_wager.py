# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for wager (Scheme D) pure functions.

Ports the 10 tests from features/wager_scheme/reference_impl.py plus four new:
- clamp_confidence defensive behavior (strings, None)
- breakdown-dict completeness
- fixtures.csv row-by-row parity
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hypotest.env.wager import (
    WAGER_BETA_DEFAULT,
    WAGER_GAMMA_DEFAULT,
    clamp_confidence,
    ev_crossover,
    score_with_wager,
)


def _r(proc_credit, proc_max, concl_credit, concl_max, correct, wager, beta=WAGER_BETA_DEFAULT, gamma=WAGER_GAMMA_DEFAULT):
    r, _ = score_with_wager(proc_credit, proc_max, concl_credit, concl_max, correct, wager, beta, gamma)
    return r


# ---------- ported from reference_impl.py ----------

def test_baseline_honest_correct() -> None:
    # correct, w=0: baseline reward, no bonus
    assert _r(4, 4, 5, 5, True, 0.0) == pytest.approx(9 / 9)


def test_baseline_honest_wrong() -> None:
    # wrong, w=0, gate stripped concl → proc only
    assert _r(4, 4, 0, 5, False, 0.0) == pytest.approx(4 / 9)


def test_full_confidence_correct() -> None:
    # correct, w=1: 4 + 5 + 1 * 0.5 * 5 = 11.5
    assert _r(4, 4, 5, 5, True, 1.0) == pytest.approx(11.5 / 9)


def test_full_confidence_wrong() -> None:
    # wrong, w=1: 4 * (1 - 0.3) + 0 = 2.8
    assert _r(4, 4, 0, 5, False, 1.0) == pytest.approx(2.8 / 9)


def test_half_wager_correct() -> None:
    # correct, w=0.5: 4 + 5 + 0.5 * 0.5 * 5 = 10.25
    assert _r(4, 4, 5, 5, True, 0.5) == pytest.approx(10.25 / 9)


def test_wager_clamped() -> None:
    # out of range clamps to [0, 1]
    assert _r(4, 4, 5, 5, True, 2.0) == pytest.approx(_r(4, 4, 5, 5, True, 1.0))
    assert _r(4, 4, 0, 5, False, -0.5) == pytest.approx(_r(4, 4, 0, 5, False, 0.0))


def test_ev_invariant_hard_task() -> None:
    # On a task with P(correct) below crossover, honest should beat confident in EV.
    p_crit = ev_crossover(4, 5, WAGER_BETA_DEFAULT, WAGER_GAMMA_DEFAULT)
    p = p_crit * 0.5  # deep into "hard" regime

    ev_confident = p * _r(4, 4, 5, 5, True, 1.0) + (1 - p) * _r(4, 4, 0, 5, False, 1.0)
    ev_honest = p * _r(4, 4, 5, 5, True, 0.0) + (1 - p) * _r(4, 4, 0, 5, False, 0.0)
    assert ev_honest > ev_confident


def test_ev_invariant_easy_task() -> None:
    # Above crossover, confident should beat honest.
    p_crit = ev_crossover(4, 5, WAGER_BETA_DEFAULT, WAGER_GAMMA_DEFAULT)
    p = p_crit + (1 - p_crit) * 0.5

    ev_confident = p * _r(4, 4, 5, 5, True, 1.0) + (1 - p) * _r(4, 4, 0, 5, False, 1.0)
    ev_honest = p * _r(4, 4, 5, 5, True, 0.0) + (1 - p) * _r(4, 4, 0, 5, False, 0.0)
    assert ev_confident > ev_honest


def test_crossover_value() -> None:
    # 1.2 / (1.2 + 2.5) = 1.2 / 3.7
    assert ev_crossover(4, 5, WAGER_BETA_DEFAULT, WAGER_GAMMA_DEFAULT) == pytest.approx(1.2 / 3.7)


def test_gamma_raises_crossover() -> None:
    # Higher GAMMA raises p_crit (honest wins over a wider P range).
    p_low = ev_crossover(4, 5, WAGER_BETA_DEFAULT, 0.3)
    p_mid = ev_crossover(4, 5, WAGER_BETA_DEFAULT, 0.5)
    p_hi = ev_crossover(4, 5, WAGER_BETA_DEFAULT, 0.7)
    assert p_low < p_mid < p_hi


# ---------- new tests ----------

@pytest.mark.parametrize("bad", ["high", "", None, object(), [1, 2], float("nan")])
def test_clamp_confidence_bad_inputs_return_zero(bad) -> None:
    assert clamp_confidence(bad) == 0.0


@pytest.mark.parametrize(
    ("inp", "out"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.0), (2, 1.0), ("0.3", 0.3)],
)
def test_clamp_confidence_good_inputs(inp, out) -> None:
    assert clamp_confidence(inp) == pytest.approx(out)


def test_breakdown_fields_populated() -> None:
    _, b = score_with_wager(4, 4, 5, 5, correct=True, wager=0.7)
    required = {
        "correct", "wager", "beta", "gamma",
        "proc_credit", "proc_max", "concl_credit", "concl_max",
        "total_pts", "max_pts", "bonus_applied", "penalty_applied",
    }
    assert required <= set(b.keys())
    assert b["bonus_applied"] == pytest.approx(0.7 * 0.5 * 5)
    assert b["penalty_applied"] == 0.0


def test_breakdown_penalty_on_wrong() -> None:
    _, b = score_with_wager(4, 4, 0, 5, correct=False, wager=0.8)
    assert b["penalty_applied"] == pytest.approx(4 * 0.8 * 0.3)
    assert b["bonus_applied"] == 0.0


def test_fixtures_csv_parity() -> None:
    # Every row in the reference fixture must match our port to 1e-6.
    fixtures_path = Path("/lustre/fsw/general_sa/akomaragiri/nemorl-polyphe/features/wager_scheme/validation/fixtures.csv")
    if not fixtures_path.exists():
        pytest.skip(f"fixtures not available at {fixtures_path}")

    with fixtures_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    for i, row in enumerate(rows):
        expected = float(row["reward"])
        got, _ = score_with_wager(
            proc_credit=float(row["proc_credit"]),
            proc_max=float(row["proc_max"]),
            concl_credit=float(row["concl_credit"]),
            concl_max=float(row["concl_max"]),
            correct=row["correct"] == "True",
            wager=float(row["wager"]),
            beta=float(row["beta"]),
            gamma=float(row["gamma"]),
        )
        assert got == pytest.approx(expected, abs=1e-6), f"row {i} mismatch: {row} → got {got}"
