# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scheme D: calibrated-confidence wager scoring.

Composes with the hybrid faithfulness gate — the gate must produce `correct`
(bool) for the conclusion item before score_with_wager is called. `concl_credit`
is expected to be zero when the gate stripped or the answer was wrong.

See features/wager_scheme/SPEC.md for the full design and features/wager_scheme/
reference_impl.py for the original reference.

Pure functions only — no I/O, no judge calls.
"""
from __future__ import annotations

from typing import Any


WAGER_BETA_DEFAULT = 0.5
WAGER_GAMMA_DEFAULT = 0.3


def clamp_confidence(value: Any) -> float:
    """Defensively coerce a confidence value to a float in [0, 1].

    Non-float, None, or out-of-range inputs return the safe default 0.0 —
    "missing or malformed wager = fully hedged", per spec.
    """
    try:
        w = float(value)
    except (TypeError, ValueError):
        return 0.0
    if w != w:  # NaN
        return 0.0
    return max(0.0, min(1.0, w))


def score_with_wager(
    proc_credit: float,
    proc_max: float,
    concl_credit: float,
    concl_max: float,
    correct: bool,
    wager: float,
    beta: float = WAGER_BETA_DEFAULT,
    gamma: float = WAGER_GAMMA_DEFAULT,
) -> tuple[float, dict[str, Any]]:
    """Return (reward, breakdown) under Scheme D.

    Correct:
        total = proc + concl + wager * beta * concl_max           (upside bonus)
    Wrong:
        total = proc * (1 - wager * gamma) + concl                (procedural cut)

    Reward = total / (proc_max + concl_max). May exceed 1.0 when correct with
    high wager — callers are expected to relax any downstream clamp.
    """
    w = clamp_confidence(wager)
    max_pts = proc_max + concl_max

    if correct:
        bonus_pts = w * beta * concl_max
        penalty_pts = 0.0
        total_pts = proc_credit + concl_credit + bonus_pts
    else:
        bonus_pts = 0.0
        penalty_pts = proc_credit * w * gamma
        total_pts = proc_credit * (1.0 - w * gamma) + concl_credit

    reward = total_pts / max_pts if max_pts > 0 else 0.0

    breakdown: dict[str, Any] = {
        "correct": bool(correct),
        "wager": w,
        "beta": float(beta),
        "gamma": float(gamma),
        "proc_credit": float(proc_credit),
        "proc_max": float(proc_max),
        "concl_credit": float(concl_credit),
        "concl_max": float(concl_max),
        "total_pts": float(total_pts),
        "max_pts": float(max_pts),
        "bonus_applied": float(bonus_pts),
        "penalty_applied": float(penalty_pts),
    }
    return reward, breakdown


def ev_crossover(proc_max: float, concl_max: float, beta: float, gamma: float) -> float:
    """Return the P(correct) above which full-confidence beats zero-confidence in EV.

    p_crit = (proc_max * gamma) / (proc_max * gamma + beta * concl_max)
    """
    num = proc_max * gamma
    den = num + beta * concl_max
    if den == 0:
        return 0.0
    return num / den
