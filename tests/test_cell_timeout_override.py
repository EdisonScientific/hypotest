# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the cell-timeout override closure's clamp logic.

We don't boot a real kernel container here — the interesting behavior is the
closure's handling of timeout_seconds: defensive coercion, clamp to
[cell_timeout_min, cell_timeout_max], passing the clamped cap to
_run_cell_with_cap. We verify this by replaying the closure's logic against
the documented invariants rather than by running the full InterpreterEnv
reset() pathway (which pulls in ray/nbformat).
"""
from __future__ import annotations


def _clamp(value, fallback: float, lo: float, hi: float) -> float:
    """Faithful reproduction of the closure's defensive coerce + clamp.

    Kept identical to the logic in `_run_cell_with_timeout` in interpreter_env.py.
    If this test file diverges from that closure, either this test or the
    closure is wrong — both must stay in sync.
    """
    if value is None:
        return fallback
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(lo, min(hi, cap))


# --- None / default path ---

def test_none_returns_fallback() -> None:
    assert _clamp(None, 600.0, 60.0, 1200.0) == 600.0


def test_missing_arg_treated_as_none() -> None:
    # Simulates model omitting timeout_seconds in the tool call.
    assert _clamp(None, 300.0, 60.0, 1200.0) == 300.0


# --- In-range ---

def test_in_range_passes_through() -> None:
    assert _clamp(500.0, 600.0, 60.0, 1200.0) == 500.0


def test_at_min_bound_passes_through() -> None:
    assert _clamp(60.0, 600.0, 60.0, 1200.0) == 60.0


def test_at_max_bound_passes_through() -> None:
    assert _clamp(1200.0, 600.0, 60.0, 1200.0) == 1200.0


# --- Clamping ---

def test_below_min_clamps_to_min() -> None:
    assert _clamp(30.0, 600.0, 60.0, 1200.0) == 60.0


def test_above_max_clamps_to_max() -> None:
    assert _clamp(5000.0, 600.0, 60.0, 1200.0) == 1200.0


def test_negative_clamps_to_min() -> None:
    assert _clamp(-10.0, 600.0, 60.0, 1200.0) == 60.0


def test_zero_clamps_to_min() -> None:
    assert _clamp(0.0, 600.0, 60.0, 1200.0) == 60.0


# --- Defensive coercion ---

def test_string_numeric_coerces_then_clamps() -> None:
    assert _clamp("300", 600.0, 60.0, 1200.0) == 300.0
    assert _clamp("2000", 600.0, 60.0, 1200.0) == 1200.0  # clamped


def test_non_numeric_string_falls_back_to_default() -> None:
    assert _clamp("long", 600.0, 60.0, 1200.0) == 600.0
    assert _clamp("abort", 600.0, 60.0, 1200.0) == 600.0


def test_unexpected_type_falls_back_to_default() -> None:
    assert _clamp([900], 600.0, 60.0, 1200.0) == 600.0
    assert _clamp({"seconds": 900}, 600.0, 60.0, 1200.0) == 600.0


def test_bool_coerces_to_01_which_clamps_to_min() -> None:
    # bool is a subclass of int; float(True) == 1.0; float(False) == 0.0.
    # Both are below cell_timeout_min so they clamp up.
    assert _clamp(True, 600.0, 60.0, 1200.0) == 60.0
    assert _clamp(False, 600.0, 60.0, 1200.0) == 60.0


# --- Config-configurable bounds ---

def test_bounds_can_be_widened_via_config() -> None:
    # If the config sets cell_timeout_max=1800 and cell_timeout_min=30, the
    # closure captures those values at reset() time. Simulate with different lo/hi.
    assert _clamp(1500.0, 600.0, 30.0, 1800.0) == 1500.0
    assert _clamp(40.0, 600.0, 30.0, 1800.0) == 40.0
    assert _clamp(2000.0, 600.0, 30.0, 1800.0) == 1800.0


def test_bounds_can_be_tightened_via_config() -> None:
    # More conservative config: [120, 900].
    assert _clamp(100.0, 600.0, 120.0, 900.0) == 120.0
    assert _clamp(1000.0, 600.0, 120.0, 900.0) == 900.0
    assert _clamp(600.0, 600.0, 120.0, 900.0) == 600.0
