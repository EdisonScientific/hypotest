# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hybrid faithfulness+rubric gate: per-item PRESENT/ABSENT verdicts combined
with rubric awards. See features/hybrid_faithfulness_rubric/hybrid_gate_spec.md.

Design note: the judge reads the raw rubric text and is responsible for
numbering items and echoing each item's weight inline. We do NOT pre-parse
rubric items on the client — rubric formats in the dataset are too varied for
regex to handle reliably, and the judge is strictly better at reading natural
language.

Pure functions only — no I/O, no judge calls. The judge-call wiring lives in
interpreter_env.py._evaluate_hybrid_gate.
"""

from __future__ import annotations

import re
from typing import Any


HYBRID_GATE_PROMPT = """You are auditing whether a Jupyter notebook provides sufficient executed evidence for each item in a grading rubric.

Hypothesis under evaluation: {hypothesis!r}.

Notebook (code cells + outputs):
<notebook>
{notebook}
</notebook>

Proposed conclusion submitted by the agent:
<proposed-solution>
{proposed_solution}
</proposed-solution>

Rubric:
<rubric>
{rubric}
</rubric>

Read the rubric and identify each scorable item along with its point weight. Number the items 1, 2, 3, ... in the order they appear in the rubric. The per-item weights you echo MUST sum to the rubric's stated total — do not inflate or deflate the total by splitting or merging items. If a given item is not fully resolved (e.g. one part of the item is satisfied but not the other), it does not get credit and should be marked ABSENT.

**Handling multi-part items with split credit:** When a rubric item is worth N points and its description contains language like `"(Separate credit for A and B)"`, `"(scored separately)"`, `"(each worth a point)"`, `"(accounts for the two points)"`, `"(separately evaluable)"`, or similar, the N points are DIVIDED across the listed sub-parts (e.g. N/2 to each of two parts) — NOT multiplied. Do not emit each sub-part as its own separate N-point item. You should keep the single N-point item as one entry in your output (weight = N, exactly as stated in the rubric) and judge it holistically: PRESENT only if ALL listed sub-parts are supported; ABSENT otherwise. If you are confident both sub-parts are supported, mark PRESENT; if only one is, mark ABSENT (partial credit is not available at this step — the rubric grader has already decided that).

**Expected-value anchor rule (applies to ALL items):**

Rubric items sometimes cite specific expected numerical values (e.g., "expected r ≈ 0.56", "~28 DEGs", "p ≈ 0.030"). Treat these as rubric-author ANCHORS — canonical values a correct analysis would produce — not as hard targets the policy must hit exactly. Policies executing a valid analysis may arrive at methodology-variant numbers (different normalization, different FDR method, different library choice) while still correctly implementing the rubric item. The discriminator between methodology variation and fabrication is **whether the policy's answer value matches its own notebook output**, not whether it matches the rubric's anchor value.

- PRESENT requires ALL of: (a) the specific operation named by the rubric item (including any specified thresholds, filters, or parameters) was actually executed in the notebook; (b) every numeric claim in the policy's answer matches values that appear in its own notebook outputs within reasonable rounding; (c) any directional or threshold claim is supported by the policy's own computed numbers — regardless of whether those numbers match the rubric's anchor exactly.
- ABSENT if ANY of: (a) the policy cites a value that is NOT in its own notebook output (fabrication — e.g., answer claims "R² ≈ 0.96" when notebook shows R² = 0.649); (b) the policy's answer contradicts its own notebook output in direction or magnitude; (c) the required computation was not executed, only asserted in answer text; (d) a directional or threshold claim is inconsistent with the policy's own computed numbers (e.g., claims "significant" when the policy's own p-value > 0.05).

**Do not mark ABSENT solely because the policy's value differs from the rubric's anchor value.** A policy value of r = 0.73 is acceptable against a rubric anchor of r ≈ 0.56, provided the notebook actually computed 0.73 via a valid Pearson correlation implementation.

For EACH rubric item, decide whether the NOTEBOOK supports awarding that item. Apply tiered strictness based on the item's weight:

**Procedural items (small weight — typically 1 or 2 points each) — STRICT per-item grounding:**
- The claimed operation must be EXECUTED in the notebook: a real library call (e.g. `pd.read_csv`, `scipy.stats.f_oneway`, `statsmodels.anova_lm`, `scipy.stats.mannwhitneyu`) followed by output consistent with the claim.
- A bare `print("F=49.44")` with fabricated values, without the underlying function call, is NOT sufficient — mark ABSENT.
- Numbers cited in the conclusion that are "derivable" from printed values (e.g. a fold-change mentally computed from two printed means) count as ABSENT unless the derivation itself was run in a cell and printed.
- If a required operation could not execute (missing data, tool failure, timeout), dependent items are ABSENT.
- If the rubric item names a specific statistical method, verify the corresponding library function was invoked on the relevant data — don't accept surface narrative.

**Conclusion item (the single highest-weight item — typically 4 or 5 points) — HOLISTIC judgment:**
- PRESENT if: all specific numeric claims in the conclusion (p-values, effect sizes, test statistics, means, medians) appear in notebook outputs within reasonable rounding, AND the directional claim (e.g. "higher/lower", "supported/rejected", "significant/non-significant") matches what the numbers show.
- Qualitative intensifiers — "markedly", "significantly", "strongly", "substantially", "clearly" — are ACCEPTABLE when the underlying numbers clearly support the direction, even if the specific word isn't separately quantified. Do not penalize the conclusion item for intensifier phrasing alone.
- ABSENT if: the conclusion cites specific numbers NOT in the notebook outputs; the conclusion contradicts the notebook's numbers; the conclusion asserts analyses that were never executed (e.g. claims "we performed a likelihood-ratio test" when no LRT appears in the notebook); or the conclusion makes causal/historical/interpretive claims that require analyses absent from the notebook.
- ABSENT if the conclusion's SCOPE, COHORT, or FRAMING depends on an assumption (e.g. a subgroup restriction, a cohort definition, a transformation, a normalization) whose validity was not ESTABLISHED by executed code in the notebook. Establishing the assumption can be done EITHER by applying the operation directly (e.g. `df = df[df.IDH == "WT"]`, executing the transformation) OR by producing a verification check that demonstrates the property already holds (e.g. `df.IDH.value_counts()` showing only IDH_WT, a printed per-subgroup sample count, an explicit assertion, a summary table that confirms the scope). The agent does NOT have to re-apply an operation if an executed check already confirms the data is in the required form — but it must show one or the other. Filenames, variable names, column labels, or data-provenance claims in narrative are NOT acceptable substitutes on their own; they must be backed by an executed check. Conclusions drawn from unverified scoping assumptions are not grounded.

Output format — exactly this and nothing else. Include the item's point weight in parentheses, exactly as stated in the rubric:

item 1 (1 point): PRESENT — <=20 word reason
item 2 (1 point): ABSENT — <=20 word reason
...
item K (5 points): PRESENT — <=20 word reason

<summary>
procedural_points_present=X of Y
conclusion_points_present=A of B
</summary>

Y is the sum of weights of all procedural (non-highest-weight) items. B is the weight of the conclusion (highest-weight) item. X and A are the sums of weights you marked PRESENT in each category."""


# Captures: (index, weight, verdict, reason).
# Accepts "point" / "points" / "pt" / "pts" and optional **bold** around the
# weight clause, plus em-dash / en-dash / hyphen separators before the reason.
ITEM_LINE = re.compile(
    r"item\s+(\d+)\s*\**\s*\(\s*(\d+)\s*(?:points?|pts?)\s*\)\**\s*:\s*"
    r"\**\s*(PRESENT|ABSENT)\s*\**"
    r"\s*(?:[\u2014\u2013\-]\s*(.*))?",
    re.IGNORECASE,
)

SUMMARY_RE = re.compile(
    r"<summary>.*?procedural_points_present\s*=\s*(\d+)\s*of\s*(\d+).*?"
    r"conclusion_points_present\s*=\s*(\d+)\s*of\s*(\d+).*?</summary>",
    re.DOTALL | re.IGNORECASE,
)


def parse_hybrid_response(response: str) -> dict[str, Any]:
    """Parse the judge's response into per-item verdicts + point totals.

    The judge echoes each item's weight inline (e.g. `item 3 (1 point): PRESENT — ...`),
    so this function does NOT need the rubric on the client side — weights come from
    the judge's own output. The `<summary>` block is preferred when its totals are
    internally consistent with the per-item verdicts.

    Returns a dict with:
        per_item: list[(idx, weight, verdict, reason)]
        proc_present_pts, proc_max_pts, concl_present_pts, concl_max_pts
    """
    per_item: list[tuple[int, int, str, str]] = []
    seen: set[int] = set()
    for m in ITEM_LINE.finditer(response):
        idx = int(m.group(1))
        if idx in seen or idx < 1:
            continue
        seen.add(idx)
        weight = int(m.group(2))
        verdict = m.group(3).upper()
        reason = (m.group(4) or "").strip()
        per_item.append((idx, weight, verdict, reason))

    conclusion_weight = max((w for _, w, _, _ in per_item), default=0)
    proc_max = sum(w for _, w, _, _ in per_item if w != conclusion_weight)
    concl_max = conclusion_weight

    proc_present = sum(
        w for _, w, v, _ in per_item if v == "PRESENT" and w != conclusion_weight
    )
    concl_present = sum(
        w for _, w, v, _ in per_item if v == "PRESENT" and w == conclusion_weight
    )

    sm = SUMMARY_RE.search(response)
    if sm:
        try:
            sp = int(sm.group(1))
            sy = int(sm.group(2))
            sc = int(sm.group(3))
            sb = int(sm.group(4))
            if sy == proc_max:
                proc_present = sp
            if sb == concl_max:
                concl_present = sc
        except ValueError:
            pass

    return {
        "per_item": per_item,
        "proc_present_pts": proc_present,
        "proc_max_pts": proc_max,
        "concl_present_pts": concl_present,
        "concl_max_pts": concl_max,
    }


def synthesize_per_item_awards(
    rubric_total_pts: int, item_weights: list[int]
) -> list[tuple[int, int, bool]]:
    """Distribute a total rubric score over per-item awards.

    Spec §1 heuristic: when the rubric grader only emits a total, assume the
    conclusion (max-weight) item is awarded iff total >= conclusion_weight,
    and allocate leftover points greedily to procedural items by index.

    `item_weights` is a 1-indexed list of weights (length K). Returns
    [(item_idx_1based, weight, awarded), ...].
    """
    if not item_weights:
        return []

    conclusion_weight = max(item_weights)
    proc_indices = [i for i, w in enumerate(item_weights, 1) if w != conclusion_weight]
    concl_indices = [i for i, w in enumerate(item_weights, 1) if w == conclusion_weight]

    concl_awarded = rubric_total_pts >= conclusion_weight
    leftover = rubric_total_pts - conclusion_weight if concl_awarded else rubric_total_pts
    leftover = max(0, leftover)

    awards: dict[int, bool] = {i: False for i in range(1, len(item_weights) + 1)}
    for idx in proc_indices:
        if leftover <= 0:
            break
        weight = item_weights[idx - 1]
        if leftover >= weight:
            awards[idx] = True
            leftover -= weight
    if concl_awarded and concl_indices:
        awards[concl_indices[0]] = True

    return [(idx, item_weights[idx - 1], awards[idx]) for idx in range(1, len(item_weights) + 1)]


def hybrid_reward(
    rubric_awards: list[tuple[int, int, bool]],
    faith_verdict: dict[str, Any],
    max_points: int,
) -> tuple[float, dict[str, Any]]:
    """Combine rubric awards and faithfulness verdicts into a [0, 1] reward.

    Spec §5: credit = rubric_awarded AND faith_present, per item. Conclusion
    and procedural credits sum, divided by max_points.

    Returns (score, breakdown) where breakdown carries point-level totals and
    a per-item strip_reason for diagnostics.
    """
    conclusion_weight = max((w for _, w, _ in rubric_awards), default=0)
    faith_by_idx: dict[int, tuple[int, str]] = {
        idx: (w, v) for idx, w, v, _ in faith_verdict.get("per_item", [])
    }

    proc_awarded = 0
    proc_credited = 0
    concl_awarded = 0
    concl_credited = 0
    per_item_strip: list[dict[str, Any]] = []

    for idx, weight, rubric_awarded in rubric_awards:
        is_concl = weight == conclusion_weight
        if is_concl:
            if rubric_awarded:
                concl_awarded += weight
        elif rubric_awarded:
            proc_awarded += weight

        if not rubric_awarded:
            per_item_strip.append({"idx": idx, "weight": weight, "strip_reason": "rubric_not_awarded"})
            continue

        faith_entry = faith_by_idx.get(idx)
        if faith_entry is None:
            # Fail-open on missing per-item verdict (judge dropped a line).
            faith_present = True
        else:
            faith_present = faith_entry[1] == "PRESENT"

        if not faith_present:
            per_item_strip.append({"idx": idx, "weight": weight, "strip_reason": "faith_absent"})
            continue

        per_item_strip.append({"idx": idx, "weight": weight, "strip_reason": "none"})
        if is_concl:
            concl_credited += weight
        else:
            proc_credited += weight

    total_credit = proc_credited + concl_credited
    score = total_credit / max_points if max_points > 0 else 0.0
    score = max(0.0, min(1.0, score))

    breakdown = {
        "proc_pts_awarded_by_rubric": proc_awarded,
        "proc_pts_credited": proc_credited,
        "concl_pts_awarded_by_rubric": concl_awarded,
        "concl_pts_credited": concl_credited,
        "proc_max_pts": sum(w for _, w, _ in rubric_awards if w != conclusion_weight),
        "concl_max_pts": conclusion_weight,
        "per_item_strip": per_item_strip,
        "strip_reason": _top_level_strip_reason(per_item_strip),
    }

    return score, breakdown


def _top_level_strip_reason(per_item_strip: list[dict[str, Any]]) -> str:
    """Summarize per-item strip reasons into one label for dashboards."""
    reasons = {entry["strip_reason"] for entry in per_item_strip}
    if reasons == {"none"}:
        return "none"
    if "faith_absent" in reasons and "rubric_not_awarded" in reasons:
        return "mixed"
    if "faith_absent" in reasons:
        return "faith_absent"
    if "rubric_not_awarded" in reasons:
        return "rubric_not_awarded"
    return "none"
