"""Tests for sanitized OpenSandbox kernel-wedge diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "probe_opensandbox_kernel_wedge.py"
SPEC = importlib.util.spec_from_file_location("hypotest_probe_opensandbox_kernel_wedge", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_structured_failure_evidence_retains_only_allowlisted_facts():
    raw = (
        "pod private-workload-name container kernel terminated with reason OOMKilled, "
        "exited with code 137; https://private.example/path"
    )

    evidence = probe._structured_failure_evidence(raw)

    assert evidence["canonical_termination_tokens"] == ["OOMKilled"]
    assert evidence["exit_codes"] == [137]
    assert evidence["redacted_summary"] == "termination=OOMKilled; exit_code=137"
    assert evidence["source_message_length"] == len(raw)
    assert len(evidence["source_message_sha256"]) == 64
    assert "private" not in str(evidence)


def test_structured_failure_evidence_handles_empty_message():
    assert probe._structured_failure_evidence(None) == {
        "canonical_termination_tokens": [],
        "exit_codes": [],
        "redacted_summary": None,
        "source_message_length": 0,
        "source_message_sha256": None,
    }
