"""Pure contracts for the frozen S2-03T action-authority audit evidence."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "scripts/stage2_isaac/audit_s2_03t_action_authority.py"
SUMMARY = ROOT / "reports/stage2/s2_03t_action_authority_summary.json"


def test_authority_auditor_is_syntactically_valid_and_diagnostic_only() -> None:
    source = AUDITOR.read_text(encoding="utf-8")
    ast.parse(source)
    for token in (
        '"contact_not_attempted": True',
        '"box_push_commanded": False',
        '"training_started": False',
        '"checkpoint_loaded": False',
        '"diagnostic_valid": True',
        '"safe_counterfactual_trace.jsonl"',
        '"finite_safe_action_sweep.json"',
    ):
        assert token in source
    assert "OnPolicyRunner" not in source
    assert "runner.learn" not in source


def test_authority_summary_preserves_valid_negative_diagnosis() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["audit_status"] == "PASS"
    assert summary["audit_status_semantics"] == "DIAGNOSTIC_VALIDITY_ONLY"
    assert summary["authority_status"] == "INSUFFICIENT_FOR_CONTACT"
    assert summary["primary_diagnosis"] == "ARM_RESIDUAL_AUTHORITY_INSUFFICIENT"
    assert all(value > 0.0 for value in summary["minimum_actual_counterfactual_gap_m"])
    assert summary["safe_replay_metrics"]["finite"] is True
    assert summary["safe_replay_metrics"]["forbidden_collision"] is False
    assert summary["disposition"]["old_no_qualified_contact_policy_preserved"] is True
    assert summary["disposition"]["falcon_migration_decision"] == "NO"


def test_authority_summary_commits_only_hashes_not_raw_artifacts() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert set(summary["source_artifacts"]) == {
        "result.json",
        "reference_audit.json",
        "jacobian_audit.json",
        "finite_safe_action_sweep.json",
        "safe_counterfactual_trace.jsonl",
    }
    for record in summary["source_artifacts"].values():
        assert record["size_bytes"] > 0
        assert len(record["sha256"]) == 64
