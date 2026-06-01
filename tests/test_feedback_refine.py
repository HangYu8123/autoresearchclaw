"""Tests for the researchclaw refine feedback feature."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.feedback.dispatcher import dispatch_feedback
from researchclaw.feedback.loader import _classify_heuristic, parse_feedback
from researchclaw.feedback.refiner import RefineReport, run_refine
from researchclaw.feedback.validator import validate_refine
from researchclaw.pipeline.stages import Stage


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def test_parse_feedback_with_tags(tmp_path: Path) -> None:
    fb = tmp_path / "fb.md"
    fb.write_text(
        "- [paper] Tighten the abstract.\n"
        "- [code] Fix the off-by-one in loop.\n"
        "- [experiment] Add a baseline.\n"
        "- [consistency] Algorithm pseudocode mismatches text.\n",
        encoding="utf-8",
    )
    doc = parse_feedback(fb)
    cats = [it.category for it in doc.items]
    assert cats == ["paper", "code", "experiment", "consistency"]
    assert doc.items[0].text == "Tighten the abstract."
    assert doc.items[0].id == 1
    assert doc.items[0].source_line == 1


def test_parse_feedback_plain_text_default_paper(tmp_path: Path) -> None:
    fb = tmp_path / "fb.txt"
    fb.write_text("Just a paragraph with no bullets at all.\n", encoding="utf-8")
    doc = parse_feedback(fb)
    assert len(doc.items) == 1
    assert doc.items[0].category == "paper"
    assert "paragraph" in doc.items[0].text


def test_parse_feedback_numbered_list(tmp_path: Path) -> None:
    fb = tmp_path / "fb.md"
    fb.write_text("1. First item about figure.\n2. Second item about baseline.\n", encoding="utf-8")
    doc = parse_feedback(fb)
    assert len(doc.items) == 2
    assert doc.items[0].text.startswith("First item")
    assert doc.items[1].text.startswith("Second item")


def test_classify_heuristic_paper_keywords() -> None:
    assert _classify_heuristic("Improve the abstract wording") == "paper"
    assert _classify_heuristic("Figure 2 caption needs update") == "paper"


def test_classify_heuristic_experiment_keywords() -> None:
    assert _classify_heuristic("Add a stronger baseline") == "experiment"
    assert _classify_heuristic("Try another seed") == "experiment"


def test_classify_heuristic_consistency_keywords() -> None:
    assert _classify_heuristic("Algorithm pseudocode mismatches the text") == "consistency"
    assert _classify_heuristic("paper says X but code does Y") == "consistency"


def test_classify_heuristic_code_keywords() -> None:
    assert _classify_heuristic("There is a bug in the loop") == "code"
    assert _classify_heuristic("Traceback in main") == "code"


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "rc-run"
    rd.mkdir()
    (rd / "checkpoint.json").write_text("{}", encoding="utf-8")
    return rd


def test_dispatcher_writes_iteration_context_for_paper(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] Tighten the intro.\n", encoding="utf-8")
    doc = parse_feedback(fb)
    plan = dispatch_feedback(rd, doc, refine_id="refine-x")
    ctx = json.loads((rd / "iteration_context.json").read_text(encoding="utf-8"))
    assert ctx["iteration"] >= 2
    assert "Human Feedback" in ctx["reviews_excerpt"]
    assert ctx["source"] == "user_feedback_refine"
    assert plan.min_from_stage == Stage.PAPER_OUTLINE


def test_dispatcher_backs_up_and_appends_reviews_for_consistency(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    stage18 = rd / "stage-18"
    stage18.mkdir()
    original = "# existing reviews\n"
    (stage18 / "reviews.md").write_text(original, encoding="utf-8")
    fb = tmp_path / "fb.md"
    fb.write_text("- [consistency] Pseudocode mismatches text.\n", encoding="utf-8")
    doc = parse_feedback(fb)
    plan = dispatch_feedback(rd, doc, refine_id="refine-y")
    assert (stage18 / "reviews.md.pre-refine").read_text(encoding="utf-8") == original
    appended = (stage18 / "reviews.md").read_text(encoding="utf-8")
    assert "Human Feedback (Consistency Review)" in appended
    assert "Pseudocode mismatches text" in appended
    assert plan.min_from_stage == Stage.PAPER_REVISION


def test_dispatcher_writes_user_directives_for_experiment(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [experiment] Add stronger baseline.\n", encoding="utf-8")
    doc = parse_feedback(fb)
    dispatch_feedback(rd, doc, refine_id="refine-z")
    directives = (rd / "feedback" / "user_directives.md").read_text(encoding="utf-8")
    assert "Add stronger baseline" in directives
    assert "# User Refinement Directives" in directives


def test_dispatcher_min_from_stage_paper_only_is_paper_outline(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] X\n", encoding="utf-8")
    plan = dispatch_feedback(rd, parse_feedback(fb), refine_id="r1")
    assert plan.min_from_stage == Stage.PAPER_OUTLINE
    assert int(plan.min_from_stage) == 16


def test_dispatcher_min_from_stage_with_experiment_is_iterative_refine(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] X\n- [experiment] Y\n", encoding="utf-8")
    plan = dispatch_feedback(rd, parse_feedback(fb), refine_id="r2")
    assert plan.min_from_stage == Stage.ITERATIVE_REFINE
    assert int(plan.min_from_stage) == 13


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

def test_validator_reads_quality_and_verification_reports(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    (rd / "stage-20").mkdir()
    (rd / "stage-20" / "quality_report.json").write_text(
        json.dumps({"quality_score": 0.85}), encoding="utf-8"
    )
    (rd / "stage-23").mkdir()
    (rd / "stage-23" / "verification_report.json").write_text(
        json.dumps({"severity": "PASS", "fabrication_rate": 0.01}), encoding="utf-8"
    )
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] X\n", encoding="utf-8")
    doc = parse_feedback(fb)
    plan = dispatch_feedback(rd, doc, refine_id="r-val")
    report = validate_refine(
        run_dir=rd, doc=doc, refine_id="r-val",
        quality_before=0.5, plan=plan, config=MagicMock(),
    )
    assert report.quality_after == 0.85
    assert report.verifier_severity == "PASS"


def test_validator_writes_report(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] X\n", encoding="utf-8")
    doc = parse_feedback(fb)
    plan = dispatch_feedback(rd, doc, refine_id="r-w")
    report = validate_refine(
        run_dir=rd, doc=doc, refine_id="r-w",
        quality_before=None, plan=plan, config=MagicMock(),
    )
    assert report.report_path.exists()
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["refine_id"] == "r-w"
    assert payload["items_total"] == 1
    assert "items" in payload and len(payload["items"]) == 1
    assert (rd / "feedback" / "r-w" / "refinement_report.md").exists()


# ---------------------------------------------------------------------------
# refiner
# ---------------------------------------------------------------------------

def test_run_refine_dry_run_returns_plan_without_execute_pipeline(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] Tighten abstract.\n", encoding="utf-8")

    def _boom(**_kwargs: object) -> object:
        raise AssertionError("execute_pipeline must not be called in dry-run")

    with patch("researchclaw.pipeline.runner.execute_pipeline", side_effect=_boom):
        report = run_refine(
            run_dir=rd, feedback_path=fb, config=MagicMock(),
            auto_approve=True, dry_run=True,
        )
    assert isinstance(report, RefineReport)
    assert report.from_stage == Stage.PAPER_OUTLINE
    assert report.to_stage == Stage.CITATION_VERIFY
    assert report.items_total == 1


def test_run_refine_raises_for_invalid_run_dir(tmp_path: Path) -> None:
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] X\n", encoding="utf-8")
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError):
        run_refine(
            run_dir=missing, feedback_path=fb, config=MagicMock(),
            auto_approve=False, dry_run=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_refine_dispatch_invokes_run_refine(tmp_path: Path) -> None:
    from researchclaw import cli

    rd = _make_run_dir(tmp_path)
    fb = tmp_path / "fb.md"
    fb.write_text("- [paper] X\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("dummy: true\n", encoding="utf-8")

    fake_report = RefineReport(
        refine_id="rfake",
        from_stage=Stage.PAPER_OUTLINE,
        to_stage=Stage.CITATION_VERIFY,
        quality_before=0.5,
        quality_after=0.7,
        verifier_severity="PASS",
        items_total=1,
        items_addressed=1,
        overall_pass=True,
        report_path=tmp_path / "report.json",
    )

    args = MagicMock()
    args.run_dir = str(rd)
    args.feedback = str(fb)
    args.config = str(cfg_path)
    args.auto_approve = True
    args.dry_run = False

    with patch.object(cli, "_resolve_config_or_exit", return_value=cfg_path), \
         patch.object(cli.RCConfig, "load", return_value=MagicMock()), \
         patch("researchclaw.feedback.refiner.run_refine", return_value=fake_report) as mock_run:
        rc = cli.cmd_refine(args)
    assert rc == 0
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["run_dir"] == rd
    assert kwargs["feedback_path"] == fb
    assert kwargs["auto_approve"] is True
    assert kwargs["dry_run"] is False
