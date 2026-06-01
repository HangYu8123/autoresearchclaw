"""Top-level refine entry point: parse feedback, dispatch, re-run pipeline, validate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from researchclaw.feedback.dispatcher import DispatchPlan, dispatch_feedback
from researchclaw.feedback.loader import FeedbackDocument, parse_feedback
from researchclaw.feedback.validator import validate_refine
from researchclaw.pipeline.stages import Stage

if TYPE_CHECKING:
    from researchclaw.config import RCConfig

logger = logging.getLogger(__name__)


@dataclass
class RefineReport:
    refine_id: str
    from_stage: Stage
    to_stage: Stage
    quality_before: float | None
    quality_after: float | None
    verifier_severity: str
    items_total: int
    items_addressed: int
    overall_pass: bool
    report_path: Path


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _read_quality_score(run_dir: Path) -> float | None:
    import json
    qpath = run_dir / "stage-20" / "quality_report.json"
    if not qpath.exists():
        return None
    try:
        data = json.loads(qpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        val = data.get("quality_score")
        if isinstance(val, (int, float)):
            return float(val)
    if isinstance(data, (int, float)):
        return float(data)
    return None


def run_refine(
    *,
    run_dir: Path,
    feedback_path: Path,
    config: "RCConfig",
    auto_approve: bool,
    dry_run: bool = False,
) -> RefineReport | None:
    if not run_dir.is_dir():
        raise ValueError(f"run_dir is not a directory: {run_dir}")
    has_checkpoint = (run_dir / "checkpoint.json").exists()
    has_stage_dir = any(run_dir.glob("stage-*"))
    if not (has_checkpoint or has_stage_dir):
        raise ValueError(
            f"run_dir does not look like a completed run (no checkpoint.json or stage-* dirs): {run_dir}"
        )

    doc: FeedbackDocument = parse_feedback(feedback_path)
    refine_id = f"refine-{_utc_compact()}"
    plan: DispatchPlan = dispatch_feedback(run_dir, doc, refine_id=refine_id)
    quality_before = _read_quality_score(run_dir)

    if dry_run:
        print(f"[refine] dry-run plan (refine_id={refine_id})")
        print(f"  categories: {sorted(plan.categories_present)}")
        print(f"  min_from_stage: {plan.min_from_stage.name} ({int(plan.min_from_stage)})")
        print(f"  items: {len(doc.items)}")
        for it in doc.items:
            text_preview = it.text[:80].replace("\n", " ")
            print(f"    [{it.category}] #{it.id}: {text_preview}")
        print(f"  paths_written: {[str(p) for p in plan.paths_written]}")
        return RefineReport(
            refine_id=refine_id,
            from_stage=plan.min_from_stage,
            to_stage=Stage.CITATION_VERIFY,
            quality_before=quality_before,
            quality_after=quality_before,
            verifier_severity="UNKNOWN",
            items_total=len(doc.items),
            items_addressed=0,
            overall_pass=True,
            report_path=plan.dispatch_json_path or (run_dir / "feedback" / refine_id / "dispatch.json"),
        )

    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.runner import execute_pipeline

    adapters = AdapterBundle()
    try:
        execute_pipeline(
            run_dir=run_dir,
            run_id=f"{run_dir.name}-{refine_id}",
            config=config,
            adapters=adapters,
            from_stage=plan.min_from_stage,
            to_stage=Stage.CITATION_VERIFY,
            auto_approve_gates=auto_approve,
            stop_on_gate=not auto_approve,
            skip_noncritical=False,
            kb_root=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("refine: pipeline execution raised; validating partial outputs: %s", exc)

    report = validate_refine(
        run_dir=run_dir,
        doc=doc,
        refine_id=refine_id,
        quality_before=quality_before,
        plan=plan,
        config=config,
    )
    return report
