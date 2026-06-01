"""Deterministic validation of a refine round: read quality/verification and per-item checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from researchclaw.feedback.dispatcher import DispatchPlan
from researchclaw.feedback.loader import FeedbackDocument, FeedbackItem
from researchclaw.pipeline.stages import Stage

if TYPE_CHECKING:
    from researchclaw.config import RCConfig
    from researchclaw.feedback.refiner import RefineReport


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_quality_score(run_dir: Path) -> float | None:
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


def _read_verifier_severity(run_dir: Path) -> str:
    vpath = run_dir / "stage-23" / "verification_report.json"
    if not vpath.exists():
        return "UNKNOWN"
    try:
        data = json.loads(vpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "UNKNOWN"
    if isinstance(data, dict):
        sev = data.get("severity")
        if isinstance(sev, str) and sev:
            return sev.upper()
        rate = data.get("fabrication_rate")
        if isinstance(rate, (int, float)):
            if rate > 0.2:
                return "REJECT"
            if rate > 0.05:
                return "WARN"
            return "PASS"
    return "UNKNOWN"


def _mtime_after(path: Path, threshold: float) -> bool:
    try:
        return path.stat().st_mtime > threshold
    except OSError:
        return False


def _any_file_mtime_after(paths: list[Path], threshold: float) -> bool:
    for p in paths:
        if p.is_file() and _mtime_after(p, threshold):
            return True
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and _mtime_after(sub, threshold):
                    return True
    return False


def _check_item(item: FeedbackItem, run_dir: Path, threshold: float) -> bool:
    cat = item.category
    if cat == "paper":
        candidates = [
            run_dir / "stage-17" / "paper_draft.md",
            run_dir / "stage-19" / "paper_revised.md",
        ]
        return any(p.exists() and _mtime_after(p, threshold) for p in candidates)
    if cat == "consistency":
        p = run_dir / "stage-19" / "paper_revised.md"
        return p.exists() and _mtime_after(p, threshold)
    if cat == "experiment":
        log_path = run_dir / "stage-13" / "refinement_log.json"
        if not log_path.exists():
            return False
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        entries: list[object] = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    entries.extend(v)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("timestamp") or entry.get("time") or entry.get("ts")
            if isinstance(ts, (int, float)) and ts > threshold:
                return True
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.timestamp() > threshold:
                        return True
                except ValueError:
                    continue
        return False
    if cat == "code":
        candidates = [
            run_dir / "stage-13" / "experiment_final",
            run_dir / "stage-13" / "experiment_final.py",
        ]
        return _any_file_mtime_after(candidates, threshold)
    return False


def validate_refine(
    *,
    run_dir: Path,
    doc: FeedbackDocument,
    refine_id: str,
    quality_before: float | None,
    plan: DispatchPlan,
    config: "RCConfig",  # noqa: ARG001
) -> "RefineReport":
    from researchclaw.feedback.refiner import RefineReport

    feedback_root = run_dir / "feedback" / refine_id
    feedback_root.mkdir(parents=True, exist_ok=True)

    # Threshold: dispatch.json mtime (or now-fallback).
    threshold = 0.0
    if plan.dispatch_json_path is not None and plan.dispatch_json_path.exists():
        try:
            threshold = plan.dispatch_json_path.stat().st_mtime
        except OSError:
            threshold = 0.0

    quality_after = _read_quality_score(run_dir)
    verifier_severity = _read_verifier_severity(run_dir)

    per_item: list[dict[str, object]] = []
    addressed = 0
    for it in doc.items:
        ok = _check_item(it, run_dir, threshold)
        if ok:
            addressed += 1
        per_item.append(
            {
                "id": it.id,
                "category": it.category,
                "text": it.text,
                "source_line": it.source_line,
                "addressed": ok,
            }
        )

    total = len(doc.items)
    overall_pass = (verifier_severity != "REJECT") and (
        addressed >= max(1, int(total * 0.6))
    )

    # TODO(v2): optional batched LLM per-item check to refine `addressed` verdicts.

    report_json_path = feedback_root / "refinement_report.json"
    report_payload = {
        "refine_id": refine_id,
        "generated": _utc_iso(),
        "from_stage": plan.min_from_stage.name,
        "from_stage_num": int(plan.min_from_stage),
        "to_stage": Stage.CITATION_VERIFY.name,
        "to_stage_num": int(Stage.CITATION_VERIFY),
        "quality_before": quality_before,
        "quality_after": quality_after,
        "verifier_severity": verifier_severity,
        "items_total": total,
        "items_addressed": addressed,
        "overall_pass": overall_pass,
        "items": per_item,
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# Refinement Report — {refine_id}",
        "",
        f"- from_stage: {plan.min_from_stage.name} ({int(plan.min_from_stage)})",
        f"- to_stage: {Stage.CITATION_VERIFY.name} ({int(Stage.CITATION_VERIFY)})",
        f"- quality_before: {quality_before}",
        f"- quality_after: {quality_after}",
        f"- verifier_severity: {verifier_severity}",
        f"- items_addressed: {addressed}/{total}",
        f"- overall_pass: {overall_pass}",
        "",
        "## Items",
        "",
        "| ID | Category | Addressed | Text |",
        "|---|---|---|---|",
    ]
    for row in per_item:
        text = str(row["text"]).replace("|", "\\|").replace("\n", " ")[:120]
        md_lines.append(
            f"| {row['id']} | {row['category']} | {row['addressed']} | {text} |"
        )
    (feedback_root / "refinement_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    return RefineReport(
        refine_id=refine_id,
        from_stage=plan.min_from_stage,
        to_stage=Stage.CITATION_VERIFY,
        quality_before=quality_before,
        quality_after=quality_after,
        verifier_severity=verifier_severity,
        items_total=total,
        items_addressed=addressed,
        overall_pass=overall_pass,
        report_path=report_json_path,
    )
