"""Dispatch parsed feedback items into the appropriate run-dir artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from researchclaw.feedback.loader import FeedbackDocument, FeedbackItem
from researchclaw.pipeline.stages import Stage


_CATEGORY_FROM_STAGE: dict[str, Stage] = {
    "paper": Stage.PAPER_OUTLINE,
    "consistency": Stage.PAPER_REVISION,
    "experiment": Stage.ITERATIVE_REFINE,
    "code": Stage.ITERATIVE_REFINE,
}


@dataclass
class DispatchPlan:
    min_from_stage: Stage
    categories_present: set[str]
    paths_written: list[Path] = field(default_factory=list)
    dispatch_json_path: Path | None = None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _numbered(items: list[FeedbackItem]) -> str:
    return "\n".join(f"{i+1}. {it.text}" for i, it in enumerate(items))


def dispatch_feedback(
    run_dir: Path, doc: FeedbackDocument, *, refine_id: str
) -> DispatchPlan:
    categories: set[str] = {it.category for it in doc.items}
    paths_written: list[Path] = []

    feedback_root = run_dir / "feedback" / refine_id
    feedback_root.mkdir(parents=True, exist_ok=True)

    # Audit copy of raw feedback.
    audit_path = feedback_root / "user_feedback.md"
    audit_path.write_text(doc.raw_text, encoding="utf-8")
    paths_written.append(audit_path)

    # --- paper -> iteration_context.json ---
    paper_items = [it for it in doc.items if it.category == "paper"]
    if paper_items:
        new_excerpt = _numbered(paper_items)
        ctx_path = run_dir / "iteration_context.json"
        prior: dict[str, object] = {}
        if ctx_path.exists():
            try:
                prior = json.loads(ctx_path.read_text(encoding="utf-8"))
                if not isinstance(prior, dict):
                    prior = {}
            except (json.JSONDecodeError, OSError):
                prior = {}
        prior_iter = prior.get("iteration", 1)
        try:
            prior_iter_int = int(prior_iter)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            prior_iter_int = 1
        prior_excerpt = prior.get("reviews_excerpt", "") or ""
        if not isinstance(prior_excerpt, str):
            prior_excerpt = ""
        merged_excerpt = (
            (prior_excerpt + "\n\n" if prior_excerpt else "")
            + "## Human Feedback (Refine Round)\n"
            + new_excerpt
        )
        merged = {
            "iteration": max(prior_iter_int + 1, 2),
            "quality_score": prior.get("quality_score"),
            "reviews_excerpt": merged_excerpt,
            "generated": _utc_iso(),
            "source": "user_feedback_refine",
        }
        ctx_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        paths_written.append(ctx_path)

    # --- consistency -> stage-18/reviews.md ---
    consistency_items = [it for it in doc.items if it.category == "consistency"]
    if consistency_items:
        reviews_dir = run_dir / "stage-18"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        reviews_path = reviews_dir / "reviews.md"
        append_block = (
            "\n\n## Human Feedback (Consistency Review)\n"
            + _numbered(consistency_items)
            + "\n"
        )
        if reviews_path.exists():
            backup_path = reviews_dir / "reviews.md.pre-refine"
            if not backup_path.exists():
                shutil.copy2(reviews_path, backup_path)
                paths_written.append(backup_path)
            with reviews_path.open("a", encoding="utf-8") as fh:
                fh.write(append_block)
        else:
            reviews_path.write_text(
                "# Peer Review (synthetic — user feedback)\n" + append_block,
                encoding="utf-8",
            )
        paths_written.append(reviews_path)

    # --- experiment / code -> feedback/user_directives.md ---
    directive_items = [it for it in doc.items if it.category in ("experiment", "code")]
    if directive_items:
        directives_path = run_dir / "feedback" / "user_directives.md"
        directives_path.parent.mkdir(parents=True, exist_ok=True)
        header = "# User Refinement Directives\n\n"
        directives_path.write_text(
            header + _numbered(directive_items) + "\n", encoding="utf-8"
        )
        paths_written.append(directives_path)

    min_stage = min(
        (_CATEGORY_FROM_STAGE[c] for c in categories if c in _CATEGORY_FROM_STAGE),
        key=int,
        default=Stage.PAPER_OUTLINE,
    )

    counts = {c: sum(1 for it in doc.items if it.category == c) for c in categories}
    dispatch_json_path = feedback_root / "dispatch.json"
    dispatch_json_path.write_text(
        json.dumps(
            {
                "refine_id": refine_id,
                "timestamp": _utc_iso(),
                "items_count_by_category": counts,
                "paths_written": [str(p) for p in paths_written],
                "min_from_stage_name": min_stage.name,
                "min_from_stage_num": int(min_stage),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths_written.append(dispatch_json_path)

    return DispatchPlan(
        min_from_stage=min_stage,
        categories_present=categories,
        paths_written=paths_written,
        dispatch_json_path=dispatch_json_path,
    )
