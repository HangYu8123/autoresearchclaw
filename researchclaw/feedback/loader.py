"""Parse a human-feedback file into structured FeedbackItem records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Category = Literal["paper", "experiment", "code", "consistency"]
_VALID_CATEGORIES: set[str] = {"paper", "experiment", "code", "consistency"}

_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*)$")
_TAG_RE = re.compile(r"^\[([A-Za-z]+)\]\s*(.*)$")


@dataclass
class FeedbackItem:
    id: int
    category: Category
    text: str
    source_line: int | None = None


@dataclass
class FeedbackDocument:
    path: Path
    raw_text: str
    items: list[FeedbackItem] = field(default_factory=list)


def _classify_heuristic(text: str) -> Category:
    t = text.lower()
    consistency_kw = [
        "mismatch", "inconsistent", "does not match", "doesn't match",
        "paper says", "algorithm", "vs.", " vs ", "contradict",
    ]
    code_kw = [
        "bug", "off-by-one", "exception", "import", "stack trace",
        "traceback", " line ", "refactor",
    ]
    experiment_kw = [
        "baseline", "ablation", "seed", "metric", "hyperparam", "training",
        "dataset", "runtime", "variance", "re-run", "rerun",
    ]
    paper_kw = [
        "abstract", "intro", "section", "figure", "table", "prose", "writing",
        "wording", "citation", "reference", "caption", "hedging", "tighten",
    ]
    if any(k in t for k in consistency_kw):
        return "consistency"
    has_paper = any(k in t for k in paper_kw)
    if any(k in t for k in code_kw) and not has_paper:
        return "code"
    if any(k in t for k in experiment_kw):
        return "experiment"
    return "paper"


def parse_feedback(path: Path) -> FeedbackDocument:
    raw = path.read_text(encoding="utf-8")
    items: list[FeedbackItem] = []
    next_id = 1
    for lineno, line in enumerate(raw.splitlines(), start=1):
        m = _BULLET_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue
        category: Category | None = None
        tag_match = _TAG_RE.match(body)
        if tag_match:
            tag = tag_match.group(1).lower()
            if tag in _VALID_CATEGORIES:
                category = tag  # type: ignore[assignment]
                body = tag_match.group(2).strip()
        if category is None:
            category = _classify_heuristic(body)
        items.append(FeedbackItem(id=next_id, category=category, text=body, source_line=lineno))
        next_id += 1

    if not items:
        items.append(FeedbackItem(id=1, category="paper", text=raw.strip(), source_line=None))

    return FeedbackDocument(path=path, raw_text=raw, items=items)
