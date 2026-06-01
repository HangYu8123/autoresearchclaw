from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from researchclaw.experiment.sandbox import validate_python_path
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._code_generation import (
    _execute_code_generation,
    _stage10_generation_max_tokens,
)
from researchclaw.pipeline.stages import StageStatus
from researchclaw.prompts import PromptManager


class EmptyLLM:
    def __init__(self, content: str = "") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(content=self.content, model="fake-model")


def _stage10_config(
    mode: str = "sandbox", *, code_agent: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        research=SimpleNamespace(
            topic="machine learning classification benchmark",
            domains=("machine-learning",),
        ),
        llm=SimpleNamespace(primary_model="fake-model", max_tokens=65536),
        experiment=SimpleNamespace(
            mode=mode,
            metric_key="primary_metric",
            metric_direction="minimize",
            time_budget_sec=300,
            max_iterations=10,
            sandbox=SimpleNamespace(python_path=sys.executable),
            opencode=SimpleNamespace(enabled=False),
            code_agent=SimpleNamespace(enabled=code_agent),
        ),
    )


def test_validate_python_path_reports_missing_relative_path(tmp_path: Path) -> None:
    err = validate_python_path(".venv/bin/python3", base_dir=tmp_path)

    assert err is not None
    assert "experiment.sandbox.python_path does not exist" in err
    assert ".venv/bin/python3" in err


def test_stage10_fails_fast_on_invalid_sandbox_python(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)
    config = SimpleNamespace(
        experiment=SimpleNamespace(
            mode="sandbox",
            metric_key="primary_metric",
            sandbox=SimpleNamespace(python_path=str(tmp_path / "missing-python")),
        )
    )

    result = _execute_code_generation(
        stage_dir,
        run_dir,
        config,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        llm=None,
        prompts=None,
    )

    assert result.status is StageStatus.FAILED
    assert "Stage 10 sandbox preflight failed" in (result.error or "")
    assert (stage_dir / "validation_report.md").exists()


def test_stage10_real_mode_fails_when_generation_returns_no_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)

    result = _execute_code_generation(
        stage_dir,
        run_dir,
        _stage10_config("sandbox"),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        llm=EmptyLLM(""),
        prompts=PromptManager(),
    )

    assert result.status is StageStatus.FAILED
    assert "No experiment files were generated" in (result.error or "")
    diagnostics = json.loads(
        (stage_dir / "code_generation_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["status"] == "failed"
    assert diagnostics["fallback_allowed"] is False
    assert diagnostics["generated_files"] == []
    assert not (stage_dir / "experiment" / "main.py").exists()


def test_stage10_real_mode_fails_without_main_entrypoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)
    llm = EmptyLLM('```filename:utils.py\nprint("helper")\n```')

    result = _execute_code_generation(
        stage_dir,
        run_dir,
        _stage10_config("sandbox"),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        llm=llm,
        prompts=PromptManager(),
    )

    assert result.status is StageStatus.FAILED
    assert "entry point" in (result.error or "")
    diagnostics = json.loads(
        (stage_dir / "code_generation_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["status"] == "failed"
    assert diagnostics["fallback_allowed"] is False
    assert not (stage_dir / "experiment" / "main.py").exists()


def test_stage10_code_agent_exception_records_no_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from researchclaw.pipeline import code_agent as code_agent_module

    class RaisingCodeAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise TimeoutError("boom")

    monkeypatch.setattr(code_agent_module, "CodeAgent", RaisingCodeAgent)
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)

    result = _execute_code_generation(
        stage_dir,
        run_dir,
        _stage10_config("sandbox", code_agent=True),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        llm=EmptyLLM(""),
        prompts=PromptManager(),
    )

    assert result.status is StageStatus.FAILED
    code_agent_log = json.loads(
        (stage_dir / "code_agent_log.json").read_text(encoding="utf-8")
    )
    assert code_agent_log["fallback"] == "none"
    diagnostics = json.loads(
        (stage_dir / "code_generation_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["errors"][0]["source"] == "code_agent"


def test_stage10_simulated_mode_allows_topic_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)

    result = _execute_code_generation(
        stage_dir,
        run_dir,
        _stage10_config("simulated"),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        llm=EmptyLLM(""),
        prompts=PromptManager(),
    )

    assert result.status is StageStatus.DONE
    assert "code_generation_diagnostics.json" in result.artifacts
    diagnostics = json.loads(
        (stage_dir / "code_generation_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["status"] == "fallback"
    assert diagnostics["fallback_allowed"] is True
    assert (stage_dir / "experiment" / "main.py").exists()


def test_stage10_generation_tokens_use_prompt_budget_floor() -> None:
    pm = PromptManager()
    config = _stage10_config("sandbox")

    assert _stage10_generation_max_tokens(config, pm) == 65536
    assert _stage10_generation_max_tokens(config, None) == 65536

    config.llm.max_tokens = 4096
    assert _stage10_generation_max_tokens(config, pm) == 8192
    assert _stage10_generation_max_tokens(config, None) == 8192
