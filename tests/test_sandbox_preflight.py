from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from researchclaw.experiment.sandbox import validate_python_path
from researchclaw.pipeline.stage_impls._code_generation import _execute_code_generation
from researchclaw.pipeline.stages import StageStatus


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
