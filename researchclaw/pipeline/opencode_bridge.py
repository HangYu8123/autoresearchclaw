"""OpenCode 'Beast Mode' bridge — routes complex code generation to OpenCode CLI.

OpenCode (https://github.com/anomalyco/opencode) is an external AI coding agent
invoked via ``opencode run --format json "prompt"``.  This module provides:

1. **ComplexityScore / score_complexity()** — analyses an experiment plan to
   decide whether beast mode is warranted.
2. **OpenCodeBridge** — manages workspace creation, OpenCode invocation, file
   collection, and cleanup.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

# Keywords that indicate multi-component architectures
_COMPONENT_KEYWORDS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "discriminator",
    "generator",
    "critic",
    "actor",
    "teacher",
    "student",
    "backbone",
    "head",
    "neck",
    "classifier",
    "embedder",
    "attention",
    "transformer",
    "tokenizer",
    "vae",
    "autoencoder",
)

# Indicators that multi-file generation is needed
_FILE_HINT_KEYWORDS: tuple[str, ...] = (
    "model.py",
    "trainer.py",
    "dataset.py",
    "utils.py",
    "config.py",
    "multiple files",
    "modular",
    "separate module",
    "multi-file",
)

# Domain-complexity keywords
_DOMAIN_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "multi-modal",
    "multimodal",
    "distributed",
    "gan",
    "diffusion",
    "nerf",
    "mixture of experts",
    "moe",
    "meta-learning",
    "meta learning",
    "maml",
    "neural ode",
    "neural sde",
    "physics-informed",
    "pinn",
    "graph neural",
    "gnn",
    "reinforcement learning",
    "multi-agent",
    "world model",
    "vision-language",
    "text-to-image",
    "image-to-text",
)

# Patterns suggesting deep dependency chains
_DEPENDENCY_KEYWORDS: tuple[str, ...] = (
    "custom layer",
    "custom loss",
    "wrapper",
    "registry",
    "hook",
    "callback",
    "scheduler",
    "custom optimizer",
    "custom dataset",
    "custom sampler",
    "custom transform",
)


@dataclass
class ComplexityScore:
    """Result of complexity analysis on an experiment plan."""

    score: float  # 0.0-1.0
    signals: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""  # "beast_mode" | "code_agent" | "legacy"
    reason: str = ""


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_complexity(
    exp_plan: str,
    topic: str = "",
    *,
    historical_failures: int = 0,
    threshold: float = 0.6,
) -> ComplexityScore:
    """Score the complexity of an experiment to determine if beast mode is warranted.

    Returns a ComplexityScore with score in [0.0, 1.0].
    """
    if not exp_plan and not topic:
        return ComplexityScore(
            score=0.0,
            signals={},
            recommendation="legacy",
            reason="Empty plan",
        )

    combined = f"{topic}\n{exp_plan}"

    # Signal 1: Component count (weight 0.25)
    comp_hits = _count_keyword_hits(combined, _COMPONENT_KEYWORDS)
    component_score = min(comp_hits / 5.0, 1.0)

    # Signal 2: File count hint (weight 0.20)
    file_hits = _count_keyword_hits(combined, _FILE_HINT_KEYWORDS)
    file_score = min(file_hits / 3.0, 1.0)

    # Signal 3: Domain complexity (weight 0.20)
    domain_hits = _count_keyword_hits(combined, _DOMAIN_COMPLEX_KEYWORDS)
    domain_score = min(domain_hits / 3.0, 1.0)

    # Signal 4: Condition count (weight 0.15)
    # Look for numbered conditions, ablation mentions, variant mentions
    condition_pattern = re.compile(
        r"(?:condition|ablation|variant|experiment)\s*[\-_:]?\s*\d+",
        re.IGNORECASE,
    )
    condition_matches = len(condition_pattern.findall(combined))
    # Also count bullet points in conditions/ablations sections
    condition_matches += combined.lower().count("baseline")
    condition_score = min(condition_matches / 8.0, 1.0)

    # Signal 5: Historical failures (weight 0.10)
    failure_score = min(historical_failures / 3.0, 1.0)

    # Signal 6: Dependency depth (weight 0.10)
    dep_hits = _count_keyword_hits(combined, _DEPENDENCY_KEYWORDS)
    dep_score = min(dep_hits / 3.0, 1.0)

    # Weighted sum
    weighted = (
        0.25 * component_score
        + 0.20 * file_score
        + 0.20 * domain_score
        + 0.15 * condition_score
        + 0.10 * failure_score
        + 0.10 * dep_score
    )
    final_score = min(max(weighted, 0.0), 1.0)

    signals = {
        "component_count": round(component_score, 3),
        "file_count_hint": round(file_score, 3),
        "domain_complexity": round(domain_score, 3),
        "condition_count": round(condition_score, 3),
        "historical_failure": round(failure_score, 3),
        "dependency_depth": round(dep_score, 3),
    }

    if final_score >= threshold:
        recommendation = "beast_mode"
        reason = (
            f"Complexity {final_score:.2f} >= threshold {threshold:.2f}: "
            f"top signals: "
            + ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(signals.items(), key=lambda x: -x[1])[:3]
            )
        )
    else:
        recommendation = "code_agent"
        reason = f"Complexity {final_score:.2f} < threshold {threshold:.2f}"

    return ComplexityScore(
        score=round(final_score, 4),
        signals=signals,
        recommendation=recommendation,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# OpenCode bridge
# ---------------------------------------------------------------------------

@dataclass
class OpenCodeResult:
    """Result from an OpenCode invocation."""

    success: bool
    files: dict[str, str] = field(default_factory=dict)
    opencode_log: str = ""
    elapsed_sec: float = 0.0
    error: str = ""


_MEGA_PROMPT_TEMPLATE = """\
You are implementing a complete, runnable ML/science experiment.

Read the files in the current workspace:
- EXPERIMENT_PLAN.yaml — the full experiment design
- GUIDANCE.md — topic, metric, environment constraints, domain-specific guidance

Your task:
1. Design the file structure (main.py is the required entry point).
2. Implement ALL files with complete, runnable code. No placeholders or TODOs.
3. main.py must be the entry point and print the primary metric as:
   {metric}: <value>
4. Include numerical stability guards (gradient clipping, NaN detection, etc.).
5. Use multi-seed evaluation (seeds 0, 1, 2) and report mean ± std.
6. Each ablation/condition MUST be genuinely different — not copy-paste with a renamed variable.
7. Implement a time guard: stop gracefully at 80% of the time budget ({time_budget_sec} seconds).
8. Write requirements.txt listing any extra pip packages needed.
9. If the experiment needs dataset downloads, write a setup.py that handles them.

IMPORTANT CONSTRAINTS:
- The code will run in an isolated Docker container with PyTorch, torchvision, and common ML packages pre-installed.
- Do NOT use argparse or CLI arguments — hardcode all configuration.
- All output must go to stdout (print statements).
- Keep the experiment feasible within {time_budget_sec} seconds total.
"""


class OpenCodeBridge:
    """Manages OpenCode CLI invocations for beast mode code generation."""

    def __init__(
        self,
        *,
        model: str = "",
        llm_base_url: str = "",
        api_key_env: str = "",
        api_key: str = "",
        llm_provider: str = "openai-compatible",
        timeout_sec: int = 600,
        max_retries: int = 1,
        workspace_cleanup: bool = True,
    ) -> None:
        self._model = (model or "").strip()
        self._llm_base_url = llm_base_url
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._llm_provider = llm_provider
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._workspace_cleanup = workspace_cleanup

    # -- availability check ---------------------------------------------------

    @staticmethod
    def check_available() -> bool:
        """Return True if the ``opencode`` CLI is installed and callable."""
        opencode_cmd = shutil.which("opencode")
        if not opencode_cmd:
            return False
            
        try:
            result = subprocess.run(
                [opencode_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:  # noqa: BLE001
            return False

    # -- workspace preparation ------------------------------------------------

    def _prepare_workspace(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        extra_guidance: str,
        time_budget_sec: int,
    ) -> Path:
        """Create a temporary workspace directory with context files."""
        ws = stage_dir / f"opencode_beast_{int(time.time())}_{time.monotonic_ns() % 100000}"
        ws.mkdir(parents=True, exist_ok=True)

        # Write experiment plan
        (ws / "EXPERIMENT_PLAN.yaml").write_text(
            exp_plan or "# No experiment plan provided\n",
            encoding="utf-8",
        )

        # Write guidance document
        guidance_parts = [
            f"# Experiment Guidance\n",
            f"## Topic\n{topic}\n",
            f"## Primary Metric\n{metric}\n",
            f"## Time Budget\n{time_budget_sec} seconds\n",
        ]
        if pkg_hint:
            guidance_parts.append(f"## Environment\n{pkg_hint}\n")
        if extra_guidance:
            guidance_parts.append(f"## Additional Guidance\n{extra_guidance}\n")
        (ws / "GUIDANCE.md").write_text(
            "\n".join(guidance_parts), encoding="utf-8",
        )

        # Write opencode.json config
        opencode_cfg = self._build_opencode_config()
        (ws / "opencode.json").write_text(
            json.dumps(opencode_cfg, indent=2), encoding="utf-8",
        )

        # OpenCode requires a git repository — initialise one with
        # a single commit so that ``opencode run`` doesn't hang.
        # BUG-OB-01/OB-02: Check return codes and catch TimeoutExpired.
        try:
            r = subprocess.run(
                ["git", "init"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git init failed: {r.stderr}")
            r = subprocess.run(
                ["git", "add", "-A"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git add failed: {r.stderr}")
            r = subprocess.run(
                ["git", "-c", "user.email=beast@researchclaw",
                 "-c", "user.name=BeastMode",
                 "commit", "-m", "init workspace"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git commit failed: {r.stderr}")
        except subprocess.TimeoutExpired as exc:
            raise OSError(f"git workspace init timed out: {exc}") from exc

        return ws

    def _is_azure(self) -> bool:
        """Detect Azure OpenAI from base URL or provider string."""
        return (
            "azure" in (self._llm_base_url or "").lower()
            or "azure" in (self._llm_provider or "").lower()
        )

    def _provider_id(self) -> str:
        """Return the OpenCode provider id to register in opencode.json."""
        raw = (self._llm_provider or "openai-compatible").strip().lower()
        provider = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-")
        return provider or "openai-compatible"

    def _model_provider_and_id(self) -> tuple[str, str]:
        """Return (provider_id, provider-local model id) for OpenCode."""
        if not self._model:
            raise ValueError(
                "OpenCode model is empty; pass config.llm.primary_model or "
                "configure experiment.opencode.model"
            )

        provider_id = self._provider_id()
        model = self._model
        if "/" in model:
            first, rest = model.split("/", 1)
            if first == provider_id:
                return first, rest
            if provider_id == "openrouter":
                return provider_id, model
            # Respect an explicit provider/model override instead of stripping it.
            return first, rest
        return provider_id, model

    def _opencode_api_key_env_name(self) -> str:
        """Environment variable name referenced by opencode.json."""
        if self._api_key_env:
            return self._api_key_env
        if self._api_key:
            return "RESEARCHCLAW_OPENCODE_API_KEY"
        return ""

    def _resolved_api_key(self) -> str:
        """Resolve inline/env API key without exposing it in artifacts."""
        if self._api_key:
            return self._api_key
        if self._api_key_env:
            return os.environ.get(self._api_key_env, "")
        return ""

    def _redact_sensitive(self, text: str) -> str:
        """Redact API keys that may appear in subprocess output."""
        redacted = text or ""
        for secret in {
            self._api_key,
            os.environ.get(self._api_key_env, "") if self._api_key_env else "",
        }:
            if secret and len(secret) >= 4:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _build_opencode_config(self) -> dict[str, Any]:
        """Build the opencode.json configuration.

        Uses a project-local custom provider so OpenCode can address the model
        as ``provider/model`` and route OpenAI-compatible traffic to the
        configured base URL.
        """
        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
        }

        provider_id, model_id = self._model_provider_and_id() if self._model else ("", "")
        resolved_model = self._resolve_opencode_model() if self._model else ""
        if self._llm_base_url:
            if resolved_model:
                cfg["model"] = resolved_model
            api_key_env = self._opencode_api_key_env_name()
            options: dict[str, Any] = {"baseURL": self._llm_base_url}
            if api_key_env:
                options["apiKey"] = f"{{env:{api_key_env}}}"
            cfg["provider"] = {
                provider_id: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_id,
                    "options": {
                        **options,
                    },
                    "models": {},
                }
            }
            # Register the model so OpenCode knows it exists
            if model_id:
                cfg["provider"][provider_id]["models"] = {
                    model_id: {
                        "name": model_id,
                    }
                }
        elif self._model:
            cfg["model"] = resolved_model

        return cfg

    # -- model resolution -------------------------------------------------------

    def _resolve_opencode_model(self) -> str:
        """Resolve the model identifier for OpenCode CLI's ``-m`` flag.

        OpenCode expects model names in ``provider/model`` form.  Bare models
        are qualified with the configured provider id; explicitly qualified
        models are preserved.
        """
        provider_id, model_id = self._model_provider_and_id()
        return f"{provider_id}/{model_id}"

    # -- invocation ------------------------------------------------------------

    @staticmethod
    def _kill_process_tree(proc: subprocess.Popen) -> None:
        """Kill a process and its entire process group / tree.

        Uses ``os.killpg`` on POSIX (requires ``start_new_session=True`` when
        spawning so the process gets its own pgid) and ``taskkill /F /T`` on
        Windows so that any grandchild processes spawned by OpenCode are also
        terminated.  Falls back to plain ``proc.kill()`` if either approach
        fails.
        """
        try:
            if os.name != "nt":
                import signal as _signal  # noqa: PLC0415
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                    return
                except ProcessLookupError:
                    return  # already gone
                except Exception:  # noqa: BLE001
                    pass  # fall through to proc.kill()
            else:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=15,
                    )
                    return
                except Exception:  # noqa: BLE001
                    pass  # fall through to proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def _invoke_opencode(
        self,
        workspace: Path,
        prompt: str,
    ) -> tuple[bool, str, float]:
        """Run ``opencode run`` in the workspace. Returns (success, log, elapsed).

        Uses ``Popen`` directly (instead of ``subprocess.run``) so that, on
        timeout, we can kill the *entire process group* (OpenCode spawns child
        processes that inherit stdout/stderr pipes; killing only the parent
        leaves children running and makes the subsequent ``communicate()``
        block indefinitely — causing >1000 s overruns despite a 600 s limit).
        After the group kill we drain remaining pipe data with a short bounded
        ``communicate(timeout=30)`` to avoid any residual deadlock.
        """
        env = os.environ.copy()
        api_key_env = self._opencode_api_key_env_name()
        api_key = self._resolved_api_key()
        if api_key_env and api_key:
            env[api_key_env] = api_key
        env["OPENCODE_CONFIG"] = str(workspace / "opencode.json")

        # Use -m flag to specify model (more reliable than opencode.json)
        resolved_model = self._resolve_opencode_model()
        opencode_cmd = shutil.which("opencode") or "opencode"
        cmd = [
            opencode_cmd,
            "run",
            "-m",
            resolved_model,
            "--format",
            "json",
            "--dangerously-skip-permissions",
            prompt,
        ]

        # Put OpenCode in its own session/process group so the whole tree
        # can be killed atomically on timeout.
        popen_kwargs: dict[str, Any] = {
            "cwd": str(workspace),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "env": env,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        t0 = time.monotonic()
        wall_t0 = time.time()
        deadline = t0 + self._timeout_sec
        output_stable_sec = 45
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_process_tree(proc)
                    try:
                        stdout, stderr = proc.communicate(timeout=30)
                    except subprocess.TimeoutExpired:
                        stdout, stderr = "", ""
                    elapsed = time.monotonic() - t0
                    partial_out = (stdout or "")[:2000]
                    log = f"TIMEOUT after {elapsed:.1f}s"
                    if partial_out:
                        log += f"\nstdout: {partial_out}"
                    return False, self._redact_sensitive(log), elapsed

                try:
                    stdout, stderr = proc.communicate(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    newest_output = self._collectible_output_mtime(workspace)
                    if (
                        newest_output is not None
                        and time.time() - newest_output >= output_stable_sec
                        and time.time() - wall_t0 >= output_stable_sec
                    ):
                        self._kill_process_tree(proc)
                        try:
                            stdout, stderr = proc.communicate(timeout=30)
                        except subprocess.TimeoutExpired:
                            stdout, stderr = "", ""
                        elapsed = time.monotonic() - t0
                        log = (
                            "OpenCode produced stable collectable files "
                            f"after {elapsed:.1f}s; stopped lingering process."
                        )
                        if stdout:
                            log += f"\nstdout: {stdout[:2000]}"
                        if stderr:
                            log += f"\nstderr: {stderr[:2000]}"
                        return True, self._redact_sensitive(log), elapsed

            elapsed = time.monotonic() - t0
            log = self._redact_sensitive((stdout or "") + "\n" + (stderr or ""))
            return proc.returncode == 0, log, elapsed
        except FileNotFoundError:
            return False, "opencode CLI not found", 0.0
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            return False, self._redact_sensitive(f"Unexpected error: {exc}"), elapsed
        finally:
            # Ensure the process is reaped even if an unexpected exception occurs.
            if proc is not None and proc.poll() is None:
                try:
                    self._kill_process_tree(proc)
                except Exception:  # noqa: BLE001
                    pass

    # -- file collection -------------------------------------------------------

    @staticmethod
    def _collectible_output_mtime(workspace: Path) -> float | None:
        paths = [p for p in workspace.rglob("*.py") if "__pycache__" not in p.parts]
        for extra in ("requirements.txt", "setup.py"):
            path = workspace / extra
            if path.exists():
                paths.append(path)
        if not any(path.name == "main.py" for path in paths):
            return None
        try:
            return max(path.stat().st_mtime for path in paths)
        except OSError:
            return None

    @staticmethod
    def _collect_files(workspace: Path) -> dict[str, str]:
        """Collect generated Python files, requirements.txt, and setup.py.

        File names are flattened to basenames (e.g. ``src/main.py`` → ``main.py``)
        because the downstream executor expects a flat file dict.  If two files
        share the same basename, the one closer to the workspace root wins.
        """
        files: dict[str, str] = {}
        # Sort by depth (fewer parts first) so root-level files take priority
        py_files = sorted(
            workspace.rglob("*.py"),
            key=lambda p: len(p.relative_to(workspace).parts),
        )
        for py_file in py_files:
            rel = py_file.relative_to(workspace)
            parts = rel.parts
            if any(p.startswith("__pycache__") or p.startswith(".") for p in parts):
                continue
            # Flatten to basename — executor expects flat structure
            basename = rel.name
            if basename not in files:
                try:
                    files[basename] = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Beast mode: failed to read %s: %s", py_file, exc)

        # Also collect requirements.txt and setup.py at root
        for extra in ("requirements.txt", "setup.py"):
            p = workspace / extra
            if p.exists() and extra not in files:
                files[extra] = p.read_text(encoding="utf-8", errors="replace")

        return files

    # -- entry-point validation ------------------------------------------------

    @staticmethod
    def _has_main_guard(source: str) -> bool:
        """Return True if *source* contains ``if __name__ == "__main__":``."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
                    if test.left.id == "__name__" and len(test.comparators) == 1:
                        comp = test.comparators[0]
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            return True
        return False

    @staticmethod
    def _ensure_main_entry_point(files: dict[str, str]) -> dict[str, str]:
        """Ensure ``main.py`` has an ``if __name__ == "__main__"`` guard.

        Beast Mode often generates multi-file projects where ``main.py`` is a
        library module and the real entry point lives in another file (e.g.
        ``run_experiment.py``).  Since the Docker sandbox always executes
        ``python3 main.py``, a library-only ``main.py`` exits immediately with
        no output.

        Strategy:
        1. If ``main.py`` already has the guard → return unchanged.
        2. Find the first other ``.py`` file that **does** have the guard.
        3. Swap: rename that file to ``main.py`` and the old ``main.py`` to a
           helper module (its original basename, or ``_lib.py``).
        4. If no file has a guard, append a minimal stub to ``main.py`` that
           calls the most likely entry function (``main()``, ``run()``, etc.).
        """
        main_code = files.get("main.py", "")
        if not main_code:
            return files

        if OpenCodeBridge._has_main_guard(main_code):
            return files

        # -- Strategy 2/3: find another file with the guard and swap -----------
        for fname, code in files.items():
            if fname == "main.py" or not fname.endswith(".py"):
                continue
            if OpenCodeBridge._has_main_guard(code):
                logger.info(
                    "Beast mode: main.py lacks __main__ guard; swapping "
                    "entry point with %s",
                    fname,
                )
                new_files = dict(files)
                # Rename original main.py → helper module
                helper_name = fname  # reuse the other file's name for old main
                new_files[helper_name] = main_code
                new_files["main.py"] = code
                return new_files

        # -- Strategy 4: inject a minimal entry point into main.py -------------
        # Look for common entry functions defined in main.py
        entry_func: str | None = None
        try:
            tree = ast.parse(main_code)
            candidates = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ("main", "run", "run_experiment", "train",
                               "run_experiments", "experiment", "run_all")
            ]
            if candidates:
                entry_func = candidates[0]
        except SyntaxError:
            pass

        if entry_func:
            logger.info(
                "Beast mode: main.py lacks __main__ guard; injecting call "
                "to %s()",
                entry_func,
            )
            new_files = dict(files)
            new_files["main.py"] = (
                main_code.rstrip()
                + "\n\n\nif __name__ == \"__main__\":\n"
                + f"    {entry_func}()\n"
            )
            return new_files

        logger.warning(
            "Beast mode: main.py lacks __main__ guard and no known entry "
            "function found — experiment may exit without producing output",
        )
        return files

    # -- main entry point ------------------------------------------------------

    def generate(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str = "",
        extra_guidance: str = "",
        time_budget_sec: int = 300,
    ) -> OpenCodeResult:
        """Run OpenCode to generate experiment code.

        Returns an OpenCodeResult with success status and generated files.
        """
        # Check availability first
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="OpenCode CLI not installed or not callable",
            )
        try:
            self._resolve_opencode_model()
        except ValueError as exc:
            return OpenCodeResult(success=False, error=str(exc))

        workspace: Path | None = None
        last_error = ""
        last_elapsed = 0.0
        attempts_made = 0

        for attempt in range(1 + self._max_retries):
            attempts_made = attempt + 1
            # Prepare workspace
            try:
                workspace = self._prepare_workspace(
                    stage_dir=stage_dir,
                    topic=topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint,
                    extra_guidance=extra_guidance,
                    time_budget_sec=time_budget_sec,
                )
            except OSError as exc:
                last_error = f"Failed to prepare workspace: {exc}"
                logger.warning("Beast mode: %s", last_error)
                continue

            # Build the mega-prompt (use replace instead of .format() to
            # avoid KeyError when metric contains curly braces like "F{1}")
            prompt = _MEGA_PROMPT_TEMPLATE.replace(
                "{metric}", metric
            ).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )

            logger.info(
                "Beast mode: invoking OpenCode (attempt %d/%d, timeout=%ds)",
                attempt + 1,
                1 + self._max_retries,
                self._timeout_sec,
            )

            success, log, elapsed = self._invoke_opencode(workspace, prompt)

            if success:
                files = self._collect_files(workspace)
                if "main.py" not in files:
                    logger.warning(
                        "Beast mode: OpenCode succeeded but no main.py found "
                        "(files: %s)", list(files.keys()),
                    )
                    last_error = "No main.py in OpenCode output"
                    # Cleanup failed workspace
                    if self._workspace_cleanup and workspace.exists():
                        shutil.rmtree(workspace, ignore_errors=True)
                    continue

                # BUG-R52-01: Ensure main.py has an entry point
                files = self._ensure_main_entry_point(files)

                # Write log
                try:
                    (stage_dir / "opencode_log.txt").write_text(
                        log or "", encoding="utf-8",
                    )
                except OSError as _wexc:
                    logger.warning("Beast mode: failed to write log: %s", _wexc)

                # Cleanup workspace if configured
                if self._workspace_cleanup and workspace.exists():
                    shutil.rmtree(workspace, ignore_errors=True)

                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = log
            last_elapsed = elapsed
            logger.warning(
                "Beast mode: OpenCode attempt %d failed (%.1fs): %s",
                attempt + 1,
                elapsed,
                log[:500],
            )
            # Cleanup failed workspace
            if self._workspace_cleanup and workspace and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)
            if log.startswith("TIMEOUT"):
                break

        # All attempts failed
        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            elapsed_sec=last_elapsed,
            error=f"OpenCode failed after {attempts_made or 1} attempt(s)",
        )


# ---------------------------------------------------------------------------
# Helper: count historical failures
# ---------------------------------------------------------------------------

def count_historical_failures(run_dir: Path, stage_name: str = "stage-10") -> int:
    """Count past Stage 10 failures from stage directories and logs.

    Each stage directory is counted at most once, even if multiple failure
    indicators are present.
    """
    failures = 0
    for d in run_dir.glob(f"{stage_name}*"):
        failed = False
        # Check for beast_mode_log.json
        bm_log = d / "beast_mode_log.json"
        if bm_log.exists():
            try:
                data = json.loads(bm_log.read_text(encoding="utf-8"))
                if not data.get("success", True):
                    failed = True
            except Exception:  # noqa: BLE001
                pass
        # Check for stage health failures
        if not failed:
            health = d / "stage_health.json"
            if health.exists():
                try:
                    data = json.loads(health.read_text(encoding="utf-8"))
                    if data.get("status") == "FAILED":
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        # Check for validation report with FAILED status
        if not failed:
            vr = d / "validation_report.md"
            if vr.exists():
                try:
                    content = vr.read_text(encoding="utf-8")
                    if "BLOCKED" in content or "FAILED" in content:
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        if failed:
            failures += 1
    return failures
