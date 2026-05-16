"""Quick-start interactive setup wizard."""

from __future__ import annotations

import sys
from typing import Any

from researchclaw.wizard.templates import TEMPLATES


_WIZARD_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "primary_model": "gpt-4o",
        "fallback_models": ["gpt-4.1", "gpt-4o-mini"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "primary_model": "anthropic/claude-3.5-sonnet",
        "fallback_models": [
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-70b-instruct",
        ],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "primary_model": "deepseek-v4-pro",
        "fallback_models": ["deepseek-v4-flash"],
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "primary_model": "MiniMax-M2.5",
        "fallback_models": ["MiniMax-M2.5-highspeed"],
    },
    "volcengine": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "VOLCENGINE_API_KEY",
        "primary_model": "doubao-seed-2-0-pro-260215",
        "fallback_models": [
            "doubao-seed-2-0-lite-260215",
            "doubao-seed-2-0-mini-260215",
            "doubao-seed-2-0-code-preview-260215",
            "kimi-k2-5-260127",
            "glm-4-7-251222",
            "deepseek-v3-2-251201",
        ],
    },
    "volcengine-coding-plan": {
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "api_key_env": "VOLCENGINE_API_KEY",
        "primary_model": "doubao-seed-2.0-code",
        "fallback_models": [
            "doubao-seed-2.0-pro",
            "doubao-seed-2.0-lite",
            "doubao-seed-code",
            "minimax-m2.5",
            "glm-4.7",
            "deepseek-v3.2",
            "kimi-k2.5",
        ],
    },
    "byteplus": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "api_key_env": "BYTEPLUS_API_KEY",
        "primary_model": "seed-2-0-pro-260328",
        "fallback_models": [
            "seed-2-0-lite-260228",
            "seed-2-0-mini-260215",
            "kimi-k2-5-260127",
            "glm-4-7-251222",
        ],
    },
    "byteplus-coding-plan": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/coding/v3",
        "api_key_env": "BYTEPLUS_API_KEY",
        "primary_model": "dola-seed-2.0-pro",
        "fallback_models": [
            "dola-seed-2.0-lite",
            "bytedance-seed-code",
            "glm-4.7",
            "kimi-k2.5",
            "gpt-oss-120b",
        ],
    },
}


class QuickStartWizard:
    """Interactive configuration generator."""

    def run_interactive(self, template: str | None = None) -> dict[str, Any]:
        """CLI interactive wizard — returns a config dict."""
        print("\n=== ResearchClaw Setup Wizard ===\n")

        if template:
            return self._apply_template(template)

        config: dict[str, Any] = {}

        # 1. Project name
        name = self._ask("Project name", default="my-research")
        config["project"] = {"name": name, "mode": "full-auto"}

        # 2. Research topic
        topic = self._ask("Research topic (describe in one sentence)")
        if not topic:
            print("Topic is required.")
            return {}
        config["research"] = {"topic": topic}

        # 3. Research domain
        domains_str = self._ask(
            "Research domains (comma-separated: cv, nlp, rl, ml, ai4science)",
            default="ml",
        )
        config["research"]["domains"] = [
            d.strip() for d in domains_str.split(",") if d.strip()
        ]

        # 4. Experiment mode
        mode = self._choose(
            "Experiment mode",
            ["simulated", "docker", "sandbox"],
            default="docker",
        )
        config["experiment"] = {"mode": mode}

        if mode == "docker":
            gpu = self._ask_yn("Enable GPU?", default=True)
            config["experiment"]["docker"] = {
                "gpu_enabled": gpu,
                "network_policy": "setup_only",
            }
            budget = self._ask("Time budget (seconds)", default="600")
            config["experiment"]["time_budget_sec"] = int(budget)

        # 5. LLM provider
        print("\n--- LLM Configuration ---")
        provider = self._choose(
            "LLM provider",
            [
                "openai-compatible",
                "openai",
                "openrouter",
                "deepseek",
                "minimax",
                "volcengine",
                "volcengine-coding-plan",
                "byteplus",
                "byteplus-coding-plan",
                "acp",
            ],
            default="openai-compatible",
        )
        config["llm"] = {"provider": provider}

        if provider == "openai-compatible":
            base_url = self._ask("API base URL", default="https://api.openai.com/v1")
            api_key_env = self._ask("API key env var", default="OPENAI_API_KEY")
            model = self._ask("Model name", default="gpt-4o")
            config["llm"].update({
                "base_url": base_url,
                "api_key_env": api_key_env,
                "primary_model": model,
            })
        elif provider != "acp":
            config["llm"].update(_WIZARD_PROVIDER_DEFAULTS[provider])

        # 6. Output format
        conference = self._choose(
            "Target conference format",
            ["neurips_2025", "iclr_2025", "icml_2025", "arxiv"],
            default="neurips_2025",
        )
        config["export"] = {"target_conference": conference}

        # 7. Runtime
        config["runtime"] = {"timezone": "UTC"}
        config["notifications"] = {"channel": "console"}
        config["knowledge_base"] = {"backend": "markdown", "root": "knowledge"}

        print("\n--- Configuration Summary ---")
        self._print_summary(config)
        confirm = self._ask_yn("\nSave this configuration?", default=True)
        if not confirm:
            print("Cancelled.")
            return {}

        return config

    def run_web(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Process wizard steps from web interface."""
        config: dict[str, Any] = {}
        for step in steps:
            key = step.get("key", "")
            value = step.get("value", "")
            if key == "project_name":
                config.setdefault("project", {})["name"] = value
            elif key == "topic":
                config.setdefault("research", {})["topic"] = value
            elif key == "mode":
                config.setdefault("experiment", {})["mode"] = value
            elif key == "model":
                config.setdefault("llm", {})["primary_model"] = value
        return config

    def _apply_template(self, name: str) -> dict[str, Any]:
        """Apply a preset template."""
        mapping = {
            "quick": "quick-demo",
            "standard": "standard-cv",
            "advanced": "deep-nlp",
        }
        tpl_name = mapping.get(name, name)
        tpl = TEMPLATES.get(tpl_name)
        if not tpl:
            print(f"Unknown template: {name}")
            return {}

        config = self._template_to_config(tpl)
        print(f"Applied template: {tpl_name}")
        print(f"  Description: {tpl.get('description', '')}")
        self._print_summary(config)
        return config

    def _template_to_config(self, tpl: dict[str, Any]) -> dict[str, Any]:
        """Convert a flat template to nested config dict."""
        config: dict[str, Any] = {
            "project": {"name": "wizard-project", "mode": "full-auto"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "console"},
            "knowledge_base": {"backend": "markdown", "root": "knowledge"},
            "research": {"topic": "Generated by wizard"},
            "llm": {"provider": "openai-compatible", "api_key_env": "OPENAI_API_KEY"},
        }

        for key, value in tpl.items():
            if key == "description":
                continue
            parts = key.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value

        return config

    def _ask(self, prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"  {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        return answer or default

    def _ask_yn(self, prompt: str, default: bool = True) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        try:
            answer = input(f"  {prompt}{suffix}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not answer:
            return default
        return answer in ("y", "yes", "1", "true")

    def _choose(
        self,
        prompt: str,
        options: list[str],
        default: str = "",
    ) -> str:
        print(f"  {prompt}:")
        for i, opt in enumerate(options, 1):
            marker = " *" if opt == default else ""
            print(f"    {i}. {opt}{marker}")
        try:
            answer = input(f"  Choice [default={default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not answer:
            return default
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            if answer in options:
                return answer
        return default

    def _print_summary(self, config: dict[str, Any], indent: int = 2) -> None:
        import yaml

        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
