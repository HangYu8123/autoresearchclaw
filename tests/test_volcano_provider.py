"""Tests for Volcengine and BytePlus provider integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from researchclaw.llm import PROVIDER_PRESETS
from researchclaw.llm.client import LLMClient
from researchclaw.wizard.quickstart import QuickStartWizard


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("volcengine", "https://ark.cn-beijing.volces.com/api/v3"),
        ("volcengine-coding-plan", "https://ark.cn-beijing.volces.com/api/coding/v3"),
        ("byteplus", "https://ark.ap-southeast.bytepluses.com/api/v3"),
        ("byteplus-coding-plan", "https://ark.ap-southeast.bytepluses.com/api/coding/v3"),
    ],
)
def test_provider_presets_include_volcano_variants(
    provider: str, base_url: str
) -> None:
    assert PROVIDER_PRESETS[provider]["base_url"] == base_url


@pytest.mark.parametrize(
    ("provider", "env_var", "base_url", "primary_model"),
    [
        (
            "volcengine",
            "VOLCENGINE_API_KEY",
            "https://ark.cn-beijing.volces.com/api/v3",
            "doubao-seed-2-0-pro-260215",
        ),
        (
            "volcengine-coding-plan",
            "VOLCENGINE_API_KEY",
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "doubao-seed-2.0-code",
        ),
        (
            "byteplus",
            "BYTEPLUS_API_KEY",
            "https://ark.ap-southeast.bytepluses.com/api/v3",
            "seed-2-0-pro-260328",
        ),
        (
            "byteplus-coding-plan",
            "BYTEPLUS_API_KEY",
            "https://ark.ap-southeast.bytepluses.com/api/coding/v3",
            "dola-seed-2.0-pro",
        ),
    ],
)
def test_from_rc_config_uses_volcano_presets(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    env_var: str,
    base_url: str,
    primary_model: str,
) -> None:
    monkeypatch.setenv(env_var, "test-key")
    rc_config = SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            base_url="",
            api_key="",
            api_key_env=env_var,
            primary_model=primary_model,
            fallback_models=(),
        ),
    )

    client = LLMClient.from_rc_config(rc_config)

    assert client.config.base_url == base_url
    assert client.config.api_key == "test-key"
    assert client.config.primary_model == primary_model


@pytest.mark.parametrize(
    ("choice", "provider", "api_key_env"),
    [
        ("5", "volcengine", "VOLCENGINE_API_KEY"),
        ("6", "volcengine-coding-plan", "VOLCENGINE_API_KEY"),
        ("7", "byteplus", "BYTEPLUS_API_KEY"),
        ("8", "byteplus-coding-plan", "BYTEPLUS_API_KEY"),
    ],
)
def test_cli_provider_choices_include_volcano_variants(
    choice: str, provider: str, api_key_env: str
) -> None:
    from researchclaw.cli import _PROVIDER_CHOICES

    assert _PROVIDER_CHOICES[choice] == (provider, api_key_env)


@pytest.mark.parametrize(
    ("provider", "base_url", "primary_model", "fallback_len"),
    [
        (
            "volcengine",
            "https://ark.cn-beijing.volces.com/api/v3",
            "doubao-seed-2-0-pro-260215",
            6,
        ),
        (
            "volcengine-coding-plan",
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "doubao-seed-2.0-code",
            7,
        ),
        (
            "byteplus",
            "https://ark.ap-southeast.bytepluses.com/api/v3",
            "seed-2-0-pro-260328",
            4,
        ),
        (
            "byteplus-coding-plan",
            "https://ark.ap-southeast.bytepluses.com/api/coding/v3",
            "dola-seed-2.0-pro",
            5,
        ),
    ],
)
def test_cli_provider_models_include_volcano_variants(
    provider: str, base_url: str, primary_model: str, fallback_len: int
) -> None:
    from researchclaw.cli import _PROVIDER_MODELS, _PROVIDER_URLS

    assert _PROVIDER_URLS[provider] == base_url
    primary, fallbacks = _PROVIDER_MODELS[provider]
    assert primary == primary_model
    assert len(fallbacks) == fallback_len


def test_quickstart_wizard_applies_volcengine_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(
        [
            "volcano-project",
            "Build an automated literature review agent",
            "ml",
            "",
            "y",
            "",
            "6",
            "",
            "y",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    config = QuickStartWizard().run_interactive()

    assert config["llm"]["provider"] == "volcengine"
    assert config["llm"]["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert config["llm"]["api_key_env"] == "VOLCENGINE_API_KEY"
    assert config["llm"]["primary_model"] == "doubao-seed-2-0-pro-260215"
    assert config["llm"]["fallback_models"][0] == "doubao-seed-2-0-lite-260215"
