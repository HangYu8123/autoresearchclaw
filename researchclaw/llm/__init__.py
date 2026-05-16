"""LLM integration — OpenAI-compatible and ACP agent clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from researchclaw.config import RCConfig
    from researchclaw.llm.acp_client import ACPClient
    from researchclaw.llm.client import LLMClient

# Provider presets for common LLM services
PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
    },
    "kimi-anthropic": {
        "base_url": "https://api.kimi.com/coding/",
    },
    "novita": {
        "base_url": "https://api.novita.ai/openai",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
    },
    "volcengine": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    "volcengine-coding-plan": {
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
    },
    "byteplus": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
    },
    "byteplus-coding-plan": {
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/coding/v3",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "openai-compatible": {
        "base_url": None,  # Use user-provided base_url
    },
}


def create_llm_client(config: RCConfig) -> LLMClient | ACPClient:
    """Factory: return the right LLM client based on ``config.llm.provider``.

    - ``"acp"`` → :class:`ACPClient` (spawns an ACP-compatible agent)
    - ``"anthropic"`` → :class:`LLMClient` with Anthropic Messages API adapter
    - ``"kimi-anthropic"`` → :class:`LLMClient` with Kimi Coding Anthropic adapter
    - ``"openrouter"`` → :class:`LLMClient` with OpenRouter base URL
    - ``"openai"`` → :class:`LLMClient` with OpenAI base URL
    - ``"deepseek"`` → :class:`LLMClient` with DeepSeek base URL
    - ``"novita"`` → :class:`LLMClient` with Novita AI base URL
    - ``"minimax"`` → :class:`LLMClient` with MiniMax base URL
    - ``"volcengine"`` → :class:`LLMClient` with Volcengine ARK base URL
    - ``"volcengine-coding-plan"`` → :class:`LLMClient` with Volcengine
      Coding Plan base URL
    - ``"byteplus"`` → :class:`LLMClient` with BytePlus ModelArk base URL
    - ``"byteplus-coding-plan"`` → :class:`LLMClient` with BytePlus
      Coding Plan base URL
    - ``"gemini"`` → :class:`LLMClient` with Gemini Native Adapter
    - ``"openai-compatible"`` (default) → :class:`LLMClient` with custom base_url

    OpenRouter is fully compatible with the OpenAI API format, making it
    a drop-in replacement with access to 200+ models from Anthropic, Google,
    Meta, Mistral, and more. See: https://openrouter.ai/models
    """
    if config.llm.provider == "acp":
        from researchclaw.llm.acp_client import ACPClient as _ACP
        return _ACP.from_rc_config(config)

    from researchclaw.llm.client import LLMClient as _LLM

    # Use from_rc_config to properly initialize adapters (e.g., Anthropic)
    return _LLM.from_rc_config(config)
