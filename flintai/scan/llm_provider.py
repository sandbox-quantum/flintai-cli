"""
llm_provider.py — LLM model factory for the agent scanner.

Single entry point:

  make_model(model_string)
      Returns an ADK-compatible LiteLlm model object for all providers.
      Used by both the agentic reasoner (ADK Runner) and single-pass
      completions (triage) via complete_text().

  complete_text(model, system_prompt, user_message, ...)
      Runs a single-pass LLM completion using generate_content_async
      on the model returned by make_model().

Configuration via AGENT_SCANNER_MODEL env var in 'provider:model' format
(e.g., 'google:gemini-2.5-flash', 'openai:gpt-5.4', 'anthropic:claude-sonnet-4-6').

API keys are read from standard env vars by the underlying frameworks:
GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import warnings
from typing import Any

from flintai.scan.schema import ADKModel

logger = logging.getLogger(__name__)

_REDACT_PATTERNS = re.compile(
    r"("
    r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*"
    r"|sk-[A-Za-z0-9]{10,}"
    r"|key-[A-Za-z0-9]{10,}"
    r"|pat-[A-Za-z0-9]{10,}"
    r"|AIza[A-Za-z0-9_\-]{35,}"
    r"|[A-Za-z0-9]{40,}"
    r")",
    re.IGNORECASE,
)


def _safe_error(exc: Exception) -> str:
    """Return a log-safe string with token-like patterns redacted."""
    return _REDACT_PATTERNS.sub("[REDACTED]", str(exc))


PROVIDER_GOOGLE = "google"
PROVIDER_LITELLM = "litellm"

DEFAULT_MODEL = "gemini-2.5-flash"

_PROVIDER_ALIASES = {"gemini": "google"}


def parse_model_string(model_string: str) -> tuple[str, str | None]:
    """Parse 'provider:model' -> (provider, model)."""
    if ":" in model_string:
        provider, model = model_string.split(":", 1)
    else:
        provider, model = model_string, None
    provider = provider.strip().lower()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    return provider, model.strip() if model else None


def _resolve_model_string(model_string: str | None = None) -> tuple[str, str]:
    """Resolve provider and model from argument or AGENT_SCANNER_MODEL env var."""
    ms = (model_string or os.getenv("AGENT_SCANNER_MODEL", "")).strip()
    if not ms:
        return PROVIDER_GOOGLE, DEFAULT_MODEL
    provider, model = parse_model_string(ms)
    return provider, model or DEFAULT_MODEL


def make_model(model_string: str | None = None, temperature: float = 0.0) -> ADKModel:
    """Return an ADK-compatible LiteLlm model for any provider.

    Always returns a LiteLlm object so that all callers (agentic
    reasoner, triage) use the same generate_content_async interface.
    """
    provider, model = _resolve_model_string(model_string)

    LiteLlm = _import_litellm()
    if provider == PROVIDER_GOOGLE:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = LiteLlm(model=f"gemini/{model}", temperature=temperature)
        for w in caught:
            logger.debug("%s", w.message)
        return result
    if provider == PROVIDER_LITELLM:
        return LiteLlm(model=model, temperature=temperature)
    return LiteLlm(model=f"{provider}/{model}", temperature=temperature)


def get_model_name(model_string: str | None = None) -> str:
    """Return the resolved model name for logging/metadata."""
    _, model = _resolve_model_string(model_string)
    return model


def _import_litellm() -> type[Any]:
    """Import and return the LiteLlm class from google-adk or a minimal shim."""
    try:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm
    except ImportError:
        pass
    try:
        from google.adk.models import LiteLlm  # type: ignore[no-redef]

        return LiteLlm
    except ImportError:
        pass
    try:
        import litellm as _litellm  # noqa: F401

        class _LiteLlmShim:
            def __init__(self, model: str, **kwargs):
                self.model = model
                self._kwargs = kwargs

            def __repr__(self) -> str:
                return f"LiteLlm(model={self.model!r})"

        return _LiteLlmShim
    except ImportError as e:
        raise ImportError(
            "google-adk or litellm is required for non-Google providers. "
            "Run: pip install google-adk litellm"
        ) from e


# ── Single-pass completion ───────────────────────────────────────────────────


def complete_text(
    model: ADKModel,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 8000,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str | None:
    """Run a single-pass LLM completion via model.generate_content_async."""
    from google.adk.models.llm_request import LlmRequest  # lazy: ADK optional
    from google.genai import types as genai_types  # lazy: ADK optional

    model_name = getattr(model, "model", str(model))

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_message)],
        )
    ]

    async def _run() -> str | None:
        request = LlmRequest(
            model=model_name,
            contents=contents,
            config=config,
        )
        text = ""
        async for resp in model.generate_content_async(request):
            if resp.content and resp.content.parts:
                for part in resp.content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
        return text.strip() or None

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _run()).result(timeout=120)
    except RuntimeError:
        pass

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.error("LLM call failed: %s", _safe_error(e))
        return None
