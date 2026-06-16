import asyncio
import logging
import random
from typing import Any

from flintai.eval.common.schema import Content, Message, Role
from flintai.eval.core.models.model import Model, ModelContent, ModelResponse

logger = logging.getLogger(__name__)

_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    for attr in ("status_code", "status", "code"):
        status = getattr(exc, attr, None)
        if status in _TRANSIENT_STATUS_CODES:
            return True
    return False


class ExponentialRetryModel(Model):
    """Wraps a Model and retries on transient errors with
    exponential backoff."""

    def __init__(
        self,
        model: Model,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ):
        self._model = model
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def generate(
        self, contents: ModelContent, **kwargs: Any,
    ) -> ModelResponse:
        if isinstance(contents, str):
            messages = [Message(
                content=Content.text(Role.USER, contents),
            )]
        elif isinstance(contents, Message):
            messages = [contents]
        else:
            messages = contents
        return await self._generate(messages, **kwargs)

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._model.generate(messages, **kwargs)
            except Exception as exc:
                if not _is_transient(exc) or attempt == self._max_retries:
                    raise
                last_exc = exc
                base = self._base_delay * (2 ** attempt)
                jitter = random.uniform(0, base)
                delay = base + jitter
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.1fs (%s: %s)",
                    attempt + 1, self._max_retries, delay,
                    type(exc).__name__, exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # unreachable, but keeps type checker happy


def _fibonacci_delays(max_value: float):
    """Yield fibonacci-sequence delays capped at max_value."""
    a, b = 1.0, 1.0
    while True:
        yield min(a, max_value)
        a, b = b, a + b


class FibonacciRetryModel(Model):
    """Wraps a Model and retries on transient errors with
    fibonacci backoff and jitter, capped at a maximum delay."""

    def __init__(
        self,
        model: Model,
        max_retries: int = 5,
        max_delay: float = 70.0,
    ):
        self._model = model
        self._max_retries = max_retries
        self._max_delay = max_delay

    async def generate(
        self, contents: ModelContent, **kwargs: Any,
    ) -> ModelResponse:
        if isinstance(contents, str):
            messages = [Message(
                content=Content.text(Role.USER, contents),
            )]
        elif isinstance(contents, Message):
            messages = [contents]
        else:
            messages = contents
        return await self._generate(messages, **kwargs)

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        last_exc: Exception | None = None
        delays = _fibonacci_delays(self._max_delay)
        for attempt in range(self._max_retries + 1):
            try:
                return await self._model.generate(messages, **kwargs)
            except Exception as exc:
                if not _is_transient(exc) or attempt == self._max_retries:
                    raise
                last_exc = exc
                base = next(delays)
                delay = random.uniform(0, base)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.1fs (%s: %s)",
                    attempt + 1, self._max_retries, delay,
                    type(exc).__name__, exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # unreachable, but keeps type checker happy


RetryModel = ExponentialRetryModel
