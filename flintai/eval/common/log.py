import logging
import os
import re


_REDACT_PATTERNS = re.compile(
    r"("
    r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*"
    r"|sk-[A-Za-z0-9]{10,}"
    r"|key-[A-Za-z0-9]{10,}"
    r"|pat-[A-Za-z0-9]{10,}"
    r"|AIza[A-Za-z0-9_\-]{35,}"
    r"|xox[bspra]-[A-Za-z0-9\-]+"
    r"|ghp_[A-Za-z0-9]{36,}"
    r"|-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"
    r"[\s\S]*?-----END"
    r")",
    re.IGNORECASE,
)

_REPLACEMENT = "[REDACTED]"


class RedactingFilter(logging.Filter):
    """Scrub secrets from log records before they reach handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _REDACT_PATTERNS.sub(
                _REPLACEMENT, record.msg,
            )
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    _REDACT_PATTERNS.sub(_REPLACEMENT, a)
                    if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: _REDACT_PATTERNS.sub(_REPLACEMENT, v)
                    if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


_NOISY_LOGGERS = [
    "anthropic",
    "google.genai",
    "google.adk",
    "google_adk",
    "google.auth",
    "google.api_core",
    "google_genai.models",
    "httpcore",
    "httpx",
    "urllib3",
    "opentelemetry",
    "litellm",
    "LiteLLM",
    "openai",
    "asyncio",
]


def silence_noisy_loggers() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.captureWarnings(True)


LOG_FORMAT = (
    "%(asctime)s %(levelname)s "
    "%(name)s [%(filename)s:%(lineno)d]: %(message)s"
)


def setup_file_logging(log_path: str) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.addFilter(RedactingFilter())
    root.addHandler(file_handler)

    os.environ["LITELLM_LOG"] = "CRITICAL"
    try:
        import litellm
        litellm.suppress_debug_info = True
    except ImportError:
        pass
    silence_noisy_loggers()
