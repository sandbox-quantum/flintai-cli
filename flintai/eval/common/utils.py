import logging
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from dataclasses_json import config


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with a standard format."""
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )


def generate_id() -> str:
    return str(uuid4())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


datetime_config = config(
    encoder=datetime.isoformat,
    decoder=datetime.fromisoformat,
)


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_env(value: str | None) -> str | None:
    """Substitute ``${VAR_NAME}`` references with environment variable values.

    Plain strings without ``${…}`` are returned unchanged.
    Raises ``ValueError`` if a referenced variable is not set.
    """
    if value is None:
        return None
    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        resolved = os.environ.get(var)
        if resolved is None:
            raise ValueError(
                f"Environment variable {var!r} is not set"
            )
        return resolved
    return _ENV_PATTERN.sub(_replace, value)


def resolve_env_dict(
    d: dict[str, str],
) -> dict[str, str]:
    """Apply :func:`resolve_env` to every value in *d*."""
    return {k: resolve_env(v) for k, v in d.items()}


def strip_nulls(obj: object) -> object:
    """Recursively remove None-valued keys from dicts."""
    if isinstance(obj, dict):
        return {
            k: strip_nulls(v)
            for k, v in obj.items()
            if v is not None
        }
    if isinstance(obj, list):
        return [strip_nulls(item) for item in obj]
    return obj
