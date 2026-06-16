import asyncio
import logging

from garak.attempt import Attempt, Conversation, Turn, Message as GarakMessage

from flintai.eval.common.schema import PartType, Role
from flintai.eval.core.models.model import ModelResponse
from flintai.eval.core.detectors.detector import Detector, DetectorResult

logger = logging.getLogger(__name__)


class GarakDetector(Detector):
    """A detector backed by a garak detector plugin."""

    def __init__(self, detector_name: str) -> None:
        self._detector_name = detector_name
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is None:
            logger.debug("Loading garak detector: %s", self._detector_name)
            from garak import _plugins
            self._detector = _plugins.load_plugin(
                self._detector_name
            )
        return self._detector

    async def detect(self, response: ModelResponse) -> DetectorResult:
        return await asyncio.to_thread(
            self._detect_sync, response,
        )

    def _detect_sync(self, response: ModelResponse) -> DetectorResult:
        conversation = _create_conversation(response)
        attempt = Attempt(prompt=conversation)
        results = self._ensure_loaded().detect(attempt)
        scores = [r for r in results if r is not None]
        if not scores:
            logger.debug("GarakDetector(%s): score=%.2f", self._detector_name, 1.0)
            return DetectorResult(score=1.0)
        max_hit = max(scores)
        logger.debug("GarakDetector(%s): score=%.2f", self._detector_name, 1.0 - max_hit)
        return DetectorResult(score=1.0 - max_hit)


def _extract_text(response: ModelResponse) -> str:
    if response.message is None:
        return ""
    parts = response.message.content.parts
    text_parts = [
        part.text
        for part in parts
        if part.part_type == PartType.TEXT and part.text is not None
    ]
    return "\n".join(text_parts)


def _map_role(role: Role) -> str:
    if role == Role.ASSISTANT:
        return "assistant"
    if role == Role.USER:
        return "user"
    if role == Role.SYSTEM:
        return "system"
    return str(role.value)


def _create_conversation(response: ModelResponse) -> Conversation:
    turns: list[Turn] = []
    if response.message is not None:
        role = _map_role(response.message.content.role)
        text = _extract_text(response)
        turns.append(
            Turn(
                role=role,
                content=GarakMessage(text=text),
            )
        )
    return Conversation(turns)
