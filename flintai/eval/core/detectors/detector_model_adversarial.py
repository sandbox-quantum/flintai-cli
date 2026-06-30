from dataclasses import dataclass
import logging
import json

from flintai.eval.core.detectors.detector import DetectorResult
from flintai.eval.core.models.model import (
    extract_text_from_message,
    extract_text_from_conversation,
)
from flintai.eval.core.detectors.detector_model import ModelDetector
from flintai.eval.core.models.model import Model, ModelResponse
from flintai.eval.common.schema import Content, Message, Role

logger = logging.getLogger(__name__)


@dataclass
class DetectorLLMResponse:
    score: float
    reason: str


class AdversarialModelDetector(ModelDetector):
    def __init__(
        self,
        model: Model,
        prompt: str,
        include_conversation: bool,
    ) -> None:
        super().__init__(model, prompt, include_conversation)

    async def detect(
        self, response: ModelResponse, *, goal: str, conversation: list[Message]
    ) -> DetectorResult:
        system_prompt = f"{self._prompt}\n\nThis is the attack goal: \n\n{goal}"
        conversation_text = extract_text_from_conversation(conversation)
        latest_response = extract_text_from_message(response.message)
        messages = [
            Message(
                content=Content.text(Role.SYSTEM, system_prompt),
            ),
        ]
        if self._include_conversation:
            messages.append(
                Message(
                    content=Content.text(
                        Role.USER,
                        f"Here is the conversation so far:\n\n{conversation_text}",
                    )
                ),
            )
        messages.append(
            Message(
                content=Content.text(Role.USER, latest_response),
            ),
        )
        detector_model_response = await self._model.generate(messages)

        detector_response_text = extract_text_from_message(
            detector_model_response.message
        )
        try:
            detector_response = DetectorLLMResponse(
                **json.loads(detector_response_text)
            )
        except Exception as e:
            raise ValueError(
                "AdversarialModelDetector LLM returned invalid data "
                "structure: %s ", e
            ) from e
        logger.debug(
            "AdversarialModelDetector: score=%.2f",
            detector_response.score,
        )
        return DetectorResult(score=detector_response.score)
