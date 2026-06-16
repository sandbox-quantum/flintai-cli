from __future__ import annotations

from flintai.eval.core.detectors.detector import Detector
from flintai.eval.db.base.detectors.detector_types import DbDetector, DetectorType


def create_detector(
    db_detector: DbDetector,
) -> Detector:
    """Create a Detector instance from a DbDetector."""
    if db_detector.type == DetectorType.GARAK:
        if not db_detector.detector_name:
            raise ValueError(
                "detector_name is required for "
                "GARAK detectors"
            )
        from flintai.eval.core.detectors.detector_garak import (
            GarakDetector,
        )

        return GarakDetector(db_detector.detector_name)

    elif db_detector.type == DetectorType.MODEL:
        from flintai.eval.core.detectors.detector_model import (
            ModelDetector,
        )
        from flintai.eval.core.models.generator_model import (
            get_generator_model,
        )

        model = get_generator_model()
        kwargs = {}
        if db_detector.prompt is not None:
            kwargs["prompt"] = db_detector.prompt
        return ModelDetector(model=model, **kwargs)

    elif db_detector.type == DetectorType.PII:
        from flintai.eval.core.detectors.detector_pii import (
            PIIDetector,
        )

        return PIIDetector()

    elif db_detector.type == DetectorType.SECRET:
        from flintai.eval.core.detectors.detector_secret import (
            SecretDetector,
        )

        return SecretDetector()

    elif db_detector.type == DetectorType.TOPIC_GUARD:
        from flintai.eval.core.detectors.detector_topic_guard import (
            TopicGuardDetector,
        )
        from flintai.eval.core.models.generator_model import (
            get_generator_model,
        )

        if (
            not db_detector.agent_objective
            and not db_detector.agent_instructions
        ):
            raise ValueError(
                "at least one of agent_objective or "
                "agent_instructions is required for "
                "TOPIC_GUARD detectors"
            )
        model = get_generator_model()
        return TopicGuardDetector(
            model=model,
            agent_objective=db_detector.agent_objective,
            agent_instructions=db_detector.agent_instructions,
        )

    else:
        raise ValueError(
            f"unknown detector type: {db_detector.type}"
        )
