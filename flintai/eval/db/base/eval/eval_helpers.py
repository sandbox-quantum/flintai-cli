from __future__ import annotations

from flintai.eval.core.eval.evaluation import Evaluation
from flintai.eval.core.models.generator_model import get_generator_model
from flintai.eval.db.base.detectors.detector_repository import DetectorRepository
from flintai.eval.db.base.eval.eval_types import DbEvaluation, EvaluationType
from flintai.eval.db.base.message.message_collection_repository import (
    MessageCollectionRepository,
)


def create_evaluation(
    db_evaluation: DbEvaluation,
    message_collection_repo: MessageCollectionRepository | None = None,
    detector_repo: DetectorRepository | None = None,
) -> Evaluation:
    """Create an Evaluation instance from a DbEvaluation."""
    if db_evaluation.type == EvaluationType.MESSAGE_COLLECTION:
        if message_collection_repo is None:
            raise ValueError(
                "message_collection_repo is required for "
                "MESSAGE_COLLECTION evaluations"
            )
        if db_evaluation.message_collection_id is None:
            raise ValueError(
                "message_collection_id must be set on "
                "the DbEvaluation"
            )
        if detector_repo is None:
            raise ValueError(
                "detector_repo is required for "
                "MESSAGE_COLLECTION evaluations"
            )
        if db_evaluation.detector_id is None:
            raise ValueError(
                "detector_id must be set on "
                "the DbEvaluation"
            )
        from flintai.eval.core.eval.evaluation_message_list import (
            MessageListEvaluation,
        )

        message_collection = (
            message_collection_repo.get_message_collection(
                db_evaluation.message_collection_id
            )
        )
        messages = message_collection.load()
        detector = detector_repo.get_detector(
            db_evaluation.detector_id,
        )
        return MessageListEvaluation(
            messages=messages,
            detector=detector,
            num_prompts=db_evaluation.num_prompts,
        )

    elif db_evaluation.type == EvaluationType.GARAK_PROBE:
        from flintai.eval.core.eval.evaluation_garak_probe import (
            GarakProbeEvaluation,
        )

        if db_evaluation.probe_name is None:
            raise ValueError(
                "probe_name must be set on "
                "the DbEvaluation"
            )
        return GarakProbeEvaluation(
            probe_name=db_evaluation.probe_name,
        )

    elif db_evaluation.type == EvaluationType.GARAK_MODULE:
        from flintai.eval.core.eval.evaluation_garak_module import (
            GarakModuleEvaluation,
        )

        if db_evaluation.module_name is None:
            raise ValueError(
                "module_name must be set on "
                "the DbEvaluation"
            )
        return GarakModuleEvaluation(
            module_name=db_evaluation.module_name,
            probe_names=db_evaluation.probe_names,
        )

    elif db_evaluation.type == EvaluationType.METRIC_TOXICITY:
        from flintai.eval.core.eval.metric_toxicity import (
            ToxicityMetricEvaluation,
        )

        return ToxicityMetricEvaluation()

    elif db_evaluation.type == EvaluationType.METRIC_CONCISENESS:
        from flintai.eval.core.eval.metric_conciseness import (
            ConcisenessMetricEvaluation,
        )

        return ConcisenessMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif (
        db_evaluation.type
        == EvaluationType.METRIC_FACTUAL_ACCURACY
    ):
        from flintai.eval.core.eval.metric_factual_accuracy import (
            FactualAccuracyMetricEvaluation,
        )

        return FactualAccuracyMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif (
        db_evaluation.type
        == EvaluationType.METRIC_INSTRUCTION_ADHERENCE
    ):
        from flintai.eval.core.eval.metric_instruction_adherence import (
            InstructionAdherenceMetricEvaluation,
        )

        return InstructionAdherenceMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif (
        db_evaluation.type
        == EvaluationType.METRIC_TONE
    ):
        from flintai.eval.core.eval.metric_tone import (
            ToneMetricEvaluation,
        )

        return ToneMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif (
        db_evaluation.type
        == EvaluationType.ADVERSARIAL_PROBE
    ):
        from flintai.eval.core.eval.evaluation_adversarial import (
            AdversarialEvaluation,
        )

        goals: list[str] = []
        if (
            db_evaluation.message_collection_id
            and message_collection_repo
        ):
            mc = message_collection_repo.get_message_collection(
                db_evaluation.message_collection_id,
            )
            messages = mc.load()
            goals = [
                part.text
                for msg in messages
                for part in msg.content.parts
                if part.text is not None
            ]

        if db_evaluation.adversarial_goals:
            goals.extend(db_evaluation.adversarial_goals)

        if not goals:
            raise ValueError(
                "adversarial_goals or "
                "message_collection_id must be set on "
                "the DbEvaluation"
            )
        if not db_evaluation.attack_techniques:
            raise ValueError(
                "attack_techniques must be set on "
                "the DbEvaluation for adversarial probes"
            )
        if detector_repo is None:
            raise ValueError(
                "detector_repo is required for "
                "ADVERSARIAL_PROBE evaluations"
            )
        if not db_evaluation.detector_id:
            raise ValueError(
                "detector_id must be set on "
                "the DbEvaluation for adversarial probes"
            )
        db_detector = detector_repo.get(
            db_evaluation.detector_id,
        )
        return AdversarialEvaluation(
            goals=goals,
            attack_techniques=db_evaluation.attack_techniques,
            detector_prompt=db_detector.prompt,
            num_prompts=db_evaluation.num_prompts or 5,
            max_turns=db_evaluation.max_turns or 5,
            attacker_model=get_generator_model(),
        )

    elif (
        db_evaluation.type
        == EvaluationType.TOPIC_GUARD
    ):
        from flintai.eval.core.eval.evaluation_topic_guard import (
            TopicGuardEvaluation,
        )

        if (
            not db_evaluation.agent_objective
            and not db_evaluation.agent_instructions
        ):
            raise ValueError(
                "at least one of agent_objective or "
                "agent_instructions must be set on "
                "the DbEvaluation"
            )
        return TopicGuardEvaluation(
            agent_objective=db_evaluation.agent_objective,
            agent_instructions=db_evaluation.agent_instructions,
            num_prompts=db_evaluation.num_prompts or 5,
            max_turns=db_evaluation.max_turns or 5,
            attacker_model=get_generator_model(),
        )

    else:
        raise ValueError(
            f"unknown evaluation type: {db_evaluation.type}"
        )
