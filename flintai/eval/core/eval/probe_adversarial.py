"""Backwards-compatibility re-exports.

The canonical location is now ``evaluation_adversarial``.
"""

from flintai.eval.core.eval.evaluation_adversarial import (  # noqa: F401
    AdversarialEvaluation,
    AdversarialEvaluation as AdversarialProbeEvaluation,
    AdversarialTurnEvaluation,
    AttackerJudgment,
    _extract_json,
    _extract_text,
    _parse_attacker_response,
)
