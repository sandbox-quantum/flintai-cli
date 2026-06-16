import logging
from abc import abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from flintai.eval.common.schema import Session
from flintai.eval.core.models.model import Model
from flintai.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class SingleEvaluation(Evaluation):
    status: EvaluationStatus = EvaluationStatus.WAITING
    error_message: str | None = None
    score: float = 0.0
    session: Session | None = None

    def __init__(self):
        super().__init__()
        self.status = EvaluationStatus.WAITING
        self.error_message = None
        self.score = 0.0
        self.session = None

    @abstractmethod
    async def init(self):
        pass

    def get_summary(self) -> EvaluationSummary:
        return EvaluationSummary(
            status=self.status,
            total_evaluations=1,
            finished_evaluations=1 if self.status == EvaluationStatus.FINISHED else 0,
            error_evaluations=1 if self.status == EvaluationStatus.ERROR else 0,
            max_score=1.0,
            achieved_score=self.score if self.status == EvaluationStatus.FINISHED else 0.0,
            error_messages=[self.error_message] if self.error_message else [],
        )

    async def run(self, model: Model, concurrency: int = 50):
        await self.execute(model)

    async def execute(self, model: Model):
        if self.status == EvaluationStatus.ERROR:
            return
        self.status = EvaluationStatus.RUNNING
        name = type(self).__name__

        try:
            self.score = await self.execute_internal(model)
            self.status = EvaluationStatus.FINISHED
            logger.debug("%s finished: score=%.2f", name, self.score)
        except Exception as e:
            self.error_message = str(e)
            self.status = EvaluationStatus.ERROR
            logger.error("%s failed (%s: %s)", name, type(e).__name__, e)
        finally:
            self._notify_observers()

    def get_results(self) -> list[EvaluationResult]:
        return [EvaluationResult(
            score=self.score,
            status=self.status,
            error_message=self.error_message,
            session=self.session,
        )]

    @abstractmethod
    async def execute_internal(self, model: Model) -> float:
        pass
