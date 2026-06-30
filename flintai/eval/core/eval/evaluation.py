from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from dataclasses_json import dataclass_json

from flintai.eval.common.schema import Content, Message, Role, Session
from flintai.eval.common.schema import PartType
from flintai.eval.common.schema import Session
from flintai.eval.core.models.model import Model


class EvaluationStatus(str, Enum):
    WAITING = "waiting"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


EvaluationObserver = Callable[["Evaluation"], None]


@dataclass_json
@dataclass
class EvaluationSummary:
    status: EvaluationStatus
    total_evaluations: int
    finished_evaluations: int
    error_evaluations: int
    max_score: float
    achieved_score: float
    error_messages: list[str] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        if self.status != EvaluationStatus.FINISHED:
            return None
        if self.max_score == 0:
            return None
        return self.achieved_score / self.max_score

    @property
    def progress(self) -> float:
        if self.total_evaluations == 0:
            return 100.0
        return (
            self.finished_evaluations + self.error_evaluations
        ) / self.total_evaluations


@dataclass_json
@dataclass
class EvaluationResult:
    """Result of a single leaf evaluation."""

    score: float
    status: EvaluationStatus
    error_message: str | None = None
    session: Session | None = None


@dataclass_json
@dataclass
class Evaluation(ABC):
    observers: list[EvaluationObserver] = field(
        default_factory=list,
        repr=False,
    )

    def __init__(self):
        self.observers = []

    def add_observer(self, observer: EvaluationObserver):
        self.observers.append(observer)

    def remove_observer(self, observer: EvaluationObserver):
        self.observers.remove(observer)

    def _notify_observers(self):
        for observer in self.observers:
            observer(self)

    @abstractmethod
    async def init(self):
        pass

    @abstractmethod
    def get_summary(self) -> EvaluationSummary:
        pass

    @abstractmethod
    async def run(self, model: Model, concurrency: int = 50):
        pass

    @abstractmethod
    def get_results(self) -> list[EvaluationResult]:
        pass
