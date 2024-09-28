import logging
from abc import ABC, abstractmethod

from dataset import LinearProcedure


class Heuristic(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    @abstractmethod
    def evaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> int | float:
        pass

    @abstractmethod
    async def aevaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> int | float:
        pass
