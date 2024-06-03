from abc import ABC, abstractmethod
from dataset import Procedure

class Heuristic(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    
    def evaluate(self, gold: Procedure, generated: list[str]):
        raise NotImplementedError

    
    async def aevaluate(self, gold: Procedure, generated: list[str]):
        raise NotImplementedError