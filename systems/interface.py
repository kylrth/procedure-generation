from abc import ABC, abstractmethod
from typing import List


class SystemInterface(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    @abstractmethod
    def generate(self, title: str) -> List[str]:
        """The output is a list of generations, in case of n > 1."""
        pass
