import logging
from abc import ABC, abstractmethod


class System(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    @abstractmethod
    def generate(self, title: str, logger: logging.Logger | None) -> list[str]:
        """The output is a list of generations, in case of n > 1."""

    @abstractmethod
    async def agenerate(self, title: str, logger: logging.Logger | None) -> list[str]:
        """The output is a list of generations, in case of n > 1."""
