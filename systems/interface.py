from abc import ABC, abstractmethod
import logging
from typing import List, Optional


class SystemInterface(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    @abstractmethod
    def generate(self, title: str, logger: Optional[logging.Logger]) -> List[str]:
        """The output is a list of generations, in case of n > 1."""
        pass

    @abstractmethod
    async def agenerate(self, title: str, logger: Optional[logging.Logger]) -> List[str]:
        """The output is a list of generations, in case of n > 1."""
        pass
