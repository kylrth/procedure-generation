import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils import log


@dataclass
class Response:
    # answer in the form of a list of steps
    answer: list[str]

    # the model used to generate completions
    model: str

    input_tokens: int = -1
    output_tokens: int = -1


class System(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    @abstractmethod
    def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        pass

    @abstractmethod
    async def agenerate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        pass

    _step_prefixes = re.compile(r"^\s*(?:\d+\.\s*|-)\s*(.*)$")

    @classmethod
    def parse_completion(cls, s: str) -> list[str]:
        """Utility method for parsing a list of steps from text."""
        lines = s.strip().split("\n")

        steps = []
        for line in lines:
            m = cls._step_prefixes.match(line)
            if m:
                steps.append(m.group(1))
            else:
                steps.append(line.strip())

        return steps
