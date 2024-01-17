from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseMessage


@dataclass
class Result:
    # the query passed to the system
    query: str

    # the prompt that was sent to the LLM
    prompt: str | list[BaseMessage]

    # answer(s) to the query (usually completion(s) from the LLM)
    answers: list[str]

    # the model used to generate completions
    model: BaseLanguageModel

    # any documents retrieved during processing
    retrieved_docs: list[dict[str, str | float]] | None = None

    # those same documents, formatted as they appear in the prompt
    context: str | None = None


class System(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""

    @abstractmethod
    def generate(self, query: str) -> Result:
        """The output is a list of generations, in case of n > 1."""

    @abstractmethod
    async def agenerate(self, query: str) -> Result:
        """The output is a list of generations, in case of n > 1."""
