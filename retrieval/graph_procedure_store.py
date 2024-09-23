from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import weaviate.classes.config as wc

from .store import Store


class Node[T]:
    data: T
    incoming: list["Edge"]
    outgoing: list["Edge"]


class Edge[T]:
    output: str
    to: Node[T] | None
    from_: Node[T] | None


class Input[T](Edge[T]):
    from_ = None


class Output[T](Edge[T]):
    to = None


class Graph[T]:
    inputs: list[Input[T]]
    outputs: list[Output[T]]


@dataclass
class Step:
    api: str
    desc: str
    args: list[str]


class Procedure(Graph[Step], ABC):
    """A graph of steps that has a particular formatting method."""

    @abstractmethod
    def __str__(self) -> str:
        """Procedure types must implement a formatting method."""


class Recipe(Procedure):
    def __str__(self) -> str:
        raise NotImplementedError


class LangChainProcedure(Procedure):
    def __str__(self) -> str:
        raise NotImplementedError


class MathSolution(Procedure):
    def __str__(self) -> str:
        raise NotImplementedError


class GraphProcedureStore(Store):
    NotImplemented
