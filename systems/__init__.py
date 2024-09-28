"""This package provides the systems used in the paper. By "systems" we mean general approaches, not
specific LMs."""

from .aag import AAG
from .few_shot import FewShot
from .interface import Response, System
from .rag import RAG
from .react import ReAct


__all__ = [
    "Response",
    "System",
    # systems
    "FewShot",
    "RAG",
    "ReAct",
    "AAG",
]
