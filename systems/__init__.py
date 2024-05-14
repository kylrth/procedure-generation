"""This package provides the systems used in the paper. By "systems" we mean general approaches, not
specific LMs."""

from .aag import AAG
from .few_shot import FewShot
from .interface import Response, System
from .model import Model
from .rag import RAG


__all__ = [
    "Response",
    "System",
    "Model",
    # systems
    "FewShot",
    "RAG",
    "AAG",
]
