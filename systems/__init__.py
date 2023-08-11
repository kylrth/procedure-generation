"""This package provides the systems used in the paper. By "systems" we mean general approaches, not
specific LMs."""


from .few_shot import FewShot
from .interface import System
from .model import Model
from .rag import RAG


__all__ = [
    "System",
    "Model",
    # systems
    "FewShot",
    "RAG",
]
