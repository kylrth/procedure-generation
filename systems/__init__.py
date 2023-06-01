"""This package provides the systems used in the paper. By "systems" we mean general approaches, not
specific LMs."""


from .few_shot import FewShot
from .interface import SystemInterface
from .model import Model
from .zero_shot import ZeroShot

__all__ = [
    "SystemInterface",
    "Model",
    # systems
    "ZeroShot",
    "FewShot",
]
