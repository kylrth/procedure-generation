"""This package provides the systems used in the paper. By "systems" we mean general approaches, not
specific LMs."""


from .few_shot import FewShot, format_recipe
from .interface import SystemInterface
from .model import Model
from .zero_shot import ZeroShot

__all__ = [
    "format_recipe",
    "SystemInterface",
    "Model",
    # systems
    "ZeroShot",
    "FewShot",
]
