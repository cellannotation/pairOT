from importlib.metadata import version

from . import align, pl, pp

__all__ = ["pp", "align", "pl"]

__version__ = version("pairot")
