from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pp", "tl", "pl"]

__version__ = version("pairot")
