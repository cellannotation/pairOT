from importlib.metadata import version

from ._plotting import distance, mapping, sankey

__all__ = [
    "mapping",
    "distance",
    "sankey",
]

__version__ = version("pairot")
