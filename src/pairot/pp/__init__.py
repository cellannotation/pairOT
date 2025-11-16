try:
    import pyximport

    pyximport.install(language_level="3")

    from ._auroc import calc_auroc, csr_to_csc
except ModuleNotFoundError:
    pass

from importlib.metadata import version

from ._preprocessing import preprocess_adatas
from ._selection import (
    FILTERED_GENES,
    OFFICIAL_GENES,
    filter_genes_ava,
    filter_genes_ova,
    select_genes,
)
from ._testing import rank_genes_limma
from ._utils import downsample_indices

__all__ = [
    "preprocess_adatas",
    "downsample_indices",
    "rank_genes_limma",
    "select_genes",
    "filter_genes_ova",
    "filter_genes_ava",
    "OFFICIAL_GENES",
    "FILTERED_GENES",
]

__version__ = version("pairot")
