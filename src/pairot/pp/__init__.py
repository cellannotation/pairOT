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
    select_and_combine_de_results,
    sort_and_filter_de_genes_ava,
    sort_and_filter_de_genes_ova,
)
from ._testing import calc_pseudobulk_stats
from ._utils import downsample_indices

__all__ = [
    "preprocess_adatas",
    "downsample_indices",
    "calc_pseudobulk_stats",
    "select_and_combine_de_results",
    "sort_and_filter_de_genes_ova",
    "sort_and_filter_de_genes_ava",
    "OFFICIAL_GENES",
    "FILTERED_GENES",
]

__version__ = version("pairot")
