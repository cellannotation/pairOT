import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def _get_expressed_genes_intersection(
    adata1: anndata.AnnData, adata2: anndata.AnnData, min_counts: float = 0.0
) -> list[str]:
    """
    Return the intersection of the expressed genes between adata1 and adata2.

    Parameters
    ----------
    adata1 : anndata.AnnData
        Query data.
    adata2 : anndata.AnnData
        Reference data.
    min_counts : float
        Minimum number of counts for a gene to be expressed.

    Returns
    -------
    List[str]
        List of genes that are expressed in both in adata1 and adata2.
    """
    genes1 = set(adata1.var.index.to_numpy()[np.array(adata1.X.sum(axis=0)).flatten() > min_counts])
    genes2 = set(adata2.var.index.to_numpy()[np.array(adata2.X.sum(axis=0)).flatten() > min_counts])
    return list(genes1 & genes2)


def _get_shared_highly_variable_genes(adata1: anndata.AnnData, adata2: anndata.AnnData, n_top_genes: int):
    adata_concat = anndata.concat([adata1, adata2], label="dataset", keys=["query", "ref"])
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    highly_variable = sc.pp.highly_variable_genes(
        adata_concat,
        n_top_genes=n_top_genes,
        batch_key="dataset",
        inplace=False,
    )

    return highly_variable["highly_variable"].to_numpy()


def downsample_indices(labels: np.ndarray, n_samples: int, random_state: int | None = 0) -> np.ndarray:
    """
    Downsample indices stratified by labels.

    Downsamples an array of labels and returns the indices to keep.
    For each unique label, if there are fewer than n_samples, keep all indices; otherwise, randomly sample n_samples indices.

    Parameters
    ----------
        labels: A numpy array or list of labels.
        n_samples: Minimum number of samples to uniformly sample per label.
        random_state: Seed for reproducibility (optional).

    Returns
    -------
        A numpy array of indices to keep after downsampling.
    """
    rng = np.random.default_rng(random_state)
    labels_series = pd.Series(labels)
    keep_indices = []

    for _, group in labels_series.groupby(labels_series):
        indices = group.index.to_numpy()
        if len(indices) < n_samples:
            # Keep all indices if the sampled count is lower than min_samples
            keep_indices.append(indices)
        else:
            sampled = rng.choice(indices, size=n_samples, replace=False)
            keep_indices.append(sampled)

    result = np.concatenate(keep_indices)
    return result
