from os.path import dirname, join

import anndata
import pandas as pd
import scanpy as sc

from pairot.pp._utils import _get_expressed_genes_intersection, _get_shared_highly_variable_genes


def preprocess_adatas(
    adata1: anndata.AnnData,
    adata2: anndata.AnnData,
    cell_type_column_adata1: str,
    cell_type_column_adata2: str,
    sample_column_adata1: str,
    sample_column_adata2: str,
    n_top_genes: int = 500,
    filter_genes: bool = False,
    n_samples_auroc: int = 10_000,
    n_samples_hvg_selection: int = 100_000,
) -> tuple[anndata.AnnData, anndata.AnnData]:
    """
    Function for pre-processing the input AnnData objects for usage with :class:`pairot.tl.DatasetMap`.

    Function applies the following preprocessing steps:
        1. Subset gene space to genes that are expressed in both datasets.
        2. Calculate differentially-expressed (DE) genes for each cluster.
        3. Subset to highly variable genes which are used to calculate the Spearman correlation between two cells.

    Parameters
    ----------
        adata1
            Query data.
        adata2
            Reference data.
        cell_type_column_adata1
            Name of the column in `adata.obs` that contains the cell type labels for adata1.
        cell_type_column_adata2
            Name of the column in `adata.obs` that contains the cell type labels for adata2.
        sample_column_adata1
            Name of the column in `adata.obs` that contains the sequencing sample ids/labels for adata1.
        sample_column_adata2
            Name of the column in `adata.obs` that contains the sequencing sample ids/labels for adata1.
        n_top_genes
            Number of highly variable genes to use to calculate the Spearman correlation between two cells.
        filter_genes
            Whether to remove uninformative genes.
            If true mitochondrial, ribosomal, IncRNA, TCR and BCR genes are removed.
        n_samples_auroc
            Maximum number of samples to use for AUROC calculation.
            If None, all samples are used.
            This can drastically reduce computation time for large datasets.
        n_samples_hvg_selection
            Number of samples to use for highly variable gene selection.
            If None, all samples are used.
            This can drastically reduce the memory usage for large datasets.

    Examples
    --------
        >>> import anndata as ad
        >>> from pairot.pp import preprocess_adatas
        >>>
        >>> adata_query = ad.read_h5ad("path/to/query_data.h5ad")
        >>> adata_ref = ad.read_h5ad("path/to/ref_data.h5ad")
        >>> adata_query, adata_ref = preprocess_adatas(
        >>>     adata1,
        >>>     adata2,
        >>>     cell_type_column_adata1="cell_type_col_adata1",
        >>>     cell_type_column_adata2="cell_type_col_adata2",
        >>>     sample_column_adata1="sample_id_col_adata1",
        >>>     sample_column_adata2="sample_id_col_adata2",
        >>> )

    """
    from pairot.pp._testing import rank_genes_limma

    adata1.X = adata1.X.astype("float32")
    adata2.X = adata2.X.astype("float32")

    adata1.obs["cell_type_author"] = adata1.obs[cell_type_column_adata1]
    adata1.obs["sample_id"] = adata1.obs[sample_column_adata1]
    adata2.obs["cell_type_author"] = adata2.obs[cell_type_column_adata2]
    adata2.obs["sample_id"] = adata2.obs[sample_column_adata2]
    print(f"adata1: {adata1.shape}")
    print(f"adata2: {adata2.shape}")
    # subset gene space only to filtered genes
    if filter_genes:
        print("Applying uninformative gene filtering...")
        genes_to_filter = pd.read_csv(join(dirname(__file__), "de_testing/resources/filtered-genes.csv"))[
            "feature_name"
        ].tolist()
        official_gene_names = pd.read_csv(join(dirname(__file__), "de_testing/resources/official-genes.csv"))[
            "feature_name"
        ].tolist()
        adata1 = adata1[:, adata1.var.index.isin(official_gene_names)]
        adata1 = adata1[:, ~adata1.var.index.isin(genes_to_filter)]
        adata2 = adata2[:, adata2.var.index.isin(official_gene_names)]
        adata2 = adata2[:, ~adata2.var.index.isin(genes_to_filter)]
        adata1 = adata1.copy()
        adata2 = adata2.copy()
        print(f"adata1: {adata1.shape}")
        print(f"adata2: {adata2.shape}")
    # subset gene space to genes that are expressed in both datasets
    print("Sub-setting gene space to genes that are expressed in both datasets...")
    intersection_genes = _get_expressed_genes_intersection(adata1, adata2, min_counts=10)
    adata1 = adata1[:, intersection_genes].copy()
    adata2 = adata2[:, intersection_genes].copy()
    print(f"adata1: {adata1.shape}")
    print(f"adata2: {adata2.shape}")
    # calculate DE genes
    print("Calculating differentially-expressed genes...")
    adata1.uns["de_res_ova"], adata1.uns["de_res_ava"] = rank_genes_limma(
        adata1, cluster_label="cell_type_author", sample_label="sample_id", n_samples_auroc=n_samples_auroc
    )
    adata2.uns["de_res_ova"], adata2.uns["de_res_ava"] = rank_genes_limma(
        adata2, cluster_label="cell_type_author", sample_label="sample_id", n_samples_auroc=n_samples_auroc
    )
    # subset to highly variable genes for Spearman correlation
    print("Sub-setting to highly variable genes...")
    if n_samples_hvg_selection is None:
        highly_variable = _get_shared_highly_variable_genes(adata1, adata2, n_top_genes)
    else:
        highly_variable = _get_shared_highly_variable_genes(
            sc.pp.subsample(adata1, n_obs=n_samples_hvg_selection, copy=True)
            if n_samples_hvg_selection < adata1.n_obs
            else adata1,
            sc.pp.subsample(adata2, n_obs=n_samples_hvg_selection, copy=True)
            if n_samples_hvg_selection < adata2.n_obs
            else adata2,
            n_top_genes,
        )
    adata1 = adata1[:, highly_variable].copy()
    adata2 = adata2[:, highly_variable].copy()
    print(f"adata1: {adata1.shape}")
    print(f"adata2: {adata2.shape}")

    return adata1, adata2
