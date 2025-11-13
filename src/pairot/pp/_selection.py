import warnings
from os.path import dirname, join

import numpy as np
import pandas as pd

OFFICIAL_GENES: pd.DataFrame = pd.read_csv(join(dirname(__file__), "resources/official-genes.csv"))
OFFICIAL_GENES.__doc__ = """DataFrame containing official gene names from genenames.org."""

FILTERED_GENES: pd.DataFrame = pd.read_csv(join(dirname(__file__), "resources/filtered-genes.csv"))
OFFICIAL_GENES.__doc__ = """DataFrame containing uninformative genes to filter out, e.g., mitochondrial, ribosomal, IncRNA, TCR and BCR genes."""


def _calc_scaled_jaccard(markers1: dict[str, set], markers2: dict[str, set]) -> pd.DataFrame:
    """Calculate the soft overlap index between the values of two dictionaries."""
    soft_overlap = pd.DataFrame(
        np.zeros((len(markers1), len(markers2))),
        index=list(markers1.keys()),
        columns=list(markers2.keys()),
    )
    for c1, marker_genes1 in markers1.items():
        for c2, marker_genes2 in markers2.items():
            try:
                # scale factor to account for different number of marker genes
                scale_factor = len(marker_genes2) / len(marker_genes1)
                intersection = len(marker_genes1.intersection(marker_genes2))
                union = len(marker_genes1.union(marker_genes2))
                soft_overlap.loc[c1, c2] = (intersection / union) * scale_factor
            except ZeroDivisionError:
                soft_overlap.loc[c1, c2] = 0.0
                warnings.warn(f"No marker genes provided for {c1} and {c2}. Setting overlap to 0.", stacklevel=2)

    return soft_overlap


def _filter_de_genes(de_res: dict[str, pd.DataFrame]):
    de_res_return = {}
    # subset DE results to relevant genes
    genes_to_filter = FILTERED_GENES["feature_name"].tolist()
    official_gene_names = OFFICIAL_GENES["feature_name"].tolist()
    for ct, de_df in de_res.items():
        # subset only to genes whose symbol is present on genenames.org
        de_df = de_df[de_df.index.isin(official_gene_names)]
        # filter out uninformative genes
        de_df = de_df[~de_df.index.isin(genes_to_filter)]
        de_res_return[ct] = de_df.copy()

    return de_res_return


def sort_and_filter_de_genes_ova(
    de_res: dict[str, pd.DataFrame],
    logfc_threshold: float = 1.0,
    aucroc_threshold: float = 0.6,
    adj_pval_threshold: float = 0.05,
    gene_filtering: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Sort and filter the differentially expressed genes for the OVA (one vs. all) setting.

    Parameters
    ----------
    de_res
        OVA (one vs. all) DE results from :func:`pairot.pp.calc_pseudobulk_stats`.
    logfc_threshold
        Minimum logFC threshold to consider a gene as differentially expressed. Genes with a smaller logFC will be filtered out.
    aucroc_threshold
        Minimum AUROC threshold. Genes with a smaller AUROC will be filtered out.
    adj_pval_threshold
        Maximum adjusted p-value threshold to consider a gene as differentially expressed. Genes with a larger adjusted p-value will be filtered out.
    gene_filtering
        If true, remove uninformative genes, e.g., mitochondrial, ribosomal, IncRNA, TCR and BCR genes.

    Returns
    -------
    top_de_genes_ova
        Dictionary containing the filtered and sorted (by logFC) DE results for each cluster.

    Examples
    --------
    >>> from pairot.pp import calc_pseudobulk_stats, sort_and_filter_de_genes_ova
    >>> de_res_ova, de_res_ava = calc_pseudobulk_stats(
    >>>     adata,
    >>>     cluster_label="cell_type_col",
    >>>     sample_label="sample_col",
    >>> )
    >>> de_res_ova_sorted_and_filtered = sort_and_filter_de_genes_ova(
    >>>     de_res_ova,
    >>>     logfc_threshold=1.0,
    >>>     aucroc_threshold=0.6,
    >>>     adj_pval_threshold=0.05,
    >>>     gene_filtering=True,
    >>> )
    >>> de_res_ova_sorted_and_filtered
    """
    # Subset DE results to relevant genes
    if gene_filtering:
        de_res = _filter_de_genes(de_res)
    # Sort and filter DE genes
    top_de_genes_ova = {}
    for ct, de_df in de_res.items():
        top_de_genes_ova[ct] = (
            de_df.sort_values("logFC", ascending=False)
            .query(f"`logFC` >= {logfc_threshold}")
            .query(f"`adj.P.Val` <= {adj_pval_threshold}")
            .query(f"`auroc` >= {aucroc_threshold}")
            .copy()
        )

    return top_de_genes_ova


def sort_and_filter_de_genes_ava(
    de_res: dict[str, dict[str, pd.DataFrame]],
    logfc_threshold: float = 1.0,
    aucroc_threshold: float = 0.6,
    adj_pval_threshold: float = 0.05,
    gene_filtering: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Sort and filter the differentially expressed genes for the AVA (all vs. all) setting.

    Parameters
    ----------
    de_res
        AVA (all vs. all) DE results from :func:`pairot.pp.calc_pseudobulk_stats`.
    logfc_threshold
        Minimum logFC threshold to consider a gene as differentially expressed. Genes with a smaller logFC will be filtered out.
    aucroc_threshold
        Minimum AUROC threshold. Genes with a smaller AUROC will be filtered out.
    adj_pval_threshold
        Maximum adjusted p-value threshold to consider a gene as differentially expressed. Genes with a larger adjusted p-value will be filtered out.
    gene_filtering
        If true, remove uninformative genes, e.g., mitochondrial, ribosomal, IncRNA, TCR and BCR genes.

    Returns
    -------
    top_de_genes_ava
        Dictionary containing the filtered and sorted (by logFC) DE results for each cluster pair.

    Examples
    --------
    >>> from pairot.pp import calc_pseudobulk_stats, sort_and_filter_de_genes_ava
    >>> de_res_ova, de_res_ava = calc_pseudobulk_stats(
    >>>     adata,
    >>>     cluster_label="cell_type_col",
    >>>     sample_label="sample_col",
    >>> )
    >>> de_res_ava_sorted_and_filtered = sort_and_filter_de_genes_ava(
    >>>     de_res_ava,
    >>>     logfc_threshold=1.0,
    >>>     aucroc_threshold=0.6,
    >>>     adj_pval_threshold=0.05,
    >>>     gene_filtering=True,
    >>> )
    >>> de_res_ava_sorted_and_filtered
    """
    clusters = sorted(de_res.keys())
    top_de_genes_ava = {ct: {} for ct in clusters}
    for ct1 in clusters:
        res = de_res[ct1]
        # Subset DE results to relevant genes
        if gene_filtering:
            res = _filter_de_genes(res)
        # Sort and filter DE genes
        for ct2 in clusters:
            if ct1 != ct2:
                top_de_genes_ava[ct1][ct2] = (
                    res[ct2]
                    .sort_values("logFC", ascending=False)
                    .query(f"`logFC` >= {logfc_threshold}")
                    .query(f"`adj.P.Val` <= {adj_pval_threshold}")
                    .query(f"`auroc` >= {aucroc_threshold}")
                    .copy()
                )

    return top_de_genes_ava


def select_and_combine_de_results(
    de_res_ova: dict[str, pd.DataFrame],
    de_res_ava: dict[str, dict[str, pd.DataFrame]],
    n_genes_ova: int | None = 10,
    n_genes_ava: int | None = 3,
    n_genes_max: int | None = None,
    overlap_threshold: float = 0.3,
    overlap_n_genes: int = 10,
    remove_duplicated_genes: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Select and combine DE results from OVA (one vs. all) and AVA (all vs. all) settings.

    de_res_ova
        OVA (one vs. all) DE results from :func:`pairot.pp.sort_and_filter_de_genes_ova`.
    de_res_ava
        AVA (all vs. all) DE results from :func:`pairot.pp.sort_and_filter_de_genes_ava`.
    n_genes_ova
        Number of top DE genes to select from the OVA results for each cluster.
    n_genes_ava
        Number of top DE genes to select from the AVA results for each cluster pair.
    n_genes_max
        Maximum number of DE genes to return for each cluster after combining OVA and AVA results. If None, all genes are returned.
    overlap_threshold
        Jaccard overlap threshold to determine if two clusters are similar enough to refine the DE results using AVA results.
        If the overlap of the top `overlap_n_genes` genes between the OVA results of two clusters is greater than this threshold, the top `n_genes_ava` AVA results will be added to refine the DE results.
    overlap_n_genes
        Number of genes to use for the overlap calculation between clusters.
    remove_duplicated_genes
        If true, remove duplicated genes after combining OVA and AVA results.

    Returns
    -------
    combined_de_results
        Dictionary containing the combined DE results for each cluster.

    Examples
    --------
    >>> from pairot.pp import (
    >>>     calc_pseudobulk_stats,
    >>>     sort_and_filter_de_genes_ova,
    >>>     sort_and_filter_de_genes_ava,
    >>>     select_and_combine_de_results
    >>> )
    >>> de_res_ova, de_res_ava = calc_pseudobulk_stats(
    >>>     adata,
    >>>     cluster_label="cell_type_col",
    >>>     sample_label="sample_col",
    >>> )
    >>> de_res_ova_sorted_and_filtered = sort_and_filter_de_genes_ova(
    >>>     de_res_ova,
    >>>     logfc_threshold=1.0,
    >>>     aucroc_threshold=0.6,
    >>>     adj_pval_threshold=0.05,
    >>>     gene_filtering=True,
    >>> )
    >>> de_res_ava_sorted_and_filtered = sort_and_filter_de_genes_ava(
    >>>     de_res_ava,
    >>>     logfc_threshold=1.0,
    >>>     aucroc_threshold=0.6,
    >>>     adj_pval_threshold=0.05,
    >>>     gene_filtering=True,
    >>> )
    >>> combined_de_results = select_and_combine_de_results(
    >>>     de_res_ova_sorted_and_filtered,
    >>>     de_res_ava_sorted_and_filtered,
    >>>     n_genes_ova=10,
    >>>     n_genes_ava=3,
    >>> )
    >>> combined_de_results
    """
    # Find cluster to refine
    de_overlap = _calc_scaled_jaccard(
        {ct: set(res.head(overlap_n_genes).index) for ct, res in de_res_ova.items()},
        {ct: set(res.head(overlap_n_genes).index) for ct, res in de_res_ova.items()},
    )
    clusters_to_refine = {}
    for col in de_overlap.columns:
        o = de_overlap.loc[col, :]
        clusters_to_refine[col] = o[o >= overlap_threshold].index.tolist()
        if col in clusters_to_refine[col]:
            clusters_to_refine[col].remove(col)
    # Combine DE results
    combined_de_results = {}
    for ct, de_df_ova in de_res_ova.items():
        res = [de_df_ova.head(n_genes_ova)[["logFC", "adj.P.Val", "auroc"]].assign(reference="all")]
        for ct_refined in clusters_to_refine[ct]:
            res.append(
                de_res_ava[ct][ct_refined]
                .head(n_genes_ava)[["logFC", "adj.P.Val", "auroc"]]
                .assign(reference=ct_refined)
            )
        res = pd.concat(res).sort_values("logFC", ascending=False)

        if remove_duplicated_genes:
            reference = res["reference"].groupby(res.index).apply(lambda x: list(x))
            res = res[~res.index.duplicated(keep="first")]
            res["reference"] = reference
        if n_genes_max is not None:
            res = res.head(n_genes_max)

        combined_de_results[ct] = res.copy()

    return combined_de_results
