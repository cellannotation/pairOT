import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pairot.pp import (
    FILTERED_GENES,
    OFFICIAL_GENES,
    downsample_indices,
    filter_genes_ava,
    filter_genes_ova,
    preprocess_adatas,
    select_genes,
)


def _check_de_filtering(df, logfc_threshold, adj_pval_threshold, auroc_threshold):
    assert df["logFC"].min() >= logfc_threshold
    assert df["logFC"].is_monotonic_decreasing
    assert df["adj.P.Val"].max() <= adj_pval_threshold
    assert df["auroc"].min() >= auroc_threshold

    genes_to_filter = set(FILTERED_GENES["feature_name"])
    official_gene_names = set(OFFICIAL_GENES["feature_name"].tolist())

    assert len(set(df.index).intersection(genes_to_filter)) == 0
    assert len(set(df.index).intersection(official_gene_names)) == len(df.index)


@pytest.mark.parametrize("n_samples_auroc", [None, 10_000, 100])
@pytest.mark.parametrize("n_samples_hvg_selection", [None, 10_000, 100])
def test_preprocess_adatas(
    adata_query_and_ref: tuple[ad.AnnData, ad.AnnData], n_samples_auroc: int | None, n_samples_hvg_selection: int | None
):
    ct_col = "bulk_labels"
    sample_col = "sample_id"
    n_top_genes = 100

    adata_query, adata_ref = adata_query_and_ref
    adata_query, adata_ref = preprocess_adatas(
        adata_query,
        adata_ref,
        n_top_genes=n_top_genes,
        cell_type_column_adata1=ct_col,
        cell_type_column_adata2=ct_col,
        sample_column_adata1=sample_col,
        sample_column_adata2=sample_col,
        n_samples_auroc=n_samples_auroc,
        n_samples_hvg_selection=n_samples_hvg_selection,
    )

    assert "de_res_ova" in adata_query.uns
    assert "de_res_ova" in adata_ref.uns
    assert "de_res_ava" in adata_query.uns
    assert "de_res_ava" in adata_ref.uns

    for adata in [adata_query, adata_ref]:
        # check if all OVA DE results are present
        for v in adata.uns["de_res_ova"].values():
            assert "logFC" in v.columns
            assert "adj.P.Val" in v.columns
            assert "auroc" in v.columns

        # check if all AVA De results are present
        for v1 in adata.uns["de_res_ava"].values():
            for v2 in v1.values():
                assert "logFC" in v2.columns
                assert "adj.P.Val" in v2.columns
                assert "auroc" in v2.columns

    # check HVG selection
    assert adata_query.n_vars == adata_ref.n_vars == n_top_genes
    # check that gene spaces are the same between query and reference dataset
    pd.testing.assert_index_equal(adata_query.var.index, adata_ref.var.index)


@pytest.mark.parametrize("logfc", [1.0])
@pytest.mark.parametrize("adj_pval", [0.1])
@pytest.mark.parametrize("auroc", [0.5])
def test_filter_genes_ova(
    pseudobulk_results,
    logfc: float,
    adj_pval: float,
    auroc: float,
):
    de_res_ova, _ = pseudobulk_results
    de_res_ova = filter_genes_ova(
        de_res_ova,
        logfc_threshold=logfc,
        adj_pval_threshold=adj_pval,
        aucroc_threshold=auroc,
    )

    for df in de_res_ova.values():
        _check_de_filtering(
            df,
            logfc_threshold=logfc,
            adj_pval_threshold=adj_pval,
            auroc_threshold=auroc,
        )


@pytest.mark.parametrize("logfc", [1.0])
@pytest.mark.parametrize("adj_pval", [0.1])
@pytest.mark.parametrize("auroc", [0.5])
def test_filter_genes_ava(
    pseudobulk_results,
    logfc: float,
    adj_pval: float,
    auroc: float,
):
    _, de_res_ava = pseudobulk_results
    de_res_ava = filter_genes_ava(
        de_res_ava,
        logfc_threshold=logfc,
        adj_pval_threshold=adj_pval,
        aucroc_threshold=auroc,
    )

    for elems in de_res_ava.values():
        for df in elems.values():
            _check_de_filtering(
                df,
                logfc_threshold=logfc,
                adj_pval_threshold=adj_pval,
                auroc_threshold=auroc,
            )


def test_select_genes(pseudobulk_results):
    n_genes_ova = 10
    n_genes_ava = 3
    de_res_ova, de_res_ava = pseudobulk_results
    de_res_ova = filter_genes_ova(de_res_ova, logfc_threshold=0.0, aucroc_threshold=0.0, adj_pval_threshold=1.0)
    de_res_ava = filter_genes_ava(de_res_ava)
    de_res_combined = select_genes(de_res_ova, de_res_ava, n_genes_ova=n_genes_ova, n_genes_ava=n_genes_ava)

    for df in de_res_combined.values():
        assert "logFC" in df.columns
        assert "adj.P.Val" in df.columns
        assert "auroc" in df.columns
        assert df["logFC"].is_monotonic_decreasing
        assert df.index.is_unique
        ova_clusters_indicator = df["reference"].apply(lambda x: "all" in x)
        assert ova_clusters_indicator.sum() == n_genes_ova
        assert np.nan_to_num(df.loc[~ova_clusters_indicator, "reference"].value_counts().max()) <= n_genes_ava


def test_downsample_indices():
    labels = np.array(["A", "A", "A", "A", "B", "B", "C"])
    n_samples = 2
    result = downsample_indices(labels, n_samples, random_state=42)
    # Check the correct number of samples per label
    result_labels = labels[result]
    assert np.sum(result_labels == "A") == n_samples
    assert np.sum(result_labels == "B") == n_samples
    assert np.sum(result_labels == "C") == 1
    # Test reproducibility
    result1 = downsample_indices(labels, n_samples, random_state=42)
    result2 = downsample_indices(labels, n_samples, random_state=42)
    assert np.array_equal(result1, result2)
    # Test that all returned indices are valid
    assert np.all(result >= 0)
    assert np.all(result < len(labels))
