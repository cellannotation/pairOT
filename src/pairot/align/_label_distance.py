import anndata
import numpy as np
import pandas as pd

from pairot.pp import (
    select_and_combine_de_results,
    sort_and_filter_de_genes_ava,
    sort_and_filter_de_genes_ova,
)


def _check_dimensions(x1, x2, cell_type_labels_1, cell_type_labels_2):
    assert x1.shape[1] == x2.shape[1]
    assert x1.shape[0] == len(cell_type_labels_1)
    assert x2.shape[0] == len(cell_type_labels_2)


def _check_de_results(adata: anndata.AnnData, cell_type_column: str):
    ct_labels = adata.obs[cell_type_column].unique()
    # Check if all OVA DE results are present
    for ct in ct_labels:
        if ct not in adata.uns["de_res_ova"]:
            raise ValueError(f"OVA DE results for {ct} missing in adata object.")
    # Check if all AVA DE results are present
    for ct1 in ct_labels:
        for ct2 in ct_labels:
            if ct1 != ct2 and ct2 not in adata.uns["de_res_ava"][ct1]:
                raise ValueError(f"AVA DE results for {ct1} vs {ct2} missing in adata object.")


def _calc_rank_distance(
    degs_query: dict[str, pd.DataFrame],
    degs_ref: dict[str, pd.DataFrame],
    q_norm: float = 0.33,
) -> pd.DataFrame:
    def prepare_deg_res(df):
        return (
            df.reset_index()
            .rename(columns={"index": "gene", "ID": "gene"})
            .sort_values("logFC", ascending=False)
            .drop_duplicates("gene", keep="first")
            .reset_index(drop=True)
        )

    cts_query = sorted(degs_query.keys())
    cts_ref = sorted(degs_ref.keys())
    degs_query = {k: v.pipe(prepare_deg_res) for k, v in degs_query.items()}
    degs_ref = {k: v.pipe(prepare_deg_res) for k, v in degs_ref.items()}
    distance = pd.DataFrame(np.zeros((len(cts_query), len(cts_ref))), index=cts_query, columns=cts_ref)

    for ct_query in cts_query:
        for ct_ref in cts_ref:
            rank_distances = []
            for marker in degs_query[ct_query]["gene"].values:
                rank_query = float(degs_query[ct_query].query(f"gene == '{marker}'").index.item())
                try:
                    rank_ref = float(degs_ref[ct_ref].query(f"gene == '{marker}'").index.item())
                except ValueError:
                    rank_ref = 250.0  # set to arbitrary high value if gene not found
                rank_distances.append(np.abs(np.log1p(rank_query) - np.log1p(rank_ref)))
            if rank_distances:
                distance.loc[ct_query, ct_ref] = np.mean(rank_distances)
            else:
                distance.loc[ct_query, ct_ref] = 250.0

    return (distance / np.quantile(distance, q_norm)).clip(upper=1.0)


def _compute_label_distance_matrix(
    adata1: anndata.AnnData,
    adata2: anndata.AnnData,
    n_genes_ova: int = 10,
    n_genes_ava: int = 3,
    overlap_threshold: float = 0.3,
    overlap_n_genes: int = 10,
    adj_p_val_threshold: float = 0.05,
    auroc_threshold: float = 0.6,
    logfc_threshold: float = 1.0,
    gene_filtering: bool = True,
    return_selected_genes: bool = False,
    q_norm: float = 0.33,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict[str, dict[str, pd.DataFrame]]]]:
    """Compute cell-type label distance matrix based on the rank difference (sorted by logFC) of the top differentially expressed genes from the query dataset."""
    _check_de_results(adata1, "cell_type_author")
    _check_de_results(adata2, "cell_type_author")

    de_genes_adata1 = select_and_combine_de_results(
        sort_and_filter_de_genes_ova(
            adata1.uns["de_res_ova"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            logfc_threshold=logfc_threshold,
            gene_filtering=gene_filtering,
        ),
        sort_and_filter_de_genes_ava(
            adata1.uns["de_res_ava"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            logfc_threshold=logfc_threshold,
            gene_filtering=gene_filtering,
        ),
        n_genes_ova=n_genes_ova,
        n_genes_ava=n_genes_ava,
        overlap_threshold=overlap_threshold,
        overlap_n_genes=overlap_n_genes,
    )
    de_genes_adata2 = select_and_combine_de_results(
        sort_and_filter_de_genes_ova(
            adata2.uns["de_res_ova"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            logfc_threshold=logfc_threshold,
            gene_filtering=gene_filtering,
        ),
        sort_and_filter_de_genes_ava(
            adata2.uns["de_res_ava"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            logfc_threshold=logfc_threshold,
            gene_filtering=gene_filtering,
        ),
        n_genes_ova=None,
        n_genes_ava=None,
        overlap_threshold=overlap_threshold,
        overlap_n_genes=overlap_n_genes,
    )
    rank_distance = _calc_rank_distance(de_genes_adata1, de_genes_adata2, q_norm=q_norm)

    if not return_selected_genes:
        return rank_distance
    else:
        used_genes = {}
        for ct_query in de_genes_adata1.keys():
            used_genes[ct_query] = {}
            for ct_ref in de_genes_adata2.keys():
                genes_query = de_genes_adata1[ct_query].reset_index(names="gene").reset_index(names="rank")
                genes_ref = (
                    de_genes_adata2[ct_ref]
                    .reset_index(names="gene")
                    .query(f"`gene` in {de_genes_adata1[ct_query].index.tolist()}")
                    .reset_index(names="rank")
                    .set_index("gene")
                )
                used_genes[ct_query][ct_ref] = (
                    pd.merge(
                        genes_query,
                        genes_ref,
                        on="gene",
                        suffixes=("_query", "_ref"),
                        how="outer",
                    )
                    .sort_values("rank_query")
                    .set_index("gene")
                )

        return rank_distance, used_genes
