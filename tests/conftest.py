import numpy as np
import pytest
from scanpy.datasets import pbmc68k_reduced

from pairot.pp import calc_pseudobulk_stats, preprocess_adatas


@pytest.fixture
def adata_query_and_ref():
    rng = np.random.default_rng(1)
    # convert back to raw counts
    pbmc_68k = pbmc68k_reduced().raw.to_adata()
    pbmc_68k.X = pbmc_68k.X.expm1().multiply(pbmc_68k.obs["n_counts"].to_numpy().reshape((-1, 1))).tocsr()
    # Filter out rare cell types for testing
    cts_to_keep = pbmc_68k.obs["bulk_labels"].value_counts().to_frame().query("count > 50").index.tolist()
    pbmc_68k = pbmc_68k[pbmc_68k.obs["bulk_labels"].isin(cts_to_keep)].copy()
    pbmc_68k.obs["sample_id"] = rng.choice([f"donor_{i}" for i in range(3)], size=pbmc_68k.n_obs)
    # split into query and reference for testing
    idxs_query, idxs_ref = np.array_split(rng.permutation(np.arange(pbmc_68k.n_obs)), 2)
    adata_query, adata_ref = pbmc_68k[idxs_query], pbmc_68k[idxs_ref]
    # only take a random subset as reference gene space to test gene space alignment
    adata_ref = adata_ref[:, rng.choice(adata_ref.var.index, size=700, replace=False)]

    return adata_query.copy(), adata_ref.copy()


@pytest.fixture
def adata_query_and_ref_preprocessed(adata_query_and_ref):
    adata_query, adata_ref = adata_query_and_ref

    return preprocess_adatas(
        adata_query,
        adata_ref,
        cell_type_column_adata1="bulk_labels",
        cell_type_column_adata2="bulk_labels",
        sample_column_adata1="sample_id",
        sample_column_adata2="sample_id",
        n_samples_auroc=100,
        n_top_genes=50,
    )


@pytest.fixture
def pseudobulk_results(adata_query_and_ref):
    return calc_pseudobulk_stats(
        adata_query_and_ref[0], cluster_label="bulk_labels", sample_label="sample_id", n_samples_auroc=100
    )
