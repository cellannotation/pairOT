from pairot.align import DatasetMapping


def test_dataset_mapping(adata_query_and_ref_preprocessed):
    adata_query, adata_ref = adata_query_and_ref_preprocessed
    cts_query = adata_query.obs["bulk_labels"].unique()
    cts_ref = adata_ref.obs["bulk_labels"].unique()

    dataset_map = DatasetMapping(adata_query, adata_ref)
    dataset_map.init_geom()
    dataset_map.init_problem()
    dataset_map.solve()

    mapping = dataset_map.compute_cluster_mapping(aggregation_method="mean")
    distance = dataset_map.compute_cluster_distances()

    assert mapping.shape[0] == len(cts_ref)
    assert mapping.shape[1] == len(cts_query)
    assert distance.shape[0] == len(cts_ref)
    assert distance.shape[1] == len(cts_query)
