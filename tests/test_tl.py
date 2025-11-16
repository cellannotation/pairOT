import pairot as pr


def test_dataset_mapping(adata_query_and_ref_preprocessed):
    adata_query, adata_ref = adata_query_and_ref_preprocessed
    cts_query = adata_query.obs["bulk_labels"].unique()
    cts_ref = adata_ref.obs["bulk_labels"].unique()

    dataset_map = pr.tl.DatasetMap(adata_query, adata_ref)
    dataset_map.init_geom()
    dataset_map.init_problem()
    dataset_map.solve()

    mapping = dataset_map.compute_mapping(aggregation_method="mean")
    distance = dataset_map.compute_distance()
    similar_cluster = dataset_map.select_similar_clusters(mapping, distance)

    assert mapping.shape[0] == len(cts_ref)
    assert mapping.shape[1] == len(cts_query)
    assert distance.shape[0] == len(cts_ref)
    assert distance.shape[1] == len(cts_query)
    assert isinstance(similar_cluster, dict)
    assert sorted(similar_cluster.keys()) == sorted(cts_query)
    for v in similar_cluster.values():
        assert isinstance(v, list)
