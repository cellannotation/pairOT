import pairot


def test_package_has_version():
    assert pairot.__version__ is not None


def test_imports():
    from pairot.align import DatasetMapping
    from pairot.pl import plot_cluster_distance, plot_cluster_mapping, plot_sankey
    from pairot.pp import preprocess_adatas

    assert preprocess_adatas is not None
    assert DatasetMapping is not None
    assert plot_cluster_distance is not None
    assert plot_cluster_mapping is not None
    assert plot_sankey is not None
