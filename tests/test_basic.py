import pairot


def test_package_has_version():
    assert pairot.__version__ is not None


def test_imports():
    from pairot.pl import distance, mapping, sankey
    from pairot.pp import preprocess_adatas
    from pairot.tl import DatasetMap

    assert preprocess_adatas is not None
    assert DatasetMap is not None
    assert distance is not None
    assert mapping is not None
    assert sankey is not None
