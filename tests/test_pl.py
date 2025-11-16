import numpy as np
import pandas as pd
import pytest

import pairot as pr

rng = np.random.default_rng(1)
mapping_df = pd.DataFrame(
    data=rng.uniform(0.0, 1.0, size=(10, 8)),
    columns=[f"ct_{i}" for i in range(8)],
    index=[f"ct_{i}" for i in range(10)],
)
distance_df = pd.DataFrame(
    data=rng.uniform(0.0, 2.0, size=(10, 8)),
    columns=[f"ct_{i}" for i in range(8)],
    index=[f"ct_{i}" for i in range(10)],
)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_mapping(backend):
    fig = pr.pl.mapping(mapping_df, backend=backend)
    assert fig is not None


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_distance(backend):
    fig = pr.pl.distance(distance_df, backend=backend)
    assert fig is not None


def test_plot_sankey():
    fig = pr.pl.sankey(mapping_df, distance_df)
    assert fig is not None
