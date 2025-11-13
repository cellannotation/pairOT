from typing import Literal

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def _plot_heatmap(
    data: pd.DataFrame,
    colormap: str,
    width: int = None,
    height: int = None,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    **kwargs,
):
    if width is None:
        width = max(30 * data.shape[1], 500)
    if height is None:
        height = max(20 * data.shape[0] + 275, 500)

    if backend == "plotly":
        fig = px.imshow(data, text_auto=".2f", color_continuous_scale=colormap, **kwargs)
        fig.update_layout(autosize=False, width=width, height=height)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickangle=90)
        fig.update_yaxes(tickangle=0)
    elif backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            ax=ax,
            vmin=kwargs.pop("zmin", None),
            vmax=kwargs.pop("zmax", None),
            cmap=colormap,
            cbar=False,
            square=True,
            annot_kws={"size": 6},
        )
        ax.tick_params(axis="x", colors=(0.2, 0.2, 0.2, 1), length=0, labelsize=10)
        ax.tick_params(axis="y", colors=(0.2, 0.2, 0.2, 1), length=0, labelsize=10)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Backends supported: 'plotly', 'matplotlib'.")

    return fig


def plot_cluster_mapping(
    data: pd.DataFrame,
    width: int = None,
    height: int = None,
    zmin: float = None,
    zmax: float = None,
    colormap: str = "Greens",
    sort_by_score: bool = True,
    backend: Literal["plotly", "matplotlib"] = "plotly",
):
    """
    Plot cluster mappings from :func:`pp.algin.DatasetMap.compute_cluster_mapping`.

    Parameters
    ----------
        data
            pd.DataFrame containing the cluster mapping scores.
        width
            Width of the plot in pixels.
        height
            Height of the plot in pixels.
        zmin
            Minimum value for color scale.
        zmax
            Maximum value for color scale.
        colormap
            Colormap to use for the heatmap.
        sort_by_score
            Whether to sort the clusters by their maximum mapping score.
            If True, the strongest map for each cell type cluster appears on the diagonal of the mapping plot.
        backend
            Backend to use for plotting. Options are 'plotly' and 'matplotlib'.
            The plotly backend allows for interactive plots, while the matplotlib backend is suitable for static plots.

    Examples
    --------
        >>> from pairot.pl import plot_cluster_mapping
        >>>
        >>> # Get cluster mappings between query and reference dataset
        >>> mapping = dataset_map.compute_cluster_mapping()
        >>>
        >>> # Plot the cluster mapping heatmap
        >>> plot_cluster_mapping(data)
    """
    if sort_by_score:
        data = data.loc[
            data.max(axis=1).sort_values(ascending=False).index.tolist(),
            data.max().sort_values(ascending=False).index.tolist(),
        ]
    return _plot_heatmap(data, colormap, width, height, zmin=zmin, zmax=zmax, backend=backend)


def plot_cluster_distance(
    data: pd.DataFrame,
    width: int | None = None,
    height: int | None = None,
    backend: Literal["plotly", "matplotlib"] = "plotly",
):
    """
    Plot cluster distances from :func:`pp.algin.DatasetMap.compute_cluster_distance`.

    Parameters
    ----------
        data
            pd.DataFrame containing the cluster distances.
        width
            Width of the plot in pixels.
        height
            Height of the plot in pixels.
        backend
            Backend to use for plotting. Options are 'plotly' and 'matplotlib'.
            The plotly backend allows for interactive plots, while the matplotlib backend is suitable for static plots.

    Examples
    --------
        >>> from pairot.pl import plot_cluster_distance
        >>>
        >>> # Get cluster mappings between query and reference dataset
        >>> mapping = dataset_map.compute_cluster_mapping()
        >>> # Get cluster distances between query and reference dataset
        >>> distance = dataset_map.compute_cluster_distance()
        >>> distance = distance.loc[
        >>>         mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
        >>>         mapping.max().sort_values(ascending=False).index.tolist(),
        >>>     ]  # order cluster distance matrix the same way as similarity matrix
        >>> plot_cluster_distance(distance)

    """
    return _plot_heatmap(data, "RdYlGn_r", width, height, zmin=0.0, zmax=2.0, backend=backend)


def plot_sankey(
    cluster_mapping: pd.DataFrame,
    cluster_distance: pd.DataFrame,
    filter_threshold: float = 0.25,
    width: int = None,
    height: int = None,
):
    """
    Plot cluster mappings and distances from :class:`pp.algin.DatasetMap` as a Sankey diagram.

    Parameters
    ----------
        cluster_mapping
            pd.DataFrame containing the cluster mapping scores.
        cluster_distance
            pd.DataFrame containing the cluster distances.
        filter_threshold
            Minimum mapping score to include a link in the Sankey diagram.
        width
            Width of the plot in pixels.
        height
            Height of the plot in pixels.

    Examples
    --------
        >>> from pairot.pl import plot_sankey
        >>>
        >>> # Get cluster mappings between query and reference dataset
        >>> mapping = dataset_map.compute_cluster_mapping()
        >>> # Get cluster distances between query and reference dataset
        >>> distance = dataset_map.compute_cluster_distance()
        >>> plot_sankey(
        >>>     mapping,
        >>>     distance,
        >>>     filter_threshold=0.25,  # only show links with mapping score >= 0.25
        >>> )
    """

    def _filter_dict(unfiltered_dict, threshold=0.1):
        vals = pd.DataFrame(unfiltered_dict)
        vals_filtered = vals[vals.value >= threshold]
        return {col: vals_filtered[col].tolist() for col in vals_filtered.columns}

    norm = mpl.colors.Normalize(vmin=0.0, vmax=2.0)
    cmap = plt.get_cmap("RdYlGn_r")
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    nodes_query = cluster_mapping.index.tolist()
    nodes_ref = cluster_mapping.columns.tolist()
    nodes_combined = nodes_query + nodes_ref
    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat=".2f",
                node={
                    "line": {"color": "black", "width": 1.0},
                    "label": nodes_combined,
                    "color": [
                        px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                        for i in range(len(nodes_query + nodes_ref))
                    ],
                },
                link=_filter_dict(
                    {
                        "source": np.repeat(np.arange(len(nodes_query)), len(nodes_ref)).tolist(),
                        "target": np.tile(
                            np.arange(len(nodes_query), len(nodes_combined)),
                            len(nodes_query),
                        ).tolist(),
                        "value": cluster_mapping.to_numpy().flatten().tolist(),
                        "color": [mpl.colors.rgb2hex(m.to_rgba(val)) for val in cluster_distance.to_numpy().flatten()],
                        "label": [f"distance: {val:.2f}" for val in cluster_distance.to_numpy().flatten()],
                    },
                    threshold=filter_threshold,
                ),
            )
        ]
    )
    fig.update_layout(autosize=False, width=width, height=height)

    return fig
