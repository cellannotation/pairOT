import pickle
from os import PathLike
from typing import Literal

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import ott
import pandas as pd
import tqdm
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.utils import tqdm_progress_fn
from scipy.sparse import issparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata
from sklearn.utils.class_weight import compute_class_weight

from pairot.align._label_distance import _compute_label_distance_matrix
from pairot.pp import preprocess_adatas


def _cosine_distance(x: jnp.ndarray, y: jnp.ndarray):
    """Cosine distance between vectors, denominator regularized with ridge."""
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    cosine_similarity = jnp.vdot(x, y) / (x_norm * y_norm + 1e-8)
    return 0.5 * (1.0 - cosine_similarity)


def _compute_balanced_marginal(cell_types: pd.Series) -> np.ndarray:
    """Compute marginal balanced by cell cell_type."""
    assert isinstance(cell_types.dtype, pd.CategoricalDtype)

    classes = cell_types.cat.categories.to_numpy()
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=cell_types.to_numpy())
    marginal = weights[cell_types.cat.codes.to_numpy()]

    return marginal / marginal.sum()


def _convert_to_dense_numpy(arr):
    if issparse(arr):
        return arr.toarray().astype("f4")
    else:
        return arr.astype("f4")


def _get_category_mapping(series: pd.Series) -> dict[str, int]:
    return {v: k for k, v in enumerate(series.cat.categories)}


def _select_idxs(label_arr: np.ndarray, label: int, n: int | None = None) -> jnp.ndarray:
    idxs_label = np.where(label_arr == label)[0]
    if n is not None:
        idxs_label = np.random.choice(idxs_label, size=min(n, len(idxs_label)), replace=False)

    return jnp.array(idxs_label)


def _get_label_frequency(cell_type_labels: pd.Series, subset: np.ndarray) -> pd.Series:
    return cell_type_labels[cell_type_labels.isin(subset)].cat.remove_unused_categories().value_counts(normalize=True)


def _predict_from_marginals(labels_ref: pd.Series, marginal_contrib: dict[str, np.ndarray]) -> np.ndarray:
    labels, marginals = zip(*marginal_contrib.items(), strict=True)
    labels, marginals = np.array(labels), np.stack(marginals).T
    label_freq_ref = _get_label_frequency(labels_ref, labels).loc[labels].to_numpy()

    predictions = np.empty(marginals.shape, dtype=bool)
    for i, freq in enumerate(label_freq_ref):
        x = marginals[:, i]
        predictions[:, i] = x >= np.quantile(x, 1.0 - freq)
    # correct if one cell has been assigned more than once
    highest_diff_pred = labels[
        # use relative difference
        np.argmax(marginals / np.quantile(marginals, label_freq_ref), axis=1)
    ]
    correction_mask = np.sum(predictions, axis=1) != 1
    for i, _ in enumerate(labels):
        predictions[correction_mask, i] = highest_diff_pred[correction_mask] == i

    return labels[np.argmax(predictions, axis=1)]


@jax.tree_util.register_pytree_node_class
class _CellCellTransportCost(ott.geometry.costs.CostFn):
    """Cost function to calculate the cell to cell transport cost."""

    def __init__(
        self,
        label_distance_matrix,
        lambda_feature: float,
        lambda_label: float,
    ):
        super().__init__()
        self.label_distance = label_distance_matrix
        self.lambda_feature = lambda_feature
        self.lambda_label = lambda_label

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        x_, x_label = x[:-1], x[-1].astype(int)
        y_, y_label = y[:-1], y[-1].astype(int)

        return (
            self.lambda_feature * _cosine_distance(x_, y_) + self.lambda_label * self.label_distance[x_label, y_label]
        )

    def tree_flatten(self):
        return [], {
            "label_distance_matrix": self.label_distance,
            "lambda_feature": self.lambda_feature,
            "lambda_label": self.lambda_label,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(**aux_data)


class DatasetMapping:
    """
    Align cell annotations between query and reference dataset using annotation-informed optimal transport.

    Examples
    --------
    >>> import scanpy as sc
    >>>
    >>> from pairot.pp import preprocess_adatas
    >>>
    >>> # 1. Preprocess input data
    >>> adata_query, adata_ref = preprocess_adatas(
    >>>     sc.read_h5ad("path/to/query.h5ad"),
    >>>     sc.read_h5ad("path/to/reference.h5ad"),
    >>>     n_top_genes=750,
    >>>     cell_type_column_adata1="cell_type_column_query",
    >>>     cell_type_column_adata2="cell_type_column_ref",
    >>>     sample_column_adata1="sequencing_sample_column_query",
    >>>     sample_column_adata2="sequencing_sample_column_ref",
    >>> )
    >>>
    >>> # 2. Initialize pairOT model
    >>> from pairot.align import DatasetMapping
    >>>
    >>> dataset_map = DatasetMapping(adata_query, adata_ref)
    >>> dataset_map.init_geom(batch_size=512, epsilon=0.05)
    >>> dataset_map.init_problem(tau_a=1.0, tau_b=1.0)
    >>>
    >>> # 3. Fit pairOT model
    >>> dataset_map.solve()
    >>> mapping = dataset_map.compute_cluster_mapping(aggregation_method="mean")
    >>> distance = dataset_map.compute_cluster_distances()
    >>>
    >>> # 4. Visualize results
    >>> from pairot.pl import plot_cluster_mapping, plot_cluster_distance
    >>>
    >>> plot_cluster_mapping(mapping)  # similarity matrix
    >>> distance = distance.loc[
    >>>     mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
    >>>     mapping.max().sort_values(ascending=False).index.tolist(),
    >>> ]  # order cluster distance matrix the same way as similarity matrix
    >>> plot_cluster_distance(distance)  # cluster distance matrix
    """

    def __init__(self, adata1: anndata.AnnData, adata2: anndata.AnnData):
        self._validate_input(adata1, adata2)
        self.adata1 = adata1
        self.adata2 = adata2

        self.geom: pointcloud.PointCloud | None = None
        self._used_genes: dict[str, dict[str, pd.DataFrame]] | None = None
        self._label_distance: pd.DataFrame | None = None
        self.ot_prob: linear_problem.LinearProblem | None = None
        self.ot_solution: sinkhorn.SinkhornOutput | None = None

    @staticmethod
    def _validate_input(adata1: anndata.AnnData, adata2: anndata.AnnData):
        assert "cell_type_author" in adata1.obs.columns
        assert "cell_type_author" in adata2.obs.columns
        assert adata1.var.index.equals(adata2.var.index)
        for adata in [adata1, adata2]:
            assert adata.obs.cell_type_author.dtype == "category"
            assert np.allclose(
                np.unique(adata.obs.cell_type_author.cat.codes).astype("i8"),
                np.arange(len(adata.obs.cell_type_author.cat.categories), dtype="i8"),
            )
            # Check if DE test results for all clusters are present
            assert "de_res_ova" in adata.uns
            assert "de_res_ava" in adata.uns
            ct_clusters = adata.obs["cell_type_author"].unique()
            for ct in ct_clusters:
                # Check OVA DE results
                assert ct in adata.uns["de_res_ova"]
                for col in ["logFC", "adj.P.Val", "auroc"]:
                    assert col in adata.uns["de_res_ova"][ct].columns
                # Check AVA DE results
                for ct2 in ct_clusters:
                    if ct != ct2:
                        assert ct2 in adata.uns["de_res_ava"][ct]
                        for col in ["logFC", "adj.P.Val"]:
                            assert col in adata.uns["de_res_ava"][ct][ct2].columns

    @property
    def DEGs_label_distance_matrix(self):
        """Return DEGs (differentially expressed genes) based on which label distance matrix is computed."""
        self._assert_geom_initialized()
        return self._used_genes

    @property
    def label_distance_matrix(self):
        """Return label distance matrix between cell-type clusters."""
        self._assert_geom_initialized()
        return self._label_distance

    def _assert_geom_initialized(self):
        if self.geom is None:
            raise RuntimeError("Geometry is not initialized. Run .init_geom() first.")

    def _assert_prob_initialized(self):
        if self.ot_prob is None:
            raise RuntimeError("OT problem is not initialized. Run .init_problem() first.")

    def _assert_solution_initialized(self):
        if self.ot_solution is None:
            raise RuntimeError("OT solution is not calculated. Run self.solve() first.")

    def _assert_fully_initialized(self):
        self._assert_geom_initialized()
        self._assert_prob_initialized()
        self._assert_solution_initialized()

    @staticmethod
    def preprocess_adatas(
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        cell_type_column_adata1: str = "cell_type_author",
        cell_type_column_adata2: str = "cell_type_author",
        sample_column_adata1: str = "sample_id",
        sample_column_adata2: str = "sample_id",
        n_top_genes: int = 500,
        filter_genes: bool = False,
        n_samples_auroc: int = 10_000,
        n_samples_hvg_selection: int = 100_000,
    ) -> tuple[anndata.AnnData, anndata.AnnData]:
        """
        Function for pre-processing the input AnnData objects for usage with :class:`pairot.align.DatasetMapping`.

        Function applies the following preprocessing steps:
            1. Subset gene space to genes that are expressed in both datasets.
            2. Calculate differentially-expressed (DE) genes for each cluster.
            3. Subset to highly variable genes which are used to calculate the Spearman correlation between two cells.

        Parameters
        ----------
            adata1
                Query data.
            adata2
                Reference data.
            cell_type_column_adata1
                Name of the column in `adata.obs` that contains the cell type labels for adata1.
            cell_type_column_adata2
                Name of the column in `adata.obs` that contains the cell type labels for adata2.
            sample_column_adata1
                Name of the column in `adata.obs` that contains the sequencing sample ids/labels for adata1.
            sample_column_adata2
                Name of the column in `adata.obs` that contains the sequencing sample ids/labels for adata1.
            n_top_genes
                Number of highly variable genes to use to calculate the Spearman correlation between two cells.
            filter_genes
                Whether to remove uninformative genes. If true mitochondrial, ribosomal, IncRNA, TCR and BCR genes are removed.
            n_samples_auroc
                Maximum number of samples to use for AUROC calculation. If None, all samples are used. This can drastically reduce computation time for large datasets.
            n_samples_hvg_selection
                Number of samples to use for highly variable gene selection. If None, all samples are used. This can drastically reduce the memory usage for large datasets.

        Returns
        -------
            Preprocessed AnnData objects.
        """
        return preprocess_adatas(
            adata1,
            adata2,
            n_top_genes=n_top_genes,
            cell_type_column_adata1=cell_type_column_adata1,
            cell_type_column_adata2=cell_type_column_adata2,
            sample_column_adata1=sample_column_adata1,
            sample_column_adata2=sample_column_adata2,
            filter_genes=filter_genes,
            n_samples_auroc=n_samples_auroc,
            n_samples_hvg_selection=n_samples_hvg_selection,
        )

    def _compute_label_distances(
        self,
        n_genes_ova: int = 10,
        n_genes_ava: int = 3,
        overlap_threshold_ava: float = 0.3,
        overlap_n_genes_ava: int = 10,
        adj_p_val_threshold: float = 0.05,
        auroc_threshold: float = 0.6,
        logfc_threshold: float = 1.0,
        gene_filtering: bool = True,
        q_norm: float = 0.33,
    ):
        """Compute distance between clusters based on the overlap of differentially expressed genes."""
        adata1 = self.adata1
        adata2 = self.adata2

        self._label_distance, self._used_genes = _compute_label_distance_matrix(
            adata1,
            adata2,
            n_genes_ova=n_genes_ova,
            n_genes_ava=n_genes_ava,
            overlap_threshold=overlap_threshold_ava,
            overlap_n_genes=overlap_n_genes_ava,
            adj_p_val_threshold=adj_p_val_threshold,
            auroc_threshold=auroc_threshold,
            logfc_threshold=logfc_threshold,
            gene_filtering=gene_filtering,
            return_selected_genes=True,
            q_norm=q_norm,
        )
        label_distance_ordered = np.zeros(self._label_distance.shape)
        for i, label1 in enumerate(adata1.obs["cell_type_author"].cat.categories):
            for j, label2 in enumerate(adata2.obs["cell_type_author"].cat.categories):
                label_distance_ordered[i, j] = self._label_distance.loc[label1, label2]

        return jnp.array(label_distance_ordered.astype("f4"))

    def _preprocess_data(self, embedding_layer: str | None = None):
        adata1 = self.adata1
        adata2 = self.adata2
        if embedding_layer is None:
            x1 = _convert_to_dense_numpy(adata1.X)
            x2 = _convert_to_dense_numpy(adata2.X)
            # rank and zero center data that cosine similarity is equal to Spearman correlation
            x1 = rankdata(x1, axis=1)
            x2 = rankdata(x2, axis=1)
            x1 = x1 - x1.mean(axis=1, keepdims=True)
            x2 = x2 - x2.mean(axis=1, keepdims=True)
        else:
            x1 = _convert_to_dense_numpy(adata1.obsm[embedding_layer])
            x2 = _convert_to_dense_numpy(adata2.obsm[embedding_layer])
        y1 = adata1.obs.cell_type_author.cat.codes.to_numpy().astype("f4")
        y2 = adata2.obs.cell_type_author.cat.codes.to_numpy().astype("f4")
        x = np.hstack([x1, y1.reshape((-1, 1))])
        y = np.hstack([x2, y2.reshape((-1, 1))])

        return jnp.array(x), jnp.array(y)

    def init_geom(
        self,
        epsilon: float = 0.05,
        batch_size: int | None = 1024,
        lambda_feature: float = 0.5,
        lambda_label: float = 1.5,
        n_genes_ova: int = 10,
        n_genes_ava: int = 3,
        overlap_threshold_ava: float = 0.3,
        overlap_n_genes_ava: int = 10,
        adj_p_val_threshold: float = 0.05,
        auroc_threshold: float = 0.6,
        logfc_threshold: float = 1.0,
        gene_filtering: bool = True,
        q_norm: float = 0.33,
        embedding_layer: str | None = None,
        **kwargs,
    ):
        """
        Initialize the geometry of the optimal transport problem.

        Function calls the constructor of :class:`ott.geometry.pointcloud.PointCloud`.

        Parameters
        ----------
            epsilon
                Regularization strength of the optimal transport problem.
            batch_size
                Batch size used to solve the optimal transport problem in an online fashion.
                The bigger the batch size, the better the GPU utilization.
                However, bigger batch sizes lead to a higher GPU memory consumption.
            lambda_feature
                Weight for the distance in gene/feature space for the cell to cell transport cost.
            lambda_label
                Weight for the distance in label space for the cell to cell transport cost.
            n_genes_ova
                Number of top n differentially expressed (DE) genes in `adata1` used to calculate the rank distance between
                DE genes for the label distance. This setting applies to the one-vs-all (OVA) DE test results.
            n_genes_ava
                Number of top n differentially expressed (DE) genes in `adata1` used to calculate the rank distance between
                DE genes for the label distance. This setting applies to the all-vs-all (AVA) DE test results.
            overlap_threshold_ava
                Minimum overlap of the top `overlap_n_genes_ava` DE genes to add the all-vs-all (AVA) DE test results for
                the corresponding cell type label combination.
            overlap_n_genes_ava
                Number of top DE genes used to calculate the overlap of DE genes when deciding which all-vs-all (AVA) DE
                results to include.
            adj_p_val_threshold
                Minimum adjusted p-value to consider a gene as differentially expressed.
            auroc_threshold
                Minimum AUROC score to consider a gene as differentially expressed.
            logfc_threshold
                Minimum log fold change to consider a gene as differentially expressed.
            gene_filtering
                Whether to filter DE gene results. If true mitochondrial, ribosomal, IncRNA, TCR and BCR genes are removed
                from the DE results.
            q_norm
                Quantile used to normalize label distance matrix.
            embedding_layer
                Name of the embedding layer in `adata1.obsm` and `adata1.obsm` used to calculate the distance between
                two cells.
                If this parameter is provided, the distance between two cells is calculated as the cosine distance in
                embedding space instead of the Spearman correlation in the full gene space.
            kwargs
                Keyword arguments passed to :class:`ott.geometry.pointcloud.PointCloud`.
        """
        x, y = self._preprocess_data(embedding_layer=embedding_layer)
        label_distance = self._compute_label_distances(
            n_genes_ova=n_genes_ova,
            n_genes_ava=n_genes_ava,
            overlap_threshold_ava=overlap_threshold_ava,
            overlap_n_genes_ava=overlap_n_genes_ava,
            adj_p_val_threshold=adj_p_val_threshold,
            auroc_threshold=auroc_threshold,
            logfc_threshold=logfc_threshold,
            gene_filtering=gene_filtering,
            q_norm=q_norm,
        )
        self.geom = pointcloud.PointCloud(
            x,
            y,
            epsilon=epsilon,
            batch_size=batch_size,
            cost_fn=_CellCellTransportCost(label_distance, lambda_feature=lambda_feature, lambda_label=lambda_label),
            **kwargs,
        )

    def init_problem(
        self,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        marginals_distribution: Literal["uniform", "balanced"] = "balanced",
        **kwargs,
    ):
        """
        Initialize the optimal transport problem.

        Function calls the constructor of :class:`ott.problems.linear.linear_problem.LinearProblem`.

        Parameters
        ----------
            tau_a
                If < 1, defines how much unbalanced the problem is on the first marginal.
            tau_b
                If < 1, defines how much unbalanced the problem is on the second marginal.
            marginals_distribution
                Whether the marginals should be uniform or balanced by cell-type frequency.
                Use "uniform" for uniform marginals. Meaning, each cell contributes the same mass to the marginal distribution.
                Use "balanced" for marginals balanced by cell-type frequency. Meaning, each cell-type contributes the same mass to the marginal distribution.
                This parameter is ignored, if marginals 'a' or 'b' are supplied via the **kwargs.
            kwargs
                Keyword arguments passed to :class:`ott.problems.linear.linear_problem.LinearProblem`.
        """
        self._assert_geom_initialized()

        if marginals_distribution == "uniform":
            a, b = None, None
        elif marginals_distribution == "balanced":
            a = _compute_balanced_marginal(self.adata1.obs["cell_type_author"])
            b = _compute_balanced_marginal(self.adata2.obs["cell_type_author"])
        else:
            raise ValueError(
                f"marginals argument must be either 'uniform' or 'balanced'. You provided: {marginals_distribution}"
            )

        if ("a" not in kwargs) and ("b" not in kwargs):
            assert np.all(np.array(a) > 0.0)
            assert np.all(np.array(b) > 0.0)
            self.ot_prob = linear_problem.LinearProblem(self.geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b, **kwargs)
        elif "a" in kwargs:
            assert np.all(np.array(b) > 0.0)
            self.ot_prob = linear_problem.LinearProblem(self.geom, b=b, tau_a=tau_a, tau_b=tau_b, **kwargs)
        elif "b" in kwargs:
            assert np.all(np.array(a) > 0.0)
            self.ot_prob = linear_problem.LinearProblem(self.geom, a=a, tau_a=tau_a, tau_b=tau_b, **kwargs)

    def solve(self, **kwargs):
        """
        Solve the underlying optimal transport problem.

        Function uses :class:`ott.solvers.linear.sinkhorn.Sinkhorn` to solve the optimal transport problem.

        Parameters
        ----------
            kwargs
                Keyword arguments passed to :class:`ott.solvers.linear.sinkhorn.Sinkhorn`.
        """
        self._assert_geom_initialized()
        self._assert_prob_initialized()

        with tqdm.tqdm() as pbar:
            progress_fn = tqdm_progress_fn(pbar)
            # noinspection PyTypeChecker
            solver = sinkhorn.Sinkhorn(progress_fn=progress_fn, **kwargs)
            self.ot_solution = jax.jit(solver)(self.ot_prob)

    def _get_label_vectors(self):
        assert self.geom is not None
        return tuple(np.array(arr[:, -1]).astype("i8") for arr in [self.geom.x, self.geom.y])

    def compute_cluster_mapping(
        self,
        aggregation_method: Literal["mean", "jensen_shannon", "transported_mass"] | None = "mean",
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Compute the mapping between cell-type clusters based on the aggregated transport matrix.

        Aggregation is done by cluster/cell-type.

        Parameters
        ----------
            aggregation_method
                Method used to aggregate the transport map between cell-type clusters.

        Returns
        -------
            Mappings between cell-type clusters of the query and reference datasets.
        """
        self._assert_fully_initialized()

        b = np.array(self.ot_prob.b).astype("f8")
        x_label_np, y_label_np = self._get_label_vectors()
        unique_labels_x, unique_labels_y = np.unique(x_label_np), np.unique(y_label_np)
        transport_map_agg = {
            agg: np.zeros((len(unique_labels_x), len(unique_labels_y)))
            for agg in ["mean", "jensen_shannon", "transported_mass"]
        }
        for label_x in tqdm.tqdm(unique_labels_x):
            transported_mass = np.array(self.ot_solution.apply(jnp.array(x_label_np == label_x))).astype("f8")
            # Aggregate contribution to target marginals via mean
            for label_y in unique_labels_y:
                transport_map_agg["mean"][label_x, label_y] = np.mean((transported_mass / b)[y_label_np == label_y])
            # Aggregate contribution to target marginals via Jensen-Shannon divergence
            for label_y in unique_labels_y:
                jensen_shannon = jensenshannon(p=transported_mass / b, q=y_label_np == label_y, base=2)
                transport_map_agg["jensen_shannon"][label_x, label_y] = -jensen_shannon + 1.0
            # Aggregate via fraction of total mass transported
            total_mass = float(self.ot_prob.a[x_label_np == label_x].sum())
            for label_y in unique_labels_y:
                transport_map_agg["transported_mass"][label_x, label_y] = (
                    np.sum(transported_mass[y_label_np == label_y]) / total_mass
                )

        transport_map_agg = {
            k: pd.DataFrame(
                results,
                index=self.adata1.obs.cell_type_author.cat.categories,
                columns=self.adata2.obs.cell_type_author.cat.categories,
            )
            for k, results in transport_map_agg.items()
        }
        if aggregation_method:
            return transport_map_agg[aggregation_method]
        else:
            return transport_map_agg

    def compute_cluster_distances(self, n_samples: int = 25000) -> pd.DataFrame:
        """
        Compute the distance between cell-type clusters based on the optimal transport mappings.

        Parameters
        ----------
            n_samples
                The number of samples based on which the distance is calculated.

        Returns
        -------
            Distances between cell-type clusters of the query and reference datasets.
        """
        self._assert_fully_initialized()

        x_label_np, y_label_np = self._get_label_vectors()
        unique_labels_x, unique_labels_y = np.unique(x_label_np), np.unique(y_label_np)
        with tqdm.tqdm(total=len(unique_labels_x) * len(unique_labels_y)) as pbar:
            weighted_cost = np.zeros((len(unique_labels_x), len(unique_labels_y)))
            for label_x in unique_labels_x:
                for label_y in unique_labels_y:
                    idxs_label_x = _select_idxs(x_label_np, label_x, n_samples)
                    idxs_label_y = _select_idxs(y_label_np, label_y, n_samples)
                    geom_subset = self.geom.subset(idxs_label_x, idxs_label_y)
                    transport = geom_subset.transport_from_potentials(
                        self.ot_solution.f[idxs_label_x],
                        self.ot_solution.g[idxs_label_y],
                    )
                    cost = self.geom.cost_fn.all_pairs(self.geom.x[idxs_label_x, :], self.geom.y[idxs_label_y, :])
                    total_transport_mass = jnp.sum(transport)
                    if total_transport_mass > 1e-16:
                        transport /= total_transport_mass
                        weighted_cost[label_x, label_y] = jnp.sum(cost * transport)
                    else:
                        # if no mass is being transported -> take average cost without weighting
                        weighted_cost[label_x, label_y] = jnp.mean(cost)
                    pbar.update()

        return pd.DataFrame(
            weighted_cost,
            index=self.adata1.obs.cell_type_author.cat.categories,
            columns=self.adata2.obs.cell_type_author.cat.categories,
        )

    @staticmethod
    def select_most_similar_clusters(
        mapping: pd.DataFrame,
        distance: pd.DataFrame,
        threshold_mapping: float | None = 0.25,
        threshold_distance: float | None = 1.0,
        n_top: int | None = None,
    ) -> dict[str, list[str]]:
        """
        Select the `n_top` most similar cell-type clusters in the reference data for each cell-type cluster in the query data based on the aggregated transport matrix.

        Parameters
        ----------
            mapping
                The cluster mapping matrix / aggregated transport matrix. The output of self.compute_cluster_mapping().
            distance
                The cluster distance matrix. The output of self.compute_cluster_distances().
            threshold_mapping
                The minimum transported mass between two cell-type clusters for a cluster to be suggested as a most similar
                cell-type cluster.
                If set to `None`, no filtering will be applied.
            threshold_distance
                The maximum distance between two cell-type clusters for a cluster to be suggested as a most similar
                cell-type cluster.
                If set to `None`, no filtering will be applied.
            n_top
                Maximum number of most similar clusters to return.
                If set to `None`, all cell-type clusters will be returned.

        Returns
        -------
            A dictionary mapping each cell-type cluster in the query data to a list of most similar cell type clusters in the reference data.
        """
        if threshold_mapping is None:
            threshold_mapping = 0.0  # Don't do filtering if no threshold is provided
        if threshold_distance is None:
            threshold_distance = 2.0  # Don't do filtering if no threshold is provided

        top_n_labels = {}
        labels_query = mapping.index.tolist()
        labels_ref = mapping.columns
        for label in labels_query:
            distance_ = distance.loc[label, :].to_numpy()
            mapping_ = mapping.loc[label, :].to_numpy()
            suggestions = np.argsort(-mapping_)[:n_top]
            top_n_labels[label] = [
                labels_ref[s]
                for s in suggestions
                if distance_[s] <= threshold_distance and mapping_[s] >= threshold_mapping
            ]

        return top_n_labels

    def pickle_state(self, path: PathLike | str):
        """Pickle the state of the OT model."""
        self._assert_fully_initialized()
        with open(path, "wb") as f:
            pickle.dump(
                (self.geom, self.ot_prob, self.ot_solution),
                f,
            )

    def load_state(self, path: PathLike | str):
        """Load the pickled state of the OT model."""
        with open(path, "rb") as f:
            self.geom, self.ot_prob, self.ot_solution = pickle.load(f)

    def pickle_ot_solution(self, path: PathLike | str):
        """Pickle the solution of the OT model."""
        self._assert_solution_initialized()
        with open(path, "wb") as f:
            pickle.dump(self.ot_solution, f)

    def load_ot_solution(self, path: PathLike | str):
        """Load the solution of the OT model."""
        with open(path, "rb") as f:
            self.ot_solution = pickle.load(f)
