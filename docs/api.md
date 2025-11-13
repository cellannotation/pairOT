# API

## Preprocessing

```{eval-rst}
.. module:: pairot.pp
.. currentmodule:: pairot

.. autosummary::
    :toctree: generated

    pp.preprocess_adatas
    pp.downsample_indices
    pp.calc_pseudobulk_stats
    pp.select_and_combine_de_results
    pp.sort_and_filter_de_genes_ova
    pp.sort_and_filter_de_genes_ava
```

## Align

```{eval-rst}
.. module:: pairot.align
.. currentmodule:: pairot

.. autosummary::
    :toctree: generated

    align.DatasetMapping
```

## Plotting

```{eval-rst}
.. module:: pairot.pl
.. currentmodule:: pairot

.. autosummary::
    :toctree: generated

    pl.plot_cluster_mapping
    pl.plot_cluster_distance
    pl.plot_sankey
```

## Resources

```{eval-rst}
.. module:: pairot.pp
.. currentmodule:: pairot

.. autodata:: pp.OFFICIAL_GENES
   :annotation: pandas.DataFrame

   A DataFrame containing official gene names sourced from genenames.org (HGNC).
   The main column is ``feature_name`` with HGNC-approved gene symbols that are considered valid/official.

.. autodata:: pp.FILTERED_GENES
   :annotation: pandas.DataFrame

   A DataFrame listing uninformative genes to filter out during preprocessing,
   including mitochondrial, ribosomal, lncRNA, TCR, and BCR genes.
   The main column is ``feature_name``.
```
