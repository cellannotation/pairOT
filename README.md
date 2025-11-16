[//]: # (# pairOT)

<h1 align="center">
  <img src="docs/_static/pairOT-logo.png" alt="pairOT Logo" width="500"/>
</h1>

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/cellannotation/pairOT_package/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/pairOT_package

# pairOT: Identifying similar cell types and states across single-cell transcriptomic studies
Align cell annotations across two datasets through annotation-informed optimal transport.


## Getting started
Please refer to the [documentation][],
in particular, the [API documentation][].


## Installation
For detailed installation instructions, please refer to the [installation guide](docs/installation.md).


## Tutorial
Please take a look at the following tutorials for detailed examples on how to use pairOT:

### Detailed explanation
For a detailed tutorial, please see the [in depth tutorial](https://github.com/cellannotation/pairOT_package/blob/main/docs/notebooks/Tutorial.ipynb).

### Speed up pairOT computations
For details on how to speed up pairOT model fits and reduce compute requirements, see the [reduce compute requirements tutorial](https://github.com/cellannotation/pairOT_package/blob/main/docs/notebooks/Reduce%20Compute%20Requirements.ipynb)


## Basics of using pairOT

```python
import scanpy as sc
import pairot as pr

# 1. Preprocess input data
adata_query, adata_ref = pr.pp.preprocess_adatas(
    sc.read_h5ad("path/to/query.h5ad"),
    sc.read_h5ad("path/to/reference.h5ad"),
    n_top_genes=750,
    cell_type_column_adata1="cell_type_column_query",
    cell_type_column_adata2="cell_type_column_ref",
    sample_column_adata1="sequencing_sample_column_query",
    sample_column_adata2="sequencing_sample_column_ref",
)

# 2. Initialize pairOT model
dataset_map = pr.tl.DatasetMap(adata_query, adata_ref)
dataset_map.init_geom(batch_size=512, epsilon=0.05)
dataset_map.init_problem(tau_a=1.0, tau_b=1.0)

# 3. Fit pairOT model
dataset_map.solve()
mapping = dataset_map.compute_mapping()
distance = dataset_map.compute_distance()

# 4. Visualize results
pr.pl.mapping(mapping)  # similarity matrix
distance = distance.loc[
    mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
    mapping.max().sort_values(ascending=False).index.tolist(),
]  # order cluster distance matrix the same way as similarity matrix
pr.pl.distance(distance)  # cluster distance matrix
```

## Release notes
See the [changelog][].

## Contact
If you found a bug, please use the [issue tracker][].

## Citation
> t.b.a

## References
`pairOT` was written by `Felix Fischer <felix.fischer@helmholtz-munich.de>`

Support for software development, testing, modeling, and benchmarking provided by the Cell Annotation Platform team
(Roman Mukhin)

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/cellannotation/pairOT_package/issues
[tests]: https://github.com/cellannotation/pairot/actions/workflows/test.yaml
[documentation]: https://pairOT_package.readthedocs.io
[changelog]: https://pairOT_package.readthedocs.io/en/latest/changelog.html
[api documentation]: https://pairOT_package.readthedocs.io/en/latest/api.html
