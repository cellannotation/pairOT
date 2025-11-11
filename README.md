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

Align cell annotations across two datasets through annotation-informed optimal transport.

- [Getting started](#getting-started)
- [Tutorial](#tutorial)
- [Basics of using pairOT](#basics-of-using-pairOT)
- [Installation](#installation)
  - [Running pairOT via Docker](#running-pairOT-via-docker)
  - [Install pairOT via pip](#install-pairOT-via-pip)
- [Citation](#citation)
- [References](#references)

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].


## Tutorial
Please take a look at the following tutorials for detailed examples on how to use pairOT:

### pairOT: Detailed explanation
For a detailed tutorial, please see the [in depth tutorial](https://github.com/cellannotation/pairOT_package/blob/main/docs/notebooks/Tutorial.ipynb).

### pairOT: Fit pairOT with reduced compute requirements / Speed up pairOT computations
For details on how to speed up pairOT model fits, see the [reduce compute requirements tutorial](https://github.com/cellannotation/pairOT_package/blob/main/docs/notebooks/Reduce%20Compute%20Requirements.ipynb)


## Basics of using pairOT

```python
import scanpy as sc

from pairot.pp import preprocess_adatas

# 1. Preprocess input data
adata_query, adata_ref = preprocess_adatas(
    sc.read_h5ad("path/to/query.h5ad"),
    sc.read_h5ad("path/to/reference.h5ad"),
    n_top_genes=750,
    cell_type_column_adata1="cell_type_column_query",
    cell_type_column_adata2="cell_type_column_ref",
    sample_column_adata1="sequencing_sample_column_query",
    sample_column_adata2="sequencing_sample_column_ref",
)

# 2. Initialize pairOT model
from pairot.align import DatasetMapping

dataset_map = DatasetMapping(adata_query, adata_ref)
dataset_map.init_geom(batch_size=512, epsilon=0.05)
dataset_map.init_problem(tau_a=1.0, tau_b=1.0)

# 3. Fit pairOT model
dataset_map.solve()
mapping = dataset_map.compute_cluster_mapping(aggregation_method="mean")
distance = dataset_map.compute_cluster_distances()

# 4. Visualize results
from pairot.pl import plot_cluster_mapping, plot_cluster_distance

plot_cluster_mapping(mapping)  # similarity matrix
distance = distance.loc[
    mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
    mapping.max().sort_values(ascending=False).index.tolist(),
]  # order cluster distance matrix the same way as similarity matrix
plot_cluster_distance(distance)  # cluster distance matrix
```


## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

### Running pairOT via Docker
To run pairOT, we provide a docker image that contains all the necessary dependencies: https://hub.docker.com/r/felix0097/pairot/tags
```bash
docker pull felix0097/pairot:full_v1
```

### Install pairOT via pip

#### Install R
To run the R differential expression testing code (pre-processing), you'll need to install R on your system.
Make sure to install the latest R version otherwise you might run into compatibility issues with some R packages.

You can either do it via Anaconda:
```bash
conda install conda-forge::r-base
```
Or install R directly on your system.
Please refer to the official R documentation for installation instructions: https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html#installing-r
#### Install pairOT via pip
```bash
pip install git+https://github.com/cellannotation/pairot.git@main
```

By default, the installed JAX version only uses the CPU to make JAX recognize your GPU/TPU,
see https://docs.jax.dev/en/latest/installation.html#installation
```bash
pip install -U "jax[cuda12]"
```

#### Install R dependencies
To install the required R dependencies, open a Python console and run the following commands:

```python
import rpy2.robjects as ro

INSTALL_R_PACKAGES_LATETST = """
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("limma")
BiocManager::install("rhdf5")
install.packages("Matrix")
install.packages("magrittr")
install.packages("data.table")
install.packages("glue")
install.packages("stringr")
"""

ro.r(INSTALL_R_PACKAGES_LATETST)
```
It might take a while to install all R dependencies.


**Note:** If you're using `R 4.3` you can install the following package versions:
```python
INSTALL_R_PACKAGES = """
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("limma")

install.packages("remotes")
library(remotes)
install_version("rhdf5", version = "2.46.1", repos = "https://bioconductor.org/packages/3.18/bioc")
install_version("Matrix", version = "1.6-0")
install_version("magrittr", version = "2.0.3")
install_version("data.table", version = "1.15.4")
install_version("glue", version = "1.7.0")
install_version("stringr", version = "1.5.1")
"""
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
