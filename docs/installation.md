# Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv](https://github.com/astral-sh/uv).

## Running pairOT via Docker

### Pre-built Docker image
To run pairOT, we provide a [docker image](https://hub.docker.com/r/felix0097/pairot/tags) that contains all the necessary dependencies:
```bash
docker pull felix0097/pairot:0.1
```

### Build Docker image from source
Alternatively, you can build your own Docker image as follows:
```bash
git clone https://github.com/cellannotation/pairOT.git
cd pairOT
docker build -t pairot:latest .
```

## Install pairOT via pip

### Install R
To run the R differential expression testing code (pre-processing), you'll need to install R on your system.
Make sure to install the latest R version otherwise you might run into compatibility issues with some R packages.

You can either do it via Anaconda:
```bash
conda install conda-forge::r-base
```
Or install R directly on your system.
Please refer to the official R documentation for [installation instructions](https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html#installing-r).
### Install pairOT via pip
```bash
pip install git+https://github.com/cellannotation/pairot.git@main
```

In case your `rpy2` installation fails, you might have to install the `libtirpc` development package on your system first.
```bash
sudo apt update
sudo apt install libtirpc-dev libpcre2-dev libbz2-dev liblzma-dev zlib1g-dev libicu-dev
```

By default, the installed JAX version only uses the CPU to make JAX recognize your GPU/TPU,
see [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#installation).
```bash
pip install -U "jax[cuda12]"
```

### Install R dependencies
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
import rpy2.robjects as ro

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

ro.r(INSTALL_R_PACKAGES)
```
