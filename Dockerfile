FROM nvcr.io/nvidia/jax:25.10-py3

# Install R and necessary libraries
RUN apt update -y
RUN apt install -y libtirpc-dev libpcre2-dev libbz2-dev liblzma-dev zlib1g-dev libicu-dev
RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" >> /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y r-base=4.3.3-2build2 r-base-dev=4.3.3-2build2

# Install pairOT
RUN pip install '.[dev,test]'
RUN pip install jupyterlab ipywidgets ipykernel

# Install R packages
RUN python3 -c "import rpy2.robjects as ro; \
INSTALL_R_PACKAGES = '''\
if (!require(\"BiocManager\", quietly = TRUE)) \
    install.packages(\"BiocManager\") \
BiocManager::install(\"limma\") \
\
install.packages(\"remotes\") \
library(remotes) \
install_version(\"rhdf5\", version = \"2.46.1\", repos = \"https://bioconductor.org/packages/3.18/bioc\") \
install_version(\"Matrix\", version = \"1.6-0\") \
install_version(\"magrittr\", version = \"2.0.3\") \
install_version(\"data.table\", version = \"1.15.4\") \
install_version(\"glue\", version = \"1.7.0\") \
install_version(\"stringr\", version = \"1.5.1\") \
'''; \
ro.r(INSTALL_R_PACKAGES)"
