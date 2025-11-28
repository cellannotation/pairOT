FROM nvcr.io/nvidia/jax:25.10-py3

# Install R and necessary libraries
RUN apt update -y
RUN apt install -y libtirpc-dev libpcre2-dev libbz2-dev liblzma-dev zlib1g-dev libicu-dev
RUN apt-get install -y libdeflate-dev libzstd-dev build-essential
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" >> /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y r-base r-base-dev r-recommended

# Install pairOT
COPY . /opt/pairOT
WORKDIR /opt/pairOT
RUN pip install ".[dev,test]"
RUN pip install jupyterlab ipywidgets ipykernel

# Install R packages
RUN Rscript /opt/pairOT/src/pairot/pp/resources/install_r_packages_latest.R

# Run tests to verify installation
RUN pytest /opt/pairOT/tests
