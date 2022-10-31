ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"
ARG MMCV="1.3.12"

ARG BASE_IMAGE=nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

# ===================================================

FROM ${BASE_IMAGE} as openmpi
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    openssh-client \
    libjpeg-dev \
    libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

RUN wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    gunzip -c openmpi-4.0.0.tar.gz | tar xf -
RUN    cd openmpi-4.0.0 && \
    ./configure --prefix=/opt/openmpi --with-cuda
RUN mkdir -p /opt/openmpi && cd openmpi-4.0.0 && make all install -j$(nproc) || :

# ===================================================

FROM openmpi as torch
ARG PYTHON_VERSION=3.7

ENV PATH="/opt/openmpi/bin:$PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/openmpi/lib/"

RUN ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda update conda && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
WORKDIR /opt/pytorch

RUN git clone --recursive https://github.com/pytorch/pytorch --depth 1 --branch v1.8.1

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 7.5" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pwd && ls && cd pytorch && \
    pip install -v .
RUN pip install torchvision

# ===================================================

FROM ${BASE_IMAGE} as mmseg-ssn
RUN ls /opt
COPY --from=torch /opt/openmpi /opt/openmpi
COPY --from=torch /opt/conda /opt/conda

ENV PATH="/opt/openmpi/bin:/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/openmpi/lib/"
ARG MMCV="1.3.12"
ARG PYTORCH="1.8.1"
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# Install MMSegmentation
ENV FORCE_CUDA="1"
RUN apt update -y && apt install git -y
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation --depth=1 \
    && cd /mmsegmentation/ \
    && pip install -r requirements.txt \
    && pip install --no-cache-dir -e . \
    && pip install opencv-python-headless \
    && cd / \
    && rm /mmsegmentation -r  
# Avoid conflicts with mounted volumes later

# Compile SSN
RUN apt install ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 -y
COPY ssn /ssn
RUN cd /ssn \
    && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 7.5" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python -c "import lib.ssn.ssn" \
    && rm /ssn -rf


RUN pip install yacs \
    && mkdir /data \
    && chmod og+rwx /data \
    && chmod og+rwx -R /root/ \
    && mkdir -p /.local \
    && chmod og+rwx /.local \
    && pip install --upgrade azureml-sdk\
    && pip install --upgrade --disable-pip-version-check --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65 azureml_contrib_k8s