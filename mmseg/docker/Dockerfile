ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"
ARG MMCV="1.3.12"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# Install MMSegmentation
ENV FORCE_CUDA="1"
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation --depth=1 \
    && cd /mmsegmentation/ \
    && pip install -r requirements.txt \
    && pip install --no-cache-dir -e . \
    && pip install opencv-python-headless \
    && cd / \
    && rm /mmsegmentation -r  # Avoid conflicts with mounted volumes later

# Compile SSN
COPY ssn /ssn
RUN cd /ssn \
    && python -c "import lib.ssn.ssn" \
    && rm /ssn -rf

RUN pip install yacs timm scipy \
    && mkdir /data \
    && chmod og+rwx /data \
    && chmod og+rwx -R /root/ \
    && mkdir -p /.local \
    && chmod og+rwx /.local \
    && pip install --upgrade azureml-sdk\
    && pip install --upgrade --disable-pip-version-check --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65 azureml_contrib_k8s